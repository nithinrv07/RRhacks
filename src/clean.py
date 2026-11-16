"""
SupeRR Selector S2 - Step 1: build a clean ball-state table.

Output: outputs/balls_clean.csv.gz  (one row per delivery, innings 1-2 only)

Data-pack quirks handled here (verified by inspection):
  * extras1_runs is an extras TYPE code, not a run amount
    (1=no-ball/Overstep, 2=wide, 3=leg-bye, 4=bye, 5=other, 6=none, 7=penalty).
  * The pack does not record how many runs a wide / bye / leg-bye produced,
    so each extra is counted as 1 run (+ bat runs on no-balls). Totals are
    therefore approximate -> measured in validate.py; the chase target is
    reconciled to the official result (see build()).
  * 'other' extras behave like no-balls (overs containing them have 7
    legal-counted balls), so they are treated as illegal deliveries.
  * Innings 3/4 are super overs -> dropped.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

BOWLER_WICKETS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}
ILLEGAL = {"wide", "Overstep", "other"}


def load():
    rd = lambda f: pd.read_csv(DATA / f, low_memory=False, keep_default_na=False, na_values=[""])
    return rd("ball_by_ball.csv"), rd("matches.csv"), rd("competitions.csv")


def build():
    b, m, c = load()
    b = b[b.innings.isin([1, 2])].copy()

    # ---- match context ----
    # "T20" in competitions.csv covers the IPL AND two A-team tournaments (Emerging Teams Asia Cup 2024,
    # Asia Cup Rising Stars 2025), so IPL is identified by competition name. A-team matches join the T20I group.
    c["fmt"] = np.where(c.comp_name.str.startswith("IPL"), "IPL", "T20I")
    m = m.merge(c[["comp_id", "comp_name", "fmt", "season"]], on="comp_id", how="left")
    m["revised"] = (m.first_inn_revised_overs.notna() | m.second_inn_revised_overs.notna()
                    | m.result.str.contains(r"D/L|DLS", case=False, na=False))
    m["no_result"] = m.result.str.contains("No Result|Abandon", case=False, na=False)
    m["tie"] = m.result.str.contains("Tie|Super Over", case=False, na=False)
    m["year"] = pd.to_datetime(m.match_date).dt.year
    b = b.merge(m[["match_id", "fmt", "comp_name", "year", "stage", "knock_out",
                   "win_team_id", "revised", "no_result", "tie", "result"]],
                on="match_id", how="left")

    # ---- per-ball accounting ----
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    et = b.extras_type
    b["is_wide"] = et.eq("wide")
    b["is_noball"] = et.isin(["Overstep", "other"])
    b["legal"] = ~et.isin(ILLEGAL)

    # Impute missing extra runs from strike rotation: on normal balls, batters
    # crossing <=> odd runs (100% in this data). Crossed = next ball's striker
    # is this ball's non-striker (or same striker if the over ended).
    gi = b.groupby(["match_id", "innings"], sort=False)
    nxt = gi.striker_id.shift(-1)
    over_end = gi.overid.shift(-1).ne(b.overid)
    crossed = np.where(over_end, nxt.eq(b.striker_id), nxt.eq(b.non_striker_id))
    bye = et.isin(["byes", "legbyes"])
    b["extra_runs"] = np.select(
        [et.eq("None"), bye & crossed, bye, et.eq("wide") & crossed, et.eq("wide")],
        [0, 1, 4, 2, 1], default=1)   # uncrossed byes -> 4 (best fit vs official margins)
    b["total_runs"] = b.runs + b.extra_runs
    b["bat_runs"] = np.where(et.isin(["byes", "legbyes"]), 0, b.runs)
    b["ball_faced"] = ~b.is_wide                                    # no-balls count as faced
    # bowler is charged bat runs + wides + no-balls, not byes / leg-byes
    b["bowler_runs"] = b.bat_runs + np.where(b.is_wide | b.is_noball, b.extra_runs, 0)
    b["bowler_wkt"] = b.is_wicket & b.dismissal_type.isin(BOWLER_WICKETS)
    b["is_dot"] = b.legal & (b.total_runs == 0)
    b["is_boundary"] = b.is_four | b.is_six

    # ---- running state BEFORE the ball ----
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    g = b.groupby(["match_id", "innings"], sort=False)
    b["score_before"] = g.total_runs.cumsum() - b.total_runs
    b["wkts_before"] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
    b["legal_before"] = g.legal.cumsum() - b.legal.astype(int)
    b["balls_left"] = (120 - b.legal_before).clip(lower=0)
    b["over_no"] = b.legal_before // 6 + 1                           # 1-20
    b["phase"] = pd.cut(b.over_no, [0, 6, 15, 99], labels=["powerplay", "middle", "death"])
    b["crr"] = np.where(b.legal_before > 0, b.score_before * 6 / b.legal_before.clip(lower=1), np.nan)

    # ---- innings totals / chase state ----
    tot = b.groupby(["match_id", "innings"]).agg(
        inn_total=("total_runs", "sum"), inn_wkts=("is_wicket", "sum"),
        inn_legal=("legal", "sum"), bat_team=("batting_team_id", "first")).reset_index()
    first = tot[tot.innings == 1].set_index("match_id")
    b["target_raw"] = np.where(b.innings == 2, b.match_id.map(first.inn_total) + 1, np.nan)
    # Reconcile the chase target with the official result. Missing extras-runs make the rebuilt
    # totals drift, so in ~11% of chases the rebuilt score contradicts who actually won.
    #  * chasing side won but rebuilt chase total < target   -> target = rebuilt chase total
    #  * batting-first side won by N runs                    -> target = rebuilt chase total + N + 1
    #  * tie                                                  -> target = rebuilt chase total + 1
    second = tot[tot.innings == 2].set_index("match_id")
    res = m.set_index("match_id")
    margin = res.result.str.extract(r"by (\d+) [Rr]un", expand=False).astype(float)
    chase_won = second.bat_team.eq(res.win_team_id.reindex(second.index))
    tgt = (first.inn_total + 1).reindex(second.index)
    ct = second.inn_total
    tie = res.tie.reindex(second.index).fillna(False)
    mg = margin.reindex(second.index)
    tgt = np.where(tie, ct + 1,
          np.where(chase_won & (ct < tgt), ct,
          np.where(~chase_won & mg.notna(), ct + mg + 1, tgt)))
    tgt = pd.Series(tgt, index=second.index)
    b["target"] = np.where(b.innings == 2, b.match_id.map(tgt), np.nan)
    b["runs_needed"] = b.target - b.score_before
    b["rrr"] = np.where((b.innings == 2) & (b.balls_left > 0),
                        b.runs_needed * 6 / b.balls_left.clip(lower=1), np.nan)
    b["bat_team_won"] = b.batting_team_id.eq(b.win_team_id)

    # flag matches unusable for situation modelling
    # incomplete ball-by-ball: an innings that stops early with wickets in hand
    # (1st innings short of 20 overs, or a chase that stops short of the target)
    t = tot.set_index(["match_id", "innings"])
    t["tgt"] = t.index.get_level_values(0).map(first.inn_total + 1)
    inn_no = t.index.get_level_values(1)
    won = t.bat_team.eq(t.index.get_level_values(0).map(m.set_index("match_id").win_team_id))
    # 1st innings: well short of 20 overs with wickets in hand (truncated record or unflagged reduced match)
    inc1 = (inn_no == 1) & (t.inn_legal <= 110) & (t.inn_wkts < 10)
    # chase: stopped early with wickets in hand but did NOT win (9 down allowed: absent / retired batter)
    inc2 = (inn_no == 2) & (t.inn_legal < 114) & (t.inn_wkts < 9) & ~won
    incomplete = set(t[inc1 | inc2].index.get_level_values(0))
    b["incomplete"] = b.match_id.isin(incomplete)
    b["exclude"] = b.revised | b.no_result | b.win_team_id.isna() | b.incomplete

    keep = ["match_id", "fmt", "comp_name", "year", "stage", "knock_out", "innings",
            "innings_ball_num", "over_no", "phase", "batting_team_id", "bowling_team_id",
            "striker_id", "striker_name", "non_striker_id", "bowler_id", "bowler_name",
            "batting_order", "batting_type", "bowling_type", "pace_or_spin",
            "score_before", "wkts_before", "legal_before", "balls_left", "crr",
            "target", "target_raw", "runs_needed", "rrr",
            "runs", "bat_runs", "extra_runs", "total_runs", "bowler_runs", "legal",
            "ball_faced", "is_wide", "is_noball", "is_dot", "is_four", "is_six",
            "is_boundary", "is_wicket", "bowler_wkt", "dismissal_type",
            "dismissed_player_id", "bat_team_won", "tie", "incomplete", "exclude", "result"]
    b[keep].to_csv(OUT / "balls_clean.csv.gz", index=False)
    tot.to_csv(OUT / "innings_totals.csv", index=False)
    return b[keep], tot


if __name__ == "__main__":
    b, tot = build()
    print(f"rows={len(b):,} matches={b.match_id.nunique()} "
          f"excluded matches={b[b.exclude].match_id.nunique()} "
          f"(incomplete-record matches: {b[b.incomplete].match_id.nunique()})")
    print(b.groupby("fmt").match_id.nunique())
