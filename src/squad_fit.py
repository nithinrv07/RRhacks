"""
SupeRR Selector S2 - Step 7: how the top-10 picks could strengthen Rajasthan Royals (team_id 3).

Output: outputs/rr_pressure_profile.csv   RR's pressure-ball contribution by phase vs the other IPL sides
        outputs/rr_player_fit.csv         each pick: best pressure phase, latest IPL side, availability note
Usage : python src/squad_fit.py

RR's numbers use the same per-ball contributions as the rating (vs the average IPL side in the same
situation), so 0 = league average and every IPL side is ranked on the same footing.
Country = the national side a player appeared for in the data pack (no outside information).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import rating as R
from metrics import COLS

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
RR = 3

def national_side():
    """Country = the senior national side a player appeared for in the data pack (A teams map to their country).
    Players with no international appearances are labelled uncapped."""
    pe = pd.read_csv(DATA / "playing_eleven.csv", usecols=["match_id", "team_id", "player_id"])
    m = pd.read_csv(DATA / "matches.csv", usecols=["match_id", "comp_id"])
    c = pd.read_csv(DATA / "competitions.csv")
    t = pd.read_csv(DATA / "teams.csv").drop_duplicates("team_id").set_index("team_id").team_name
    x = pe.merge(m, on="match_id").merge(c[["comp_id", "comp_name"]], on="comp_id")
    x = x[~x.comp_name.str.startswith("IPL")]
    x["country"] = x.team_id.map(t).str.replace(r"\s+A$", "", regex=True)
    return x.groupby("player_id").country.agg(lambda s: s.mode().iat[0])


# fallback only for picks with no international matches in 2022-26 data; labelled "(public info)" in the report
PUBLIC_COUNTRY = {"Sunil Narine": "West Indies", "Krunal Pandya": "India", "Mohsin Khan": "India"}


def availability(country):
    country = country.replace(" (public info)", "")
    if country == "India":
        return "Indian player: no overseas slot needed"
    if country == "uncapped":
        return "No international matches in the data: uncapped (an overseas slot is needed only if not Indian)"
    return f"Overseas ({country}): uses one of the four overseas places in the XI"


def load():
    parts = []
    for ch in pd.read_csv(OUT / "balls_wp.csv.gz", usecols=COLS + ["elo", "year", "batting_team_id",
                                                                    "bowling_team_id"], chunksize=40000):
        parts.append(ch[~ch.exclude.astype(bool) & ch.leverage.notna()])
    b = pd.concat(parts, ignore_index=True)
    b["pressure"] = b.leverage >= 1.5
    return b


def team_profile(bat, bowl):
    """Per IPL side: mean contribution per 100 pressure balls, by phase, batting and bowling."""
    rows = []
    for side, d, team_col, mets in (("batting", bat, "batting_team_id", {"Runs above expected": "c_rae",
                                                                           "Survival": "c_surv", "WPA": "c_wpa"}),
                                    ("bowling", bowl, "bowling_team_id", {"Runs saved": "c_rsave", "Dots": "c_dot",
                                                                          "WPA": "c_wpa"})):
        p = d[d.pressure & (d.fmt == "IPL")]
        for phase in ("powerplay", "middle", "death"):
            q = p[p.phase == phase]
            g = q.groupby(team_col)
            for m, c in mets.items():
                val = g[c].mean() * 100
                n = g.size()
                val = val[n >= 300]
                rk = val.rank(ascending=False, method="min")
                if RR in val.index:
                    rows.append({"side": side, "phase": phase, "metric": m, "RR": val[RR],
                                 "rank_of": f"{int(rk[RR])} of {len(val)}", "RR_balls": int(n[RR])})
    return pd.DataFrame(rows)


def latest_ipl_team(names):
    pe = pd.read_csv(DATA / "playing_eleven.csv", usecols=["match_id", "team_id", "player_id"])
    m = pd.read_csv(DATA / "matches.csv", usecols=["match_id", "comp_id", "match_date"])
    c = pd.read_csv(DATA / "competitions.csv")
    t = pd.read_csv(DATA / "teams.csv").drop_duplicates("team_id").set_index("team_id").team_name
    x = pe.merge(m, on="match_id").merge(c[["comp_id", "comp_name", "season"]], on="comp_id")
    x = x[x.comp_name.str.startswith("IPL")].sort_values("match_date")
    last = x.groupby("player_id").last()
    rr_games = x[x.team_id == RR].groupby("player_id").size()
    return last.assign(team=last.team_id.map(t)), rr_games


def best_phase(d, pid_col, pid, c, min_balls=30):
    q = d[d.pressure & (d[pid_col] == pid)]
    g = q.groupby("phase")[c].agg(["mean", "size"])
    g = g[g["size"] >= min_balls]
    if g.empty:
        return "–", np.nan, 0
    ph = g["mean"].idxmax()
    return ph, g.loc[ph, "mean"] * 100, int(g.loc[ph, "size"])


def main():
    b = load()
    bat, bowl = R.per_ball(b)
    prof = team_profile(bat, bowl)
    prof.to_csv(OUT / "rr_pressure_profile.csv", index=False)

    rb = pd.read_csv(OUT / "rating_batters.csv").head(10)
    rw = pd.read_csv(OUT / "rating_bowlers.csv").head(10)
    last, rr_games = latest_ipl_team(None)
    global NAT
    NAT = national_side()
    rows = []
    for kind, r, d, pid_col, c in (("Batter", rb, bat, "striker_id", "c_rae"), ("Bowler", rw, bowl, "bowler_id",
                                                                                "c_rsave")):
        for _, p in r.iterrows():
            ph, val, n = best_phase(d, pid_col, p.player_id, c)
            ipl = last.loc[p.player_id] if p.player_id in last.index else None
            country = NAT.get(p.player_id)
            if country is None and p["name"] in PUBLIC_COUNTRY:
                country = PUBLIC_COUNTRY[p["name"]] + " (public info)"
            country = country or "uncapped"
            rows.append({"kind": kind, "rank": int(p["rank"]), "name": p["name"], "role": p.role,
                         "best_pressure_phase": ph,
                         "best_phase_value": val, "best_phase_balls": n,
                         "latest_ipl_team": (f"{ipl.team} ({int(ipl.season)})" if ipl is not None else "no IPL games"),
                         "rr_games": int(rr_games.get(p.player_id, 0)), "country": country,
                         "availability": availability(country)})
    fit = pd.DataFrame(rows)
    fit.to_csv(OUT / "rr_player_fit.csv", index=False)

    pd.set_option("display.width", 250)
    print(prof.round(2).to_string(index=False))
    print(fit.drop(columns=["availability"]).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
