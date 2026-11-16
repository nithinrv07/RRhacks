"""
SupeRR Selector S2 - Step 3: player performance metrics under pressure.

Input : outputs/balls_wp.csv.gz (from winprob.py)
Output: outputs/batting_metrics.csv, outputs/bowling_metrics.csv, outputs/metrics_summary.txt
Usage : python src/metrics.py [--threshold 1.5] [--min-balls 120]

Every number is relative to what an AVERAGE player would do in the SAME situation:
  situation = format x innings x phase x wickets-lost band x required-rate band (chase only)
so a death-overs bowler is not punished for death-overs economy, and IPL and T20I each have their own
baseline. Thin situation cells are shrunk toward the coarser format x innings x phase average.

Batting (balls faced; wides are not faced)
  rae_100     runs above expected per 100 balls (runs off the bat)
  oae_100     dismissals above expected per 100 balls (negative = survives better than average)
  wpa_100     win probability added per 100 balls, in percentage points (WPA already prices runs vs wickets)
Bowling (all deliveries bowled)
  rsave_100   runs saved vs expected per 100 balls (bowler-charged runs; byes/leg-byes excluded)
  wae_100     bowler wickets above expected per 100 balls (run-outs excluded)
  dotae_pct   dot-ball rate above expected, percentage points
  wpa_100     win probability taken from the batting side per 100 balls, in percentage points
Each metric is computed for pressure balls (leverage >= threshold) and for the player's own ordinary balls;
drop_* = pressure minus ordinary ("does the player hold their level when it matters?").

NOTE: these are RAW numbers. Small samples are not yet shrunk (that is Step 4), so the provisional
leaderboard printed at the end is for sanity-checking only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
COLS = ["match_id", "fmt", "innings", "phase", "wkts_before", "rrr", "striker_id", "striker_name",
        "bowler_id", "bowler_name", "batting_order", "pace_or_spin", "bowling_type", "bat_runs",
        "bowler_runs", "legal", "ball_faced", "is_wicket", "dismissed_player_id", "bowler_wkt",
        "total_runs", "exclude", "wpa", "leverage", "elo"]


def situation_keys(b):
    wband = pd.cut(b.wkts_before, [-1, 2, 5, 10], labels=["0-2", "3-5", "6+"]).astype(str)
    rband = np.where(b.innings == 2, pd.cut(b.rrr, [-1, 6, 8, 10, 12, 15, 999]).astype(str), "na")
    coarse = b.fmt + "|" + b.innings.astype(str) + "|" + b.phase.astype(str)
    if "pressure" in b:   # pressure balls are compared with the average PRESSURE ball in that situation
        coarse = coarse + "|" + b.pressure.astype(str)
    return coarse + "|" + wband + "|" + rband, coarse


def expected(b, value, mask, prior=300.0):
    """Situation-average of `value` over rows in `mask`, shrunk toward the coarse cell."""
    fine, coarse = situation_keys(b)
    v = pd.Series(np.where(mask, value, np.nan), index=b.index)
    f = v.groupby(fine).agg(["sum", "count"])
    c = v.groupby(coarse).agg(["sum", "count"])
    c_mean = (c["sum"] / c["count"]).reindex(coarse).to_numpy()
    f_sum = f["sum"].reindex(fine).to_numpy()
    f_n = f["count"].reindex(fine).to_numpy()
    return (f_sum + prior * c_mean) / (f_n + prior)


def per100(x, n):
    return 100 * x / np.maximum(n, 1)


def batting(b):
    d = b[b.ball_faced].copy()
    d["out"] = (d.is_wicket & (d.dismissed_player_id == d.striker_id)).astype(float)
    d["e_runs"] = d["_e_runs"]
    d["e_out"] = d["_e_out"]
    rows = []
    for pid, g in d.groupby("striker_id"):
        r = {"player_id": pid, "name": g.striker_name.mode().iat[0],
             "usual_position": g.batting_order.median(), "matches": g.match_id.nunique(),
             "ipl_share": (g.fmt == "IPL").mean()}
        for tag, s in (("p", g[g.pressure]), ("np", g[~g.pressure])):
            n = len(s)
            r[f"balls_{tag}"] = n
            r[f"runs_{tag}"] = s.bat_runs.sum()
            r[f"outs_{tag}"] = s.out.sum()
            r[f"sr_{tag}"] = per100(s.bat_runs.sum(), n)
            r[f"rae_100_{tag}"] = per100(s.bat_runs.sum() - s.e_runs.sum(), n)
            r[f"oae_100_{tag}"] = per100(s.out.sum() - s.e_out.sum(), n)
            r[f"wpa_100_{tag}"] = per100(100 * (s.wpa - s._e_wpa_bat).sum(), n)   # vs situation average
        r["matches_p"] = g[g.pressure].match_id.nunique()
        r["wpa_total_p"] = 100 * g[g.pressure].wpa.sum()
        r["opp_edge_p"] = g[g.pressure].elo.mean() * 100   # batting side's Elo edge over the attack (+ = weaker attack)
        rows.append(r)
    t = pd.DataFrame(rows)
    for m in ("rae_100", "oae_100", "wpa_100", "sr"):
        t[f"drop_{m}"] = t[f"{m}_p"] - t[f"{m}_np"]
    return t


def bowling(b):
    d = b.copy()
    d["wkt"] = d.bowler_wkt.astype(float)
    d["is_dot_b"] = (d.legal & (d.total_runs == 0)).astype(float)
    rows = []
    for pid, g in d.groupby("bowler_id"):
        r = {"player_id": pid, "name": g.bowler_name.mode().iat[0],
             "pace_or_spin": g.pace_or_spin.mode().iat[0] if g.pace_or_spin.notna().any() else None,
             "matches": g.match_id.nunique(), "ipl_share": (g.fmt == "IPL").mean()}
        for tag, s in (("p", g[g.pressure]), ("np", g[~g.pressure])):
            n, legal = len(s), s.legal.sum()
            r[f"balls_{tag}"] = n
            r[f"legal_{tag}"] = legal
            r[f"wkts_{tag}"] = s.wkt.sum()
            r[f"econ_{tag}"] = 6 * s.bowler_runs.sum() / max(legal, 1)
            r[f"rsave_100_{tag}"] = per100(s._e_bruns.sum() - s.bowler_runs.sum(), n)
            r[f"wae_100_{tag}"] = per100(s.wkt.sum() - s._e_wkt.sum(), n)
            r[f"dotae_pct_{tag}"] = per100(s.is_dot_b.sum() - s._e_dot.sum(), n)
            r[f"wpa_100_{tag}"] = per100(-100 * (s.wpa - s._e_wpa).sum(), n)     # vs situation average
        r["matches_p"] = g[g.pressure].match_id.nunique()
        r["wpa_total_p"] = -100 * g[g.pressure].wpa.sum()
        r["opp_edge_p"] = -g[g.pressure].elo.mean() * 100  # bowling side's Elo edge over the batters (+ = weaker batting)
        rows.append(r)
    t = pd.DataFrame(rows)
    for m in ("rsave_100", "wae_100", "dotae_pct", "wpa_100", "econ"):
        t[f"drop_{m}"] = t[f"{m}_p"] - t[f"{m}_np"]
    return t


def main():
    b = pd.read_csv(OUT / "balls_wp.csv.gz", usecols=COLS, low_memory=False)
    b = b[~b.exclude & b.leverage.notna()].reset_index(drop=True)
    b["pressure"] = b.leverage >= ARGS.threshold

    faced = b.ball_faced.to_numpy()
    out = (b.is_wicket & (b.dismissed_player_id == b.striker_id)).astype(float)
    b["_e_runs"] = expected(b, b.bat_runs, faced)
    b["_e_out"] = expected(b, out, faced)
    b["_e_bruns"] = expected(b, b.bowler_runs, np.ones(len(b), bool))
    b["_e_wkt"] = expected(b, b.bowler_wkt.astype(float), np.ones(len(b), bool))
    b["_e_wpa_bat"] = expected(b, b.wpa, faced)
    b["_e_wpa"] = expected(b, b.wpa, np.ones(len(b), bool))
    b["_e_dot"] = expected(b, (b.legal & (b.total_runs == 0)).astype(float), np.ones(len(b), bool))

    # baseline check: expected should equal actual over the whole dataset (by construction, ~0 residual)
    chk = {"runs/100 faced": per100(b.bat_runs[faced].sum() - b._e_runs[faced].sum(), faced.sum()),
           "outs/100 faced": per100(out[faced].sum() - b._e_out[faced].sum(), faced.sum()),
           "bowler runs/100": per100(b.bowler_runs.sum() - b._e_bruns.sum(), len(b))}

    bat, bowl = batting(b), bowling(b)
    bat.to_csv(OUT / "batting_metrics.csv", index=False)
    bowl.to_csv(OUT / "bowling_metrics.csv", index=False)

    n = ARGS.min_balls
    L = [f"Pressure ball: leverage >= {ARGS.threshold}.  Eligibility for the leaderboard: >= {n} pressure balls.",
         "Baseline residuals over all balls (should be ~0): " + ", ".join(f"{k} {v:+.3f}" for k, v in chk.items()),
         f"Eligible batters: {(bat.balls_p >= n).sum()}   eligible bowlers: {(bowl.balls_p >= n).sum()}"]

    pd.set_option("display.width", 250)
    eb = bat[bat.balls_p >= n].sort_values("wpa_100_p", ascending=False)
    L += ["", "=== PROVISIONAL batting (raw, unshrunk) - sorted by WPA per 100 pressure balls ===",
          eb[["name", "usual_position", "ipl_share", "balls_p", "sr_p", "rae_100_p", "oae_100_p", "wpa_100_p",
              "drop_rae_100", "drop_oae_100", "matches_p", "opp_edge_p"]].head(20).round(2).to_string(index=False)]
    ew = bowl[bowl.balls_p >= n].sort_values("wpa_100_p", ascending=False)
    L += ["", "=== PROVISIONAL bowling (raw, unshrunk) - sorted by WPA per 100 pressure balls ===",
          ew[["name", "pace_or_spin", "ipl_share", "balls_p", "econ_p", "rsave_100_p", "wae_100_p",
              "dotae_pct_p", "wpa_100_p", "drop_rsave_100", "matches_p", "opp_edge_p"]].head(20).round(2).to_string(index=False)]

    # how much do the metrics agree with each other? (for choosing weights in Step 4)
    L += ["", "Spearman correlation between pressure metrics (eligible players):",
          "batting:\n" + eb[["rae_100_p", "oae_100_p", "wpa_100_p", "drop_rae_100"]].corr("spearman").round(2).to_string(),
          "bowling:\n" + ew[["rsave_100_p", "wae_100_p", "dotae_pct_p", "wpa_100_p", "drop_rsave_100"]]
          .corr("spearman").round(2).to_string()]
    # does pressure performance relate to ordinary performance? (are we just finding good players?)
    L += ["", f"Corr(pressure WPA/100, ordinary WPA/100): batters "
          f"{eb.wpa_100_p.corr(eb.wpa_100_np, method='spearman'):.2f}, bowlers "
          f"{ew.wpa_100_p.corr(ew.wpa_100_np, method='spearman'):.2f}"]
    L += [f"Corr(pressure WPA/100, opposition edge): batters {eb.wpa_100_p.corr(eb.opp_edge_p, method='spearman'):.2f}, "
          f"bowlers {ew.wpa_100_p.corr(ew.opp_edge_p, method='spearman'):.2f}  (positive = weak opposition inflates the metric)"]
    (OUT / "metrics_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--min-balls", type=int, default=120)
    ARGS = ap.parse_args()
    main()
