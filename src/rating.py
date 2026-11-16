"""
SupeRR Selector S2 - Step 4: fair comparison -> pressure rating -> top 10s.

Input : outputs/balls_wp.csv.gz
Output: outputs/rating_batters.csv, outputs/rating_bowlers.csv (all eligible players, ranked)
        outputs/rating_summary.txt
Usage : python src/rating.py [--threshold 1.5] [--min-balls 120]

1. Per-ball contributions vs the situation average (same baselines as metrics.py).
2. Roles. Batters: top order (1-3), middle (4-5), finisher (6-7), lower (8-11) by usual position.
   Bowlers: pace/spin x the phase where most of their pressure balls come (powerplay / middle / death).
3. Empirical-Bayes shrinkage. For each metric, a player's rate is pulled toward their role average by
   k / (n + k), where n = their balls and k = (ball-to-ball noise) / (true spread between players),
   both estimated from the data (method of moments). Small samples move a lot; big samples barely move.
4. Rating = weighted sum of z-scores of the shrunk metrics among eligible players (weights below).
   "Delivers when it matters" = good under pressure AND holds their own level (pressure minus ordinary).

THE WEIGHTS AND THE ELIGIBILITY RULE ARE JUDGEMENT CALLS - edit WEIGHTS / --min-balls and re-run.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import COLS, expected

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# metric -> weight. Signs: every metric is oriented so that higher = better.
# Chosen weighting "C" (Day 5b): lean on the components that repeat across halves of the data
# (runs above expected, survival, runs saved, dots); WPA keeps a share because it is what "delivers when it
# matters" means. 'Holds own level' (pressure minus ordinary) is reported per player but NOT weighted: it
# subtracts two noisy numbers and was the least repeatable term. Alternatives tested in weights_test.py.
WEIGHTS = {
    "bat": {"rae_p": 0.40, "surv_p": 0.30, "wpa_p": 0.30},
    "bowl": {"rsave_p": 0.35, "dot_p": 0.25, "wae_p": 0.15, "wpa_p": 0.25},
}
# per-ball contribution columns (higher = better) and how "holding level" is defined
CONTRIB = {
    "bat": {"wpa": "c_wpa", "rae": "c_rae", "surv": "c_surv"},
    "bowl": {"wpa": "c_wpa", "rsave": "c_rsave", "wae": "c_wae", "dot": "c_dot"},
}
def _auction_ineligible():
    """Players who cannot be bought at the IPL auction: anyone who has played for a Pakistan side (senior or A team)
    in the data pack. Pakistan players have not taken part in the IPL since 2008. Derived from playing_eleven.csv."""
    data = ROOT / "data"
    pe = pd.read_csv(data / "playing_eleven.csv", usecols=["team_id", "player_id"])
    t = pd.read_csv(data / "teams.csv").drop_duplicates("team_id")
    pk = t[t.team_name.str.contains("Pakistan")].team_id
    return set(pe[pe.team_id.isin(pk)].player_id)


INELIGIBLE = _auction_ineligible()

HOLD = {"bat": ["wpa", "rae", "surv"], "bowl": ["wpa", "rsave", "wae"]}   # averaged z of (pressure - ordinary)


def per_ball(b):
    faced = b.ball_faced.to_numpy()
    out = (b.is_wicket & (b.dismissed_player_id == b.striker_id)).astype(float)
    dot = (b.legal & (b.total_runs == 0)).astype(float)
    allb = np.ones(len(b), bool)
    bat = b[faced].copy()
    bat["c_rae"] = bat.bat_runs - expected(b, b.bat_runs, faced)[faced]
    bat["c_surv"] = expected(b, out, faced)[faced] - out[faced]          # + = survives better than average
    # WPA is centred on the situation average too: in some situations (e.g. middle-overs pressure) the batting
    # side loses WP on average, so raw WPA would reward a ROLE rather than a player.
    e_wpa_bat = expected(b, b.wpa, faced)
    e_wpa_all = expected(b, b.wpa, allb)
    bat["c_wpa"] = 100 * (bat.wpa - e_wpa_bat[faced])                     # percentage points
    bowl = b.copy()
    bowl["c_rsave"] = expected(b, b.bowler_runs, allb) - b.bowler_runs
    bowl["c_wae"] = b.bowler_wkt.astype(float) - expected(b, b.bowler_wkt.astype(float), allb)
    bowl["c_dot"] = dot - expected(b, dot, allb)
    bowl["c_wpa"] = -100 * (b.wpa - e_wpa_all)
    return bat, bowl


def roles(bat, bowl):
    pos = bat.groupby("striker_id").batting_order.median()
    bat_role = pd.cut(pos, [0, 3, 5, 7, 11], labels=["top (1-3)", "middle (4-5)", "finisher (6-7)", "lower (8-11)"])
    p = bowl[bowl.pressure]
    ph = p.groupby("bowler_id").phase.agg(lambda s: s.mode().iat[0])
    ps = bowl.groupby("bowler_id").pace_or_spin.agg(lambda s: s.mode().iat[0] if s.notna().any() else "Unknown")
    bowl_role = (ps.reindex(ph.index).fillna("Unknown") + " / " + ph).reindex(ps.index)
    bowl_role = bowl_role.fillna(ps + " / none")
    return bat_role.astype(str), bowl_role


def aggregate(d, pid, cols):
    """Per player x (pressure / ordinary): n, mean and within-player variance of each contribution."""
    g = d.groupby([pid, "pressure"])
    agg = {"n": g.size()}
    for c in cols:
        agg[f"{c}_mean"] = g[c].mean()
        agg[f"{c}_var"] = g[c].var(ddof=1)
    a = pd.DataFrame(agg).unstack("pressure")
    a.columns = [f"{m}_{'p' if p else 'np'}" for m, p in a.columns]
    return a


def shrink(mean, n, var, role, min_n=30):
    """Empirical-Bayes: pull each player's mean toward the prior by k/(n+k).
    Prior = overall average (default) or the player's role average (--prior role)."""
    ok = n >= min_n
    w = n[ok]
    overall = (mean[ok] * w).sum() / w.sum()
    if ARGS.prior == "role":
        role_mean = (mean[ok] * w).groupby(role[ok]).sum() / w.groupby(role[ok]).sum()
        prior = role.map(role_mean).fillna(overall)
    else:
        prior = pd.Series(overall, index=mean.index)
    noise = (var[ok] * (w - 1)).sum() / (w - 1).sum()                  # pooled ball-to-ball variance
    resid = mean[ok] - prior[ok]
    tau2 = max(np.average(resid ** 2, weights=w) - noise * np.average(1 / w, weights=w), noise / 2000)
    k = noise / tau2
    est = (n.fillna(0) * mean.fillna(0) + k * prior) / (n.fillna(0) + k)
    return est, k


def build(d, kind, pid, name_col, role, min_balls):
    cols = list(CONTRIB[kind].values())
    a = aggregate(d, pid, cols)
    a["role"] = role.reindex(a.index)
    a["name"] = d.groupby(pid)[name_col].agg(lambda s: s.mode().iat[0]).reindex(a.index)
    a["matches_p"] = d[d.pressure].groupby(pid).match_id.nunique().reindex(a.index).fillna(0).astype(int)
    a["ipl_share_p"] = d[d.pressure].assign(i=lambda x: x.fmt == "IPL").groupby(pid).i.mean().reindex(a.index)
    a["opp_edge_p"] = (d[d.pressure].groupby(pid).elo.mean() * (100 if kind == "bat" else -100)).reindex(a.index)
    ks = {}
    for m, c in CONTRIB[kind].items():
        for t in ("p", "np"):
            a[f"{m}_{t}_raw"] = a[f"{c}_mean_{t}"] * 100           # per 100 balls
            est, k = shrink(a[f"{c}_mean_{t}"], a[f"n_{t}"], a[f"{c}_var_{t}"], a.role)
            a[f"{m}_{t}"] = est * 100
            ks[f"{m}_{t}"] = k
    rm = a[a.n_p >= 30]
    role_avg = {m: (rm[f"{c}_mean_p"] * rm.n_p).groupby(rm.role).sum() / rm.n_p.groupby(rm.role).sum() * 100
                for m, c in CONTRIB[kind].items()}
    ks["_role_avg"] = pd.DataFrame(role_avg).round(2)
    elig = a[(a.n_p >= min_balls) & ~a.index.isin(INELIGIBLE)].copy()   # auction pool only
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    for m in CONTRIB[kind]:
        elig[f"z_{m}_p"] = z(elig[f"{m}_p"])
    elig["hold"] = sum(z(elig[f"{m}_p"] - elig[f"{m}_np"]) for m in HOLD[kind]) / len(HOLD[kind])
    elig["z_hold"] = z(elig.hold)
    W = WEIGHTS[kind]
    elig["rating"] = sum(w * elig["z_" + m] for m, w in W.items())
    elig = elig.sort_values("rating", ascending=False)
    elig.insert(0, "rank", np.arange(1, len(elig) + 1))
    return elig, ks, a


def main():
    b = pd.read_csv(OUT / "balls_wp.csv.gz", usecols=COLS + ["elo"], low_memory=False)
    b = b[~b.exclude & b.leverage.notna()].reset_index(drop=True)
    b["pressure"] = b.leverage >= ARGS.threshold
    bat, bowl = per_ball(b)
    bat_role, bowl_role = roles(bat, bowl)

    rb, kb, _ = build(bat, "bat", "striker_id", "striker_name", bat_role, ARGS.min_balls)
    rw, kw, _ = build(bowl, "bowl", "bowler_id", "bowler_name", bowl_role, ARGS.min_balls)
    rb.reset_index(names="player_id").to_csv(OUT / "rating_batters.csv", index=False)
    rw.reset_index(names="player_id").to_csv(OUT / "rating_bowlers.csv", index=False)

    pd.set_option("display.width", 250)
    L = [f"Pressure: leverage >= {ARGS.threshold}.  Eligible: >= {ARGS.min_balls} pressure balls "
         f"-> {len(rb)} batters, {len(rw)} bowlers.",
         "Weights: batting " + str(WEIGHTS["bat"]) + "   bowling " + str(WEIGHTS["bowl"]),
         "Shrinkage strength k (balls of 'average role player' added to every sample):",
         "  batting " + ", ".join(f"{m} {v:.0f}" for m, v in kb.items() if m[0] != "_"),
         "  bowling " + ", ".join(f"{m} {v:.0f}" for m, v in kw.items() if m[0] != "_"),
         f"Shrinkage prior: {ARGS.prior} average", "",
         "Role averages on pressure balls (raw, per 100 balls, players with 30+ pressure balls):",
         kb["_role_avg"].to_string(), kw["_role_avg"].to_string(), ""]
    show_b = ["rank", "name", "role", "n_p", "matches_p", "ipl_share_p", "rating", "wpa_p", "wpa_p_raw",
              "rae_p", "surv_p", "hold", "opp_edge_p"]
    show_w = ["rank", "name", "role", "n_p", "matches_p", "ipl_share_p", "rating", "wpa_p", "wpa_p_raw",
              "rsave_p", "wae_p", "dot_p", "hold", "opp_edge_p"]
    L += ["=== BATTERS (shrunk; wpa/rae/surv per 100 pressure balls; hold = z of pressure minus ordinary) ===",
          rb[show_b].head(25).round(2).to_string(index=False), "",
          "=== BOWLERS (shrunk; wpa/rsave/wae/dot per 100 pressure balls) ===",
          rw[show_w].head(25).round(2).to_string(index=False), "",
          "Top 10 by role:", "  batters: " + rb.head(10).role.value_counts().to_dict().__repr__(),
          "  bowlers: " + rw.head(10).role.value_counts().to_dict().__repr__()]
    (OUT / "rating_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=1.5)
    ap.add_argument("--min-balls", type=int, default=120)
    ap.add_argument("--prior", choices=["global", "role"], default="global",
                    help="shrink toward the overall average (default) or the player's role average")
    ARGS = ap.parse_args()
    main()
