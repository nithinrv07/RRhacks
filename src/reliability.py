"""
SupeRR Selector S2 - Step 5: how much can the top 10s be trusted?

Input : outputs/balls_wp.csv.gz (+ balls_wp_noelo.csv.gz for one sensitivity scenario)
Output: outputs/reliability_batters.csv, outputs/reliability_bowlers.csv
        outputs/reliability_summary.txt, outputs/fig_rank_intervals.png
Usage : python src/reliability.py [--boot 300]

1. Bootstrap. Resample MATCHES with replacement (balls in a match are not independent), recompute the whole
   rating (shrinkage strength included) each time -> 90% rank interval and P(top 10) for every player.
2. Split-half. Rate players separately on 2022-24 and 2025-26 (and on two random halves of matches):
   does pressure performance persist? Compared with the same check on ordinary balls.
3. Sensitivity. Re-rate under other weights, pressure thresholds, eligibility rules, shrinkage prior and
   the no-Elo pressure model; count how often each player stays in the top 10.
4. Consistency. Share of a player's pressure matches with a positive WPA contribution, and its spread.
"""
import argparse
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rating as R
from metrics import COLS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
KINDS = {"bat": ("striker_id", "striker_name"), "bowl": ("bowler_id", "bowler_name")}


# ------------------------------------------------------------------ fast engine on match-level sums
def load(path="balls_wp.csv.gz"):
    parts = []   # read in chunks to keep peak memory low
    for ch in pd.read_csv(OUT / path, usecols=COLS + ["elo", "year"], chunksize=40000):
        parts.append(ch[~ch.exclude.astype(bool) & ch.leverage.notna()])
    b = pd.concat(parts, ignore_index=True)
    for c in ("striker_name", "bowler_name", "bowling_type"):
        b[c] = b[c].astype("category")
    return b


def prepare(b, threshold):
    b = b.assign(pressure=b.leverage >= threshold)
    bat, bowl = R.per_ball(b)
    bat_role, bowl_role = R.roles(bat, bowl)
    out = {}
    for kind, d, role in (("bat", bat, bat_role), ("bowl", bowl, bowl_role)):
        pid, name = KINDS[kind]
        cols = list(R.CONTRIB[kind].values())
        d = d.assign(**{f"{c}_sq": d[c] ** 2 for c in cols}, one=1.0)
        st = d.groupby([pid, "pressure", "match_id"])[["one"] + cols + [f"{c}_sq" for c in cols]].sum()
        st = st.rename(columns={"one": "n"}).reset_index().rename(columns={pid: "pid"})
        names = d.groupby(pid)[name].agg(lambda s: s.mode().iat[0])
        out[kind] = {"stats": st, "role": role, "names": names, "cols": cols}
    years = b.groupby("match_id").year.first()
    return out, years


def rate(P, kind, weights=None, min_balls=120, prior="global", match_w=None, match_mask=None, eligible=None):
    """Rating from match-level sums. match_w: bootstrap weight per match; match_mask: subset of matches."""
    R.ARGS = Namespace(prior=prior)
    st = P["stats"]
    if match_mask is not None:
        st = st[st.match_id.isin(match_mask)]
    vals = st.drop(columns=["pid", "pressure", "match_id"])
    if match_w is not None:
        vals = vals.mul(st.match_id.map(match_w).fillna(0).to_numpy(), axis=0)
    g = vals.groupby([st.pid.to_numpy(), st.pressure.to_numpy()]).sum()
    g.index.names = ["pid", "pressure"]
    a = pd.DataFrame(index=g.index.get_level_values(0).unique())
    for t, flag in (("p", True), ("np", False)):
        s = g.xs(flag, level="pressure").reindex(a.index)
        n = s.n
        a[f"n_{t}"] = n
        for m, c in R.CONTRIB[kind].items():
            mean = s[c] / n
            var = (s[f"{c}_sq"] - n * mean ** 2) / (n - 1)
            est, _ = R.shrink(mean, n, var, P["role"].reindex(a.index))
            a[f"{m}_{t}"] = est * 100
    elig = (a[(a.n_p.fillna(0) >= min_balls) & ~a.index.isin(R.INELIGIBLE)] if eligible is None
            else a.reindex(eligible))
    elig = elig.copy()
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    for m in R.CONTRIB[kind]:
        elig[f"z_{m}_p"] = z(elig[f"{m}_p"])
    elig["z_hold"] = z(sum(z(elig[f"{m}_p"] - elig[f"{m}_np"]) for m in R.HOLD[kind]) / len(R.HOLD[kind]))
    W = weights or R.WEIGHTS[kind]
    elig["rating"] = sum(w * elig["z_" + m] for m, w in W.items())
    elig["rank"] = elig.rating.rank(ascending=False, method="first")
    return elig.sort_values("rank")


# ------------------------------------------------------------------ checks
def bootstrap(P, kind, base, B, rng):
    matches = P["stats"].match_id.unique()
    ranks = []
    for _ in range(B):
        draw = rng.integers(0, len(matches), len(matches))
        w = pd.Series(np.bincount(draw, minlength=len(matches)), index=matches)
        r = rate(P, kind, match_w=w, eligible=base.index)
        ranks.append(r["rank"].reindex(base.index))
    M = pd.concat(ranks, axis=1)
    return pd.DataFrame({"rank_p05": M.quantile(0.05, axis=1), "rank_median": M.median(axis=1),
                         "rank_p95": M.quantile(0.95, axis=1), "p_top10": (M <= 10).mean(axis=1)})


def split_half(P, kind, years, rng, min_balls=60):
    early = years[years <= 2024].index
    late = years[years >= 2025].index
    allm = years.index.to_numpy().copy()
    rng.shuffle(allm)
    h1, h2 = allm[: len(allm) // 2], allm[len(allm) // 2:]
    rows = []
    for label, (A, Bm) in {"2022-24 vs 2025-26": (early, late), "random halves of matches": (h1, h2)}.items():
        ra = rate(P, kind, min_balls=min_balls, match_mask=A)
        rb = rate(P, kind, min_balls=min_balls, match_mask=Bm)
        both = ra.index.intersection(rb.index)
        row = {"split": label, "players": len(both),
               "rating (pressure)": ra.rating[both].corr(rb.rating[both], method="spearman")}
        for m in R.CONTRIB[kind]:
            row[f"{m} pressure"] = ra[f"{m}_p"][both].corr(rb[f"{m}_p"][both], method="spearman")
            row[f"{m} ordinary"] = ra[f"{m}_np"][both].corr(rb[f"{m}_np"][both], method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def scenarios(kind):
    W = R.WEIGHTS[kind]
    eq = {m: 1 / len(W) for m in W}
    if "hold" in W:
        hold_label, hold_w = "no 'holds level' term", {m: (0 if m == "hold" else w) for m, w in W.items()}
    else:
        hold_label, hold_w = "add 'holds level' at 20%", {**{m: 0.8 * w for m, w in W.items()}, "hold": 0.2}
    wpa_only = {"wpa_p": 1.0}
    if kind == "bat":
        box = {"rae_p": 0.5, "surv_p": 0.5}
    else:
        box = {"rsave_p": 0.4, "wae_p": 0.4, "dot_p": 0.2}
    return {"equal weights": dict(weights=eq), hold_label: dict(weights=hold_w),
            "WPA only": dict(weights=wpa_only), "runs/wickets only (no WPA)": dict(weights=box),
            "eligibility 90 balls": dict(min_balls=90), "eligibility 150 balls": dict(min_balls=150),
            "shrink to role average": dict(prior="role")}


def consistency(P, kind, ids):
    st = P["stats"]
    p = st[st.pressure & st.pid.isin(ids)]
    c = R.CONTRIB[kind]["wpa"]
    return p.groupby("pid").apply(lambda g: pd.Series({
        "pressure_matches": len(g), "share_matches_positive": (g[c] > 0).mean(),
        "wpa_per_match_sd": g[c].std(ddof=1)}), include_groups=False)


def chart(tables):
    BLUE, MUTED, INK2, GRID, AXIS, SURF = "#2a78d6", "#898781", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, (title, t) in zip(axes, tables.items()):
        t = t.head(20).iloc[::-1]
        y = np.arange(len(t))
        ax.hlines(y, t.rank_p05, t.rank_p95, color=AXIS, lw=4)
        ax.scatter(t["rank"], y, color=BLUE, s=50, zorder=3, edgecolor=SURF, linewidth=2)
        ax.axvline(10.5, color=INK2, lw=1, ls="--")
        ax.text(10.7, len(t) - 0.6, "top-10 cut", color=INK2, fontsize=8)
        ax.set_yticks(y, [f"{n}  ({p:.0%})" for n, p in zip(t.name, t.p_top10)])
        ax.set_xlabel("Rank (dot = full data; bar = 90% bootstrap interval)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_facecolor(SURF)
        ax.grid(axis="x", color=GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.suptitle("How certain are the ranks? (label: share of bootstrap runs in the top 10)",
                 x=0.01, ha="left", fontweight="bold")
    fig.patch.set_facecolor(SURF)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rank_intervals.png", dpi=180)
    plt.close(fig)


def main():
    rng = np.random.default_rng(7)
    b = load()
    P, years = prepare(b, 1.5)
    L, tables = [], {}
    pd.set_option("display.width", 250)

    sens_runs = {}
    for thr in (1.25, 2.0):
        Pt, _ = prepare(b, thr)
        sens_runs[f"pressure threshold {thr}"] = (Pt, {})
    del b
    bn = load("balls_wp_noelo.csv.gz")
    Pn, _ = prepare(bn, 1.5)
    sens_runs["pressure model without Elo"] = (Pn, {})
    del bn

    for kind, label in (("bat", "BATTERS"), ("bowl", "BOWLERS")):
        base = rate(P[kind], kind)
        names = P[kind]["names"]
        # check: the fast engine reproduces rating.py
        ref = pd.read_csv(OUT / f"rating_{'batters' if kind == 'bat' else 'bowlers'}.csv").head(10).player_id
        L.append(f"{label}: engine reproduces rating.py top 10: {list(base.head(10).index) == list(ref)}")

        bs = bootstrap(P[kind], kind, base, ARGS.boot, rng)
        top10 = base.head(10).index
        sens = {}
        for sname, kw in scenarios(kind).items():
            sens[sname] = set(rate(P[kind], kind, **kw).head(10).index)
        for sname, (Pt, _) in sens_runs.items():
            sens[sname] = set(rate(Pt[kind], kind).head(10).index)
        times = pd.Series({i: sum(i in s for s in sens.values()) for i in base.index})
        cons = consistency(P[kind], kind, base.index)

        t = base[["rank", "rating", "n_p"]].join(bs).join(cons)
        t["in_top10_of_scenarios"] = times.astype(str) + f"/{len(sens)}"
        t.insert(0, "name", names.reindex(t.index))
        t.reset_index(names="player_id").to_csv(OUT / f"reliability_{'batters' if kind == 'bat' else 'bowlers'}.csv",
                                                index=False)
        tables[label.title()] = t
        L += ["", f"=== {label}: current top 20 with 90% rank intervals ({ARGS.boot} match-bootstraps) ===",
              t.head(20)[["name", "rank", "rank_p05", "rank_median", "rank_p95", "p_top10", "pressure_matches",
                          "share_matches_positive", "in_top10_of_scenarios"]].round(2).to_string(index=False)]
        L += ["", "Sensitivity: how many of the current top 10 survive each change"]
        for sname, s in sens.items():
            L.append(f"  {sname:<32} {len(s & set(top10)):>2}/10   new: "
                     + ", ".join(names.reindex(sorted(s - set(top10))).fillna("?")))
        sh = split_half(P[kind], kind, years, rng)
        L += ["", "Split-half (Spearman correlation of shrunk metrics, players with 60+ pressure balls in both):",
              sh.round(2).to_string(index=False)]
    chart(tables)
    (OUT / "reliability_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    ARGS = ap.parse_args()
    main()
