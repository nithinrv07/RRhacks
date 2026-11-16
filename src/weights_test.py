"""
SupeRR Selector S2 - Step 5b: compare candidate weightings on reliability before choosing one.

Output: outputs/weights_test.txt
Usage : python src/weights_test.py [--splits 30] [--boot 150]

For each candidate weighting:
  * repeatability  = Spearman correlation of the rating between two halves of the matches
                     (mean over --splits random halves, and the 2022-24 vs 2025-26 split)
  * stability      = mean bootstrap P(top 10) of the players it puts in its top 10 (higher = steadier list)
  * overlap        = how many of its top 10 are in the current (Day 4) top 10
The "reliability-proportional" candidate sets each weight in proportion to that component's own split-half
correlation (negative -> 0). It is derived from the same data, so treat its repeatability as optimistic.
"""
import argparse

import numpy as np
import pandas as pd

import rating as R
from reliability import load, prepare, rate


def candidates(kind, comp_rel):
    cur = R.WEIGHTS[kind]
    if kind == "bat":
        c = {"A. Day 4 (WPA-led, with 'holds level')": {"wpa_p": 0.40, "rae_p": 0.20, "surv_p": 0.20, "hold": 0.20},
             "B. current without 'holds level'": {"wpa_p": 0.5, "rae_p": 0.25, "surv_p": 0.25},
             "C. repeatable-led (runs/survival 70%, WPA 30%)": {"rae_p": 0.40, "surv_p": 0.30, "wpa_p": 0.30},
             "D. runs + survival only": {"rae_p": 0.5, "surv_p": 0.5}}
    else:
        c = {"A. Day 4 (WPA-led, with 'holds level')": {"wpa_p": 0.40, "rsave_p": 0.15, "wae_p": 0.15, "dot_p": 0.10, "hold": 0.20},
             "B. current without 'holds level'": {"wpa_p": 0.5, "rsave_p": 0.2, "wae_p": 0.2, "dot_p": 0.1},
             "C. repeatable-led (runs/dots 60%, WPA 25%)": {"rsave_p": 0.35, "dot_p": 0.25, "wae_p": 0.15,
                                                           "wpa_p": 0.25},
             "D. runs saved + dots + wickets only": {"rsave_p": 0.45, "dot_p": 0.35, "wae_p": 0.20}}
    pos = {f"{m}_p": max(r, 0) for m, r in comp_rel.items()}
    tot = sum(pos.values())
    c["E. reliability-proportional"] = {m: round(v / tot, 2) for m, v in pos.items() if v > 0}
    return c


def halves(years, n, rng):
    ids = years.index.to_numpy()
    out = [("time", years[years <= 2024].index, years[years >= 2025].index)]
    for _ in range(n):
        p = rng.permutation(ids)
        out.append(("random", p[: len(p) // 2], p[len(p) // 2:]))
    return out


def corr_halves(P, kind, W, splits):
    res = {"time": [], "random": []}
    for lab, A, B in splits:
        ra = rate(P, kind, weights=W, min_balls=60, match_mask=A)
        rb = rate(P, kind, weights=W, min_balls=60, match_mask=B)
        both = ra.index.intersection(rb.index)
        res[lab].append(ra.rating[both].corr(rb.rating[both], method="spearman"))
    return np.mean(res["random"]), np.std(res["random"]), res["time"][0]


def component_reliability(P, kind, splits):
    out = {m: [] for m in R.CONTRIB[kind]}
    for lab, A, B in splits:
        if lab != "random":
            continue
        ra = rate(P, kind, min_balls=60, match_mask=A)
        rb = rate(P, kind, min_balls=60, match_mask=B)
        both = ra.index.intersection(rb.index)
        for m in out:
            out[m].append(ra[f"{m}_p"][both].corr(rb[f"{m}_p"][both], method="spearman"))
    return {m: float(np.mean(v)) for m, v in out.items()}


def boot_top10(P, kind, W, B, rng):
    base = rate(P, kind, weights=W)
    matches = P["stats"].match_id.unique()
    hits = pd.Series(0.0, index=base.index)
    for _ in range(B):
        w = pd.Series(np.bincount(rng.integers(0, len(matches), len(matches)), minlength=len(matches)),
                      index=matches)
        r = rate(P, kind, weights=W, match_w=w, eligible=base.index)
        hits += (r["rank"].reindex(base.index) <= 10)
    p = hits / B
    top = base.head(10).index
    return base, p, float(p[top].mean())


def main():
    rng = np.random.default_rng(11)
    b = load()
    P, years = prepare(b, 1.5)
    del b
    splits = halves(years, ARGS.splits, rng)
    L = [f"Repeatability = mean Spearman of the rating between two random halves of matches ({ARGS.splits} splits, "
         "± sd), and between 2022-24 and 2025-26. Stability = mean bootstrap P(top 10) of the list's own top 10 "
         f"({ARGS.boot} resamples). Overlap = names shared with the current Day 4 top 10.", ""]
    for kind, label in (("bat", "BATTERS"), ("bowl", "BOWLERS")):
        names = P[kind]["names"]
        comp = component_reliability(P[kind], kind, splits)
        L += [f"=== {label} ===",
              "Component repeatability (random halves): " + ", ".join(f"{m} {v:+.2f}" for m, v in comp.items())]
        cur_top = rate(P[kind], kind).head(10).index
        rows, lists = [], {}
        for cname, W in candidates(kind, comp).items():
            mean_r, sd_r, time_r = corr_halves(P[kind], kind, W, splits)
            base, p, stab = boot_top10(P[kind], kind, W, ARGS.boot, rng)
            top = base.head(10).index
            rows.append({"weighting": cname, "weights": W, "repeat_random": f"{mean_r:+.2f} ± {sd_r:.2f}",
                         "repeat_time": round(time_r, 2), "stability": round(stab, 2),
                         "overlap_with_current": f"{len(set(top) & set(cur_top))}/10"})
            lists[cname[:2]] = [f"{names.get(i, i)} ({p[i]:.0%})" for i in top]
        t = pd.DataFrame(rows)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 80)
        L += [t.drop(columns="weights").to_string(index=False), "", "Weights:"]
        L += [f"  {r['weighting'][:2]} {r['weights']}" for r in rows]
        L += ["", "Top 10 under each weighting (bootstrap P(top 10) in brackets):",
              pd.DataFrame(lists, index=range(1, 11)).to_string(), ""]
    open(R.OUT / "weights_test.txt", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--boot", type=int, default=150)
    ARGS = ap.parse_args()
    main()
