"""
SupeRR Selector S2 - Step 2b: show that the pressure definition separates situations.

Input : outputs/balls_wp.csv.gz (from winprob.py)
Output: outputs/fig_pressure_definition.png   leverage map of a chase + leverage by over
        outputs/fig_pressure_validation.png   behaviour by leverage band, per format
        outputs/fig_example_match.png         win-probability worm + leverage of one close match
        outputs/pressure_tables.txt           numbers behind the charts, tag overlap, eligibility counts
Usage : python src/pressure_check.py [--threshold 1.5]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# validated default palette (dataviz skill, light mode)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BANDS = [0, 0.5, 1.0, 1.5, 2.5, 99]
BAND_LABELS = ["<0.5", "0.5–1", "1–1.5", "1.5–2.5", "2.5+"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False})


def summarise(df):
    legal = df.legal.sum()
    return pd.Series({
        "balls": len(df),
        "run_rate": 6 * df.total_runs.sum() / max(legal, 1),
        "dismissal_pct": 100 * df.is_wicket.mean(),
        "dot_pct": 100 * df.is_dot.sum() / max(legal, 1),
        "boundary_pct": 100 * df.is_boundary.sum() / max(legal, 1),
        "mean_leverage": df.leverage.mean()})


def fig_definition(b):
    ch = b[(b.innings == 2) & (b.runs_needed > 0) & (b.balls_left > 0)].copy()
    ch["over_left"] = np.ceil(ch.balls_left / 6).astype(int)
    ch["need_band"] = pd.cut(ch.runs_needed, np.arange(0, 121, 10))
    grid = ch.groupby(["need_band", "over_left"], observed=True).leverage.agg(["mean", "size"])
    grid = grid[grid["size"] >= 40]["mean"].unstack("over_left").reindex(columns=range(1, 21))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    cmap = matplotlib.colors.ListedColormap(SEQ)
    norm = matplotlib.colors.BoundaryNorm([0, 0.5, 1, 1.5, 2, 3, 5, 99], cmap.N)
    im = a1.pcolormesh(np.arange(0.5, 21), np.arange(0, 121, 10), grid.to_numpy(), cmap=cmap, norm=norm,
                       edgecolors=SURF, linewidth=1)
    a1.set_xlim(20.5, 0.5)
    a1.set_xticks([20, 15, 10, 5, 1])
    a1.set_xlabel("Overs left in the chase")
    a1.set_ylabel("Runs still needed")
    a1.grid(False)
    a1.set_title("Pressure in a chase = how much one ball can swing\nthe result (mean leverage, all wickets)")
    cb = fig.colorbar(im, ax=a1, ticks=[0.25, 0.75, 1.25, 1.75, 2.5, 4, 6], fraction=0.04, pad=0.02)
    cb.ax.set_yticklabels(["<0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3–5", "5+"])
    cb.set_label("Leverage (1 = average ball)", color=INK2)
    cb.outline.set_visible(False)

    by_over = b.groupby(["innings", "over_no"]).leverage.mean().unstack(0)
    for inn, col, lab in [(1, BLUE, "1st innings"), (2, ORANGE, "2nd innings (chase)")]:
        a2.plot(by_over.index, by_over[inn], color=col, marker="o", markersize=4, label=lab)
        a2.annotate(lab, (20, by_over[inn].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, va="center", fontsize=9)
    a2.axhline(1, color=AXIS, lw=1, ls="--")
    a2.set_xlim(0.5, 25)
    a2.set_xticks([1, 6, 10, 15, 20])
    a2.set_xlabel("Over")
    a2.set_ylabel("Mean leverage")
    a2.set_title("Pressure builds through the innings,\nmost of all at the end of a chase")
    a2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pressure_definition.png", dpi=180)
    plt.close(fig)


def fig_validation(b):
    b = b.assign(band=pd.cut(b.leverage, BANDS, labels=BAND_LABELS, right=False))
    t = b.groupby(["fmt", "band"], observed=True).apply(summarise, include_groups=False)
    metrics = [("dismissal_pct", "Dismissals per 100 balls", "Dismissal rate"),
               ("dot_pct", "Dot balls (%)", "Dot-ball rate"),
               ("run_rate", "Runs per over", "Scoring rate"),
               ("boundary_pct", "Boundaries (% of legal balls)", "Boundary rate")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    x = np.arange(len(BAND_LABELS))
    for ax, (col, ylab, title) in zip(axes, metrics):
        for fmt, colr in [("IPL", BLUE), ("T20I", ORANGE)]:
            ax.plot(x, t.loc[fmt, col].reindex(BAND_LABELS), color=colr, marker="o", markersize=6, label=fmt)
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel("Leverage band")
        ax.set_title(title)
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper left")
    fig.suptitle("Behaviour changes in the top leverage bands (1.5+): more wickets, fewer dots, more boundaries",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pressure_validation.png", dpi=180)
    plt.close(fig)
    return t


def fig_example(b):
    ko = b[(b.knock_out == 1) & b.leverage.notna()]
    pick = ko.groupby("match_id").leverage.mean().idxmax()
    m = b[b.match_id == pick].copy()
    first_team = m[m.innings == 1].batting_team_id.iloc[0]
    m["wp_first"] = np.where(m.batting_team_id == first_team, m.wp_after, 1 - m.wp_after)
    m["x"] = np.arange(len(m))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(m.x, m.wp_first, color=BLUE)
    a1.axhline(0.5, color=AXIS, lw=1, ls="--")
    a1.set_ylim(0, 1)
    a1.set_ylabel("Win prob. of team\nbatting first")
    split = m.loc[m.innings == 2, "x"].min()
    for ax in (a1, a2):
        ax.axvline(split - 0.5, color=MUTED, lw=1)
    a1.text(split + 2, 0.93, "Chase begins", color=INK2, fontsize=9)
    w = m[m.is_wicket]
    a1.scatter(w.x, w.wp_first, s=40, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=3, label="Wicket")
    a1.legend(loc="lower left")
    title = m.comp_name.iloc[0] + " – " + str(m.stage.iloc[0]) + f" (match {pick})"
    a1.set_title(f"Example: {title}")
    a2.bar(m.x, m.leverage, width=0.8, color=np.where(m.leverage >= ARGS.threshold, ORANGE, BLUE))
    a2.axhline(ARGS.threshold, color=INK2, lw=1, ls="--")
    a2.text(1, ARGS.threshold, f"pressure threshold ({ARGS.threshold})", color=INK2, fontsize=8, va="bottom")
    a2.set_ylabel("Leverage")
    a2.set_xlabel("Ball of the match")
    fig.tight_layout()
    fig.savefig(OUT / "fig_example_match.png", dpi=180)
    plt.close(fig)
    return pick


def main():
    cols = ["match_id", "fmt", "comp_name", "stage", "knock_out", "innings", "over_no", "batting_team_id",
            "striker_id", "bowler_id", "runs_needed", "balls_left", "total_runs", "legal", "ball_faced",
            "is_dot", "is_boundary", "is_wicket", "exclude", "wp_before", "wp_after", "leverage",
            "tag_chase_rrr10", "tag_collapse", "tag_death_close", "tag_knockout"]
    b = pd.read_csv(OUT / "balls_wp.csv.gz", usecols=cols, low_memory=False)
    b = b[~b.exclude & b.leverage.notna()]
    thr = ARGS.threshold
    b["pressure"] = b.leverage >= thr
    L = []

    fig_definition(b)
    t = fig_validation(b)
    pick = fig_example(b)

    L.append(f"Pressure ball = leverage >= {thr}:  {b.pressure.mean():.1%} of all balls "
             f"(IPL {b[b.fmt == 'IPL'].pressure.mean():.1%}, T20I {b[b.fmt == 'T20I'].pressure.mean():.1%})")
    comp = b.groupby(["fmt", "pressure"]).apply(summarise, include_groups=False)
    L += ["", "=== Pressure vs ordinary balls ===", comp.round(2).to_string()]
    # same comparison inside each phase and innings, so the gap is not just 'death overs vs the rest'
    b["phase"] = pd.cut(b.over_no, [0, 6, 15, 99], labels=["powerplay", "middle", "death"])
    within = b.groupby(["fmt", "innings", "phase", "pressure"], observed=True).apply(summarise, include_groups=False)
    L += ["", "=== Pressure vs ordinary, within the same format / innings / phase ===", within.round(2).to_string()]
    d = within[["dismissal_pct", "dot_pct", "run_rate", "boundary_pct"]].unstack("pressure")
    w = within["balls"].unstack("pressure")[True]
    gap = pd.DataFrame({c: d[c][True] - d[c][False] for c in ["dismissal_pct", "dot_pct", "run_rate", "boundary_pct"]})
    L += ["", "Pressure minus ordinary, averaged over cells weighted by pressure balls:",
          (gap.mul(w, axis=0).sum() / w.sum()).round(2).to_string()]
    L += ["", "=== By leverage band ===", t.round(2).to_string()]

    tags = {"Chase: RRR > 10 in overs 16-20": "tag_chase_rrr10", "Collapse (3+ wkts for <= 40)": "tag_collapse",
            "Death overs, WP 25-75%": "tag_death_close", "Knockout match": "tag_knockout"}
    rows = []
    for name, col in tags.items():
        s = b[b[col]]
        rows.append({"tag": name, "balls": len(s), "share_of_all": len(s) / len(b),
                     "mean_leverage": s.leverage.mean(), "share_pressure": s.pressure.mean()})
    rows.append({"tag": "Any tag", "balls": int(b[list(tags.values())].any(axis=1).sum()),
                 "share_of_all": b[list(tags.values())].any(axis=1).mean(),
                 "mean_leverage": b[b[list(tags.values())].any(axis=1)].leverage.mean(),
                 "share_pressure": b[b[list(tags.values())].any(axis=1)].pressure.mean()})
    L += ["", "=== Secondary tags vs leverage ===", pd.DataFrame(rows).round(3).to_string(index=False)]
    untagged = b[b.pressure & ~b[list(tags.values())].any(axis=1)]
    L.append(f"Pressure balls carrying no tag: {len(untagged) / b.pressure.sum():.1%} "
             "(the leverage measure finds pressure the simple rules miss)")

    # eligibility: how many players reach N pressure balls
    bat = b[b.pressure & b.ball_faced].groupby("striker_id").size()
    bowl = b[b.pressure & b.legal].groupby("bowler_id").size()
    L += ["", "=== Players with at least N pressure balls (for the eligibility rule) ==="]
    for n in (60, 90, 120, 150, 200):
        L.append(f"N={n:>3}:  batters {int((bat >= n).sum()):>3}   bowlers {int((bowl >= n).sum()):>3}")
    L.append(f"\nExample match chart: match_id {pick}")
    (OUT / "pressure_tables.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=1.5, help="leverage at or above which a ball counts as pressure")
    ARGS = ap.parse_args()
    main()
