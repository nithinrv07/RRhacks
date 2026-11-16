"""
SupeRR Selector S2 - Step 6: build the methodology report (PDF) from the pipeline outputs.

Output: report/SupeRR_S2_Round1_Methodology.pdf, report/submission_order.txt, report/fig_*.png
Usage : python src/report.py      (run after rating.py, reliability.py and weights_test.py)
Every number in the report is read from outputs/ or stated where it was measured.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT, REP = ROOT / "outputs", ROOT / "report"
REP.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False})


# =============================================================== data
def load():
    d = {}
    d["rb"] = pd.read_csv(OUT / "rating_batters.csv")
    d["rw"] = pd.read_csv(OUT / "rating_bowlers.csv")
    d["lb"] = pd.read_csv(OUT / "reliability_batters.csv").set_index("player_id")
    d["lw"] = pd.read_csv(OUT / "reliability_bowlers.csv").set_index("player_id")
    d["mb"] = pd.read_csv(OUT / "batting_metrics.csv").set_index("player_id")
    d["mw"] = pd.read_csv(OUT / "bowling_metrics.csv").set_index("player_id")
    d["cal"] = pd.read_csv(OUT / "wp_calibration.csv")
    rel = (OUT / "reliability_summary.txt").read_text(encoding="utf-8")
    parts = rel.split("BOWLERS: engine")
    d["sens"] = {}
    for kind, txt in zip(("bat", "bowl"), parts):
        d["sens"][kind] = re.findall(r"^  (.+?)\s{2,}(\d+)/10\s+new: (.*)$", txt, re.M)
    d["split"] = {}
    for kind, txt in zip(("bat", "bowl"), parts):
        block = txt[txt.index("Split-half"):].splitlines()
        hdr = block[1]
        rows = [l for l in block[2:4]]
        d["split"][kind] = (hdr, rows)
    wt = (OUT / "weights_test.txt").read_text(encoding="utf-8")
    wparts = wt.split("=== BOWLERS ===")
    d["wt"] = {}
    for kind, txt in zip(("bat", "bowl"), wparts):
        rows = re.findall(r"^\s*([A-E])\. .*?\s+([+-]\d\.\d\d) ± (\d\.\d\d)\s+(-?\d\.\d\d)\s+(\d\.\d\d)\s+(\d+)/10",
                          txt, re.M)
        d["wt"][kind] = rows
        d.setdefault("wt_comp", {})[kind] = re.search(r"Component repeatability \(random halves\): (.*)", txt).group(1)
    return d


# =============================================================== figures
def fig_method():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    boxes = [
        (0.1, 2.0, "1  Ball state", "score, wickets, balls left,\ntarget, format, team strength"),
        (2.6, 2.0, "2  Win probability", "chase model + projected\nfirst-innings total"),
        (5.1, 2.0, "3  Leverage", "expected WP swing of\nthis ball vs average ball"),
        (7.6, 2.0, "4  Pressure ball", "leverage ≥ 1.5\n(16.5% of balls)"),
        (0.1, 0.2, "5  Contribution", "runs, survival, wickets, dots,\nWPA vs same-situation average"),
        (2.6, 0.2, "6  Fair comparison", "≥120 pressure balls;\nshrink small samples"),
        (5.1, 0.2, "7  Rating", "weighted z-scores\n(weighting C)"),
        (7.6, 0.2, "8  Reliability", "bootstrap, split-half,\nsensitivity"),
    ]
    for x, y, t, s in boxes:
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), 2.25, 1.35, boxstyle="round,pad=0.02,rounding_size=0.08",
                                                       fc="#eef4fc", ec=BLUE, lw=1.2))
        ax.text(x + 0.12, y + 1.05, t, fontsize=10.5, fontweight="bold", color=INK, va="center")
        ax.text(x + 0.12, y + 0.48, s, fontsize=8.6, color=INK2, va="center", linespacing=1.35)
    for x in (2.35, 4.85, 7.35):
        for y in (2.68, 0.88):
            ax.annotate("", (x + 0.25, y), (x, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    # elbow connector: under box 4, back along the gap between rows, down into box 5
    ax.plot([8.72, 8.72, 1.22, 1.22], [2.0, 1.78, 1.78, 1.62], color=MUTED, lw=1.3)
    ax.annotate("", (1.22, 1.56), (1.22, 1.64), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    fig.tight_layout()
    fig.savefig(REP / "fig_method.png", dpi=200)
    plt.close(fig)


def fig_calibration(cal):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1)
    for inn, col, lab in ((1, BLUE, "1st innings"), (2, ORANGE, "2nd innings (chase)")):
        c = cal[cal.innings == inn]
        ax.plot(c.pred, c.actual, color=col, marker="o", ms=6, lw=2, label=lab)
    ax.set_xlabel("Predicted win probability (out-of-fold)")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Win-probability model is well calibrated")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(REP / "fig_calibration.png", dpi=200)
    plt.close(fig)


def fig_weights(wt):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, kind, title in ((axes[0], "bat", "Batters"), (axes[1], "bowl", "Bowlers")):
        rows = wt[kind]
        lab = [r[0] for r in rows]
        rnd = [float(r[1]) for r in rows]
        tim = [float(r[3]) for r in rows]
        x = np.arange(len(lab))
        ax.bar(x - 0.2, rnd, 0.38, color=BLUE, label="random halves (mean of 20)")
        ax.bar(x + 0.2, tim, 0.38, color=ORANGE, label="2022-24 vs 2025-26")
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(x, lab)
        ax.set_title(title)
        if kind == "bat":
            ax.set_ylabel("Spearman correlation of rating\nbetween halves")
        for i, l in enumerate(lab):
            if l == "C":
                ax.annotate("chosen", (i, max(rnd[i], tim[i], 0) + 0.03), ha="center", color=INK2, fontsize=8)
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.05, -0.1), ncol=2, fontsize=8)
    fig.suptitle("How much each weighting repeats across halves of the data (higher = more repeatable)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(REP / "fig_weights.png", dpi=200)
    plt.close(fig)


def fig_components(rb, rw):
    from rating import WEIGHTS
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9))
    spec = (("bat", rb, {"rae_p": ("Runs above expected", BLUE), "surv_p": ("Survival", AQUA),
                         "wpa_p": ("Win prob. added", ORANGE)}),
            ("bowl", rw, {"rsave_p": ("Runs saved", BLUE), "dot_p": ("Dots", AQUA),
                          "wae_p": ("Wickets", YELLOW), "wpa_p": ("Win prob. added", ORANGE)}))
    for ax, (kind, r, comps) in zip(axes, spec):
        t = r.head(10).iloc[::-1]
        y = np.arange(len(t))
        left_pos, left_neg = np.zeros(len(t)), np.zeros(len(t))
        for m, (lab, col) in comps.items():
            v = (WEIGHTS[kind][m] * t[f"z_{m}"]).to_numpy()
            start = np.where(v >= 0, left_pos, left_neg)
            ax.barh(y, v, left=start, color=col, height=0.62, label=lab, edgecolor=SURF, linewidth=1)
            left_pos += np.where(v >= 0, v, 0)
            left_neg += np.where(v < 0, v, 0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y, [f"{int(k)}. {n}" for k, n in zip(t["rank"], t["name"])])
        ax.set_xlabel("Contribution to rating (weight × z-score)")
        ax.set_title("Batters" if kind == "bat" else "Bowlers")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=7.5)
        ax.grid(axis="y", visible=False)
    fig.suptitle("What drives each top-10 rating", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(REP / "fig_components.png", dpi=200)
    plt.close(fig)


# =============================================================== PDF helpers
pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", "C:/Windows/Fonts/ariali.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")

NAVY = colors.HexColor("#0d2a4f")
S = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=21, leading=26, textColor=NAVY, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="Arial", fontSize=11.5, leading=15, textColor=colors.HexColor(INK2)),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=14.5, leading=18, textColor=NAVY, spaceBefore=10,
                         spaceAfter=5, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11, leading=14, textColor=NAVY, spaceBefore=7,
                         spaceAfter=3, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="Arial", fontSize=9.4, leading=13, spaceAfter=5, alignment=TA_LEFT),
    "b": ParagraphStyle("b", fontName="Arial", fontSize=9.4, leading=13, leftIndent=11, bulletIndent=2,
                        spaceAfter=2.5),
    "cap": ParagraphStyle("cap", fontName="Arial-Italic", fontSize=8.2, leading=10.5,
                          textColor=colors.HexColor(INK2), spaceAfter=7),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.2, leading=10.2),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.2, leading=10.2),
    "card": ParagraphStyle("card", fontName="Arial", fontSize=7.9, leading=10.1),
    "box": ParagraphStyle("box", fontName="Arial", fontSize=9.2, leading=12.6),
}


def P(t, s="p"):
    return Paragraph(t, S[s])


def bullets(items):
    return [Paragraph(i, S["b"], bulletText="•") for i in items]


def fig(path, width_mm, caption=None):
    img = Image(str(path))
    r = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_mm * mm, width_mm * mm * r
    out = [img]
    if caption:
        out.append(P(caption, "cap"))
    return KeepTogether(out)


def table(rows, widths, header=True, zebra=True, font=8.2):
    data = [[c if isinstance(c, Paragraph) else Paragraph(str(c), S["cellb" if (header and i == 0) else "cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f6f9")))
    t.setStyle(TableStyle(st))
    return t


def callout(text):
    t = Table([[Paragraph(text, S["box"])]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef4fc")),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(BLUE)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(18 * mm, 10 * mm, "SupeRR Selector S2 · Round 1 · Methodology report")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def sgn(x, d=1):
    return f"{x:+.{d}f}".replace("-", "−")


def num(x, d=1):
    return f"{x:.{d}f}".replace("-", "−")


# =============================================================== evidence cards
def card_bat(r, m, l):
    return Paragraph(
        f"<b>{int(r['rank'])}. {r['name']}</b> · {r['role']}<br/>"
        f"{int(r.n_p)} pressure balls in {int(r.matches_p)} matches · IPL share {r.ipl_share_p:.0%}<br/>"
        f"<b>Under pressure:</b> SR {num(m.sr_p)} (ordinary {num(m.sr_np)}) · runs above expected "
        f"{sgn(r.rae_p)}/100 · survival {sgn(r.surv_p, 2)} outs/100 better than expected · WPA {sgn(r.wpa_p)} pts/100 "
        f"(raw {sgn(r.wpa_p_raw, 0)})<br/>"
        f"<b>Vs own level:</b> runs above expected {sgn(r.rae_p_raw)} under pressure vs {sgn(r.rae_np_raw)} ordinary "
        f"(raw) · holds-level score {sgn(r.hold, 2)}<br/>"
        f"<b>Consistency:</b> positive WPA in {l.share_matches_positive:.0%} of pressure matches · 90% rank interval "
        f"{l.rank_p05:.0f}–{l.rank_p95:.0f} · top 10 in {l.p_top10:.0%} of resamples, {l.in_top10_of_scenarios} "
        f"scenarios · opposition edge {sgn(r.opp_edge_p, 0)} Elo", S["card"])


def card_bowl(r, m, l):
    return Paragraph(
        f"<b>{int(r['rank'])}. {r['name']}</b> · {r['role']}<br/>"
        f"{int(r.n_p)} pressure balls in {int(r.matches_p)} matches · IPL share {r.ipl_share_p:.0%}<br/>"
        f"<b>Under pressure:</b> economy {num(m.econ_p, 2)} (ordinary {num(m.econ_np, 2)}) · {int(m.wkts_p)} wickets · "
        f"runs saved {sgn(r.rsave_p)}/100 · dots {sgn(r.dot_p)} pts · wickets {sgn(r.wae_p, 2)}/100 · WPA "
        f"{sgn(r.wpa_p)} pts/100 (raw {sgn(r.wpa_p_raw, 0)})<br/>"
        f"<b>Vs own level:</b> runs saved {sgn(r.rsave_p_raw)} under pressure vs {sgn(r.rsave_np_raw)} ordinary (raw) · "
        f"holds-level score {sgn(r.hold, 2)}<br/>"
        f"<b>Consistency:</b> positive WPA in {l.share_matches_positive:.0%} of pressure matches · 90% rank interval "
        f"{l.rank_p05:.0f}–{l.rank_p95:.0f} · top 10 in {l.p_top10:.0%} of resamples, {l.in_top10_of_scenarios} "
        f"scenarios · opposition edge {sgn(r.opp_edge_p, 0)} Elo", S["card"])


def cards(r, m, l, fn):
    cells = [fn(row, m.loc[row.player_id], l.loc[row.player_id]) for _, row in r.head(10).iterrows()]
    rows = [cells[i:i + 2] for i in range(0, 10, 2)]
    t = Table(rows, colWidths=[87 * mm, 87 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor(AXIS)),
                           ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor(AXIS)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t



# =============================================================== Rajasthan Royals fit
def rr_section(rb, rw):
    prof = pd.read_csv(OUT / "rr_pressure_profile.csv")
    fit = pd.read_csv(OUT / "rr_player_fit.csv")
    out = [P("8. How the picks could strengthen Rajasthan Royals", "h1"),
           P("The rankings above are team-neutral. This section uses the same per-ball measures to show where Rajasthan "
             "Royals (RR) gain and lose on pressure balls in the IPL (2022–26), and where each pick would fit. Each IPL "
             "side is compared with the average side in the same situation; the table ranks RR among sides with at "
             "least 300 pressure balls in that phase (1 = best).")]
    phases = ["powerplay", "middle", "death"]
    rows = [["RR on pressure balls", "Powerplay", "Middle overs", "Death overs"]]
    cell_bg = []
    for side, mets in (("batting", ["Runs above expected", "Survival", "WPA"]),
                       ("bowling", ["Runs saved", "Dots", "WPA"])):
        for m in mets:
            q = prof[(prof.side == side) & (prof.metric == m)].set_index("phase")
            r = [f"{side.title()}: {m}"]
            for j, ph in enumerate(phases):
                rk, of = (int(x) for x in q.loc[ph, "rank_of"].split(" of "))
                r.append(f"{rk} of {of}")
                col = "#dff3e9" if rk <= 3 else ("#fbe3dc" if rk > of - 3 else None)
                if col:
                    cell_bg.append((j + 1, len(rows), col))
            rows.append(r)
    t = table(rows, [62, 37, 37, 37], zebra=False)
    t.setStyle(TableStyle([("BACKGROUND", (c, r), (c, r), colors.HexColor(col)) for c, r, col in cell_bg]))
    out += [t, P("Table 3. RR's rank among IPL sides on pressure balls (green = top three, red = bottom three).", "cap")]

    out.append(P("What RR needs", "h2"))
    out += bullets([
        "<b>Closing innings and chases.</b> RR score well on pressure balls at the death (2nd in runs above expected) "
        "but lose wickets (survival 6th of 9) and lose win probability (6th of 9). The gap is not scoring: it is "
        "staying in to finish.",
        "<b>Middle-overs control.</b> RR rank last (10th of 10) for dot balls and 7th for runs saved on middle-overs "
        "pressure balls, so pressure is released just when the game is being set up.",
        "<b>Death bowling.</b> RR rank 8th of 10 for both win probability and dots on death-overs pressure balls."])

    def row(r):
        return [f"{int(r['rank'])}. {r['name']}", r.role, f"{r.best_pressure_phase} ({sgn(r.best_phase_value, 0)})",
                r.latest_ipl_team, r.availability]
    out.append(P("Where each pick fits", "h2"))
    out.append(P("Best pressure phase = the phase where the player adds most per 100 pressure balls (runs above "
                 "expected for batters, runs saved for bowlers; phases with 30+ balls). IPL side = the player's latest "
                 "IPL team in the data pack. Country = the national side the player appeared for in the data pack; "
                 "three picks have no international matches in 2022–26 (Sunil Narine, Krunal Pandya, Mohsin Khan), "
                 "and for them public information is used and marked. This is the only outside information in the "
                 "report."))
    hdr = ["Player", "Role", "Best pressure phase", "Latest IPL side (data pack)", "Availability"]
    for kind in ("Batter", "Bowler"):
        f = fit[fit.kind == kind]
        out.append(table([hdr] + [row(r) for _, r in f.iterrows()], [34, 25, 26, 40, 49]))
        out.append(Spacer(1, 4))
    out.append(P("Table 4. Squad fit for the top-10 batters and bowlers.", "cap"))

    b = rb.set_index("name")
    w = rw.set_index("name")
    out.append(P("Recommended targets for RR's gaps", "h2"))
    out += bullets([
        "<b>Death-overs finisher who survives:</b> Rinku Singh, Shreyas Iyer, Ashutosh Sharma and Tim David all do their "
        "best pressure work at the death and all survive better than expected under pressure (survival "
        f"{sgn(b.loc['Rinku Singh'].surv_p, 2)}, {sgn(b.loc['Shreyas Iyer'].surv_p, 2)}, "
        f"{sgn(b.loc['Ashutosh Sharma'].surv_p, 2)} and {sgn(b.loc['Tim David'].surv_p, 2)} outs per 100 balls), "
        "which is exactly RR's weakness. The three Indian players fill it without using an overseas place.",
        "<b>Middle-overs control:</b> Krunal Pandya and Varun Chakravarthy (both Indian), Mitchell Santner and Sunil "
        f"Narine (dots {sgn(w.loc['Krunal Pandya'].dot_p)}, {sgn(w.loc['Varun Chakravarthy'].dot_p)}, "
        f"{sgn(w.loc['Mitchell Santner'].dot_p)} and {sgn(w.loc['Sunil Narine'].dot_p)} points above expected on "
        "pressure balls), directly addressing RR's last-place dot-ball rate.",
        "<b>Death bowling:</b> Sunil Narine (+61 runs saved per 100 pressure balls at the death), Rashid Khan (+42) "
        "and Josh Hazlewood (+35) do their best pressure work at the death (Table 4). Mohsin Khan (Indian, +70) has "
        "the strongest death figure but a thin sample (136 pressure balls); Taskin Ahmed (+56) has no IPL matches.",
        "<b>Caveats.</b> Towhid Hridoy, Reeza Hendricks and Taskin Ahmed have no IPL matches in the data, so their "
        "case rests on T20I evidence alone. Every pick is with another side in the data pack; whether they reach the "
        "2027 auction depends on retentions and trades, which this analysis does not model."])
    return out

# =============================================================== build
def build():
    d = load()
    rb, rw, lb, lw = d["rb"], d["rw"], d["lb"], d["lw"]
    fig_method()
    fig_calibration(d["cal"])
    fig_weights(d["wt"])
    fig_components(rb, rw)

    # submission order file: what to pick in the Unstop dropdowns
    lines = ["Unstop dropdown order (must match the report)", "", "TOP 10 BATTERS"]
    lines += [f"{int(r['rank']):>2}. {r['name']}" for _, r in rb.head(10).iterrows()]
    lines += ["", "TOP 10 BOWLERS"] + [f"{int(r['rank']):>2}. {r['name']}" for _, r in rw.head(10).iterrows()]
    (REP / "submission_order.txt").write_text("\n".join(lines), encoding="utf-8")

    st = []
    # ---------------- page 1: title + summary
    st += [P("Who Delivers When It Matters?", "title"),
           P("The most reliable players under pressure for Rajasthan Royals to back at the IPL 2027 auction: a "
             "leverage-based analysis of IPL and T20I data, 2022–2026", "sub"),
           P("SupeRR Selector Season 2 · Round 1 methodology report · [Your name / team]", "sub"), Spacer(1, 10)]
    st.append(callout(
        "<b>In one paragraph.</b> We define pressure as <b>leverage</b>: how much a single ball can swing the "
        "match result, measured with a win-probability model trained on the supplied ball-by-ball data. About one "
        "ball in six (leverage ≥ 1.5) counts as a pressure ball. On those balls we measure what each player "
        "added compared with an average player facing the same kind of pressure ball in the same situation "
        "(format, innings, phase, wickets, required rate): runs, survival, runs saved, dots, wickets and win "
        "probability. Small samples are shrunk toward the average, and every ranking is stress-tested with match "
        "bootstraps, split-half checks and ten sensitivity scenarios."))
    st.append(Spacer(1, 8))
    head = ["#", "Batter", "Role", "Pressure balls", "In top 10*"]
    rows_b = [head] + [[int(r['rank']), r['name'], r.role, int(r.n_p), f"{lb.loc[r.player_id].p_top10:.0%}"]
                       for _, r in rb.head(10).iterrows()]
    head = ["#", "Bowler", "Role", "Pressure balls", "In top 10*"]
    rows_w = [head] + [[int(r['rank']), r['name'], r.role, int(r.n_p), f"{lw.loc[r.player_id].p_top10:.0%}"]
                       for _, r in rw.head(10).iterrows()]
    st.append(P("Final rankings", "h1"))
    tb = table(rows_b, [7, 31, 22, 15, 12])
    tw = table(rows_w, [7, 31, 22, 15, 12])
    two = Table([[tb, tw]], colWidths=[88 * mm, 88 * mm])
    two.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    st += [two, P("* Share of 200 match-level bootstrap resamples in which the player is in the top 10 "
                  "(Section 6). Above 60% is well supported; below 40% is a close call.", "cap")]
    st.append(P("Key findings", "h2"))
    st += bullets([
        "Leverage separates pressure from ordinary play: it is 3.3× the average ball in the last over of a chase, and "
        "in the highest-leverage balls dismissals rise to about 7 per 100 balls (vs about 5) and dot balls fall.",
        "Comparing like with like matters. Without a same-situation baseline, raw win-probability numbers simply "
        "reward finishers and middle-overs spinners; with it, the top 10s mix roles.",
        "Pressure performance is noisy. For bowlers the final rating repeats across halves of the data "
        "(Spearman +0.46 between 2022–24 and 2025–26); for batters it does not, so the batting list is our best "
        "evidence rather than a settled verdict. We report this openly (Section 7).",
        "For Rajasthan Royals, the data points to three pressure gaps: finishing at the death, middle-overs control "
        "with the ball, and death bowling. Section 8 maps each pick to them.",
        "Robust names, in the top 10 under at least 9 of 10 alternative settings: Tim David, Shreyas Iyer, "
        "Suryakumar Yadav; Mitchell Santner, Sunil Narine, Josh Hazlewood, Krunal Pandya."])
    st.append(PageBreak())

    # ---------------- 1. data
    st.append(P("1. Data and preparation", "h1"))
    st.append(P("The data pack holds 883 matches (2022–2026). 867 have ball-by-ball records: 367 IPL, 470 T20I and "
                "30 A-team T20s (Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025). The format code “T20” in "
                "<i>competitions.csv</i> covers both the IPL and the A-team events, so IPL is identified by competition "
                "name, and the A-team matches are grouped with the T20Is. In total there are 200,961 deliveries in "
                "the two main innings."))
    st.append(P("Rebuilding the match state", "h2"))
    st += bullets([
        "For every ball we rebuilt the state <i>before</i> it: score, wickets, legal balls left, target, runs needed, "
        "required and current run rate, and phase (powerplay overs 1–6, middle 7–15, death 16–20).",
        "Wides do not count as balls faced; no-balls do. Byes and leg-byes are not charged to the bowler; run-outs are "
        "not credited to the bowler. Extras labelled “other” behave like no-balls (overs containing them have seven "
        "counted balls) and are treated as no-balls. Super-over innings are dropped.",
        "<b>Data gap.</b> The pack records the <i>type</i> of each extra but not how many runs it produced. We "
        "imputed them from strike rotation (batters crossing means an odd number of runs). Against official results, "
        "wicket margins then match exactly in 99.7% of matches and run margins in 55% (77% within ±3 runs).",
        "<b>Target reconciliation.</b> In 11% of matches the rebuilt chase total contradicted the official winner. We "
        "set each chase target so that it agrees with the official result and margin (32% of targets moved, median "
        "3 runs), and spread the difference evenly across the first innings.",
        "<b>Exclusions.</b> 56 matches are excluded from situation modelling: revised overs / DLS, no result, and "
        "incomplete records (an innings that stops early with wickets in hand). 811 matches remain."])
    st.append(fig(OUT / "eda_pressure.png", 174,
                  "Figure 1. First look (chases, 811 matches). As the required rate climbs, scoring rises a little but "
                  "dismissals more than double; behaviour changes sharply with four or fewer wickets left; IPL scores "
                  "faster than T20I in every phase, so each format needs its own baseline."))

    # ---------------- 2. pressure
    st.append(P("2. Defining pressure", "h1"))
    st.append(P("The brief lists the required run rate, wickets in hand, balls remaining, the stage of the innings and "
                "how close the contest is. Each of these matters only through one thing: how much the next ball can "
                "change who wins. So we measure that directly."))
    st.append(fig(REP / "fig_method.png", 172, "Figure 2. The method end to end."))
    st.append(P("2.1 Win-probability model", "h2"))
    st += bullets([
        "<b>Chase.</b> A gradient-boosted classifier predicts whether the chasing side wins from runs needed, balls "
        "left, wickets in hand, required rate, format, the IPL Impact Player era (2023 onwards) and team strength. "
        "Monotonic constraints keep cricket logic (more runs needed never raises win probability), and predictions "
        "are recalibrated on out-of-fold data.",
        "<b>First innings.</b> We project the final total (expected remaining runs and their spread, given balls "
        "left and wickets in hand) and ask the chase model how likely the opposition is to chase it. At the innings "
        "break both views agree exactly (average gap 0.01 percentage points), so no ball gets artificial leverage "
        "at the changeover.",
        "<b>Team strength.</b> A pre-match Elo rating, computed only from earlier results and separately for IPL and "
        "internationals, so that India vs an associate side is not treated as a close contest. It improves "
        "first-innings accuracy (log-loss 0.623 vs 0.631 without it).",
        "<b>Accuracy</b>, cross-validated by match (a match is never in its own training data): chase AUC 0.90, "
        "log-loss 0.39 (no-skill 0.69); first innings AUC 0.72, log-loss 0.62."])
    st.append(fig(REP / "fig_calibration.png", 88,
                  "Figure 3. Calibration: predicted vs actual win rates, out-of-fold, in 10 bins per innings."))
    st.append(P("2.2 Leverage and the pressure threshold", "h2"))
    st.append(P("For each ball we list its possible outcomes (wicket, wide/no-ball, 0, 1, 2, 3, 4, 6 runs), take their "
                "probabilities in that situation (format, innings, phase, wickets, required-rate band), and compute "
                "the expected absolute change in win probability. Dividing by the average ball gives <b>leverage</b>: "
                "1.0 is an average ball, 3.0 is three times as much at stake. A <b>pressure ball</b> has leverage "
                "≥ 1.5; that is 16.5% of balls (IPL 18.5%, T20I 14.9%: IPL games are closer on average)."))
    st.append(fig(OUT / "fig_pressure_definition.png", 174,
                  "Figure 4. Left: in a chase, pressure concentrates where the game is close late on. Right: average "
                  "leverage by over; the end of a chase carries over three times the stakes of an average ball."))
    st.append(P("2.3 Does it separate pressure from ordinary play?", "h2"))
    st += bullets([
        "In the top leverage band (≥ 2.5) dismissals rise to 6.7 (IPL) and 7.4 (T20I) per 100 balls, against "
        "4.4–6.0 in lower bands, and dot balls fall by three to five points (Figure 5).",
        "Part of that is simply the death overs. Comparing pressure and ordinary balls <i>within the same format, "
        "innings and phase</i>, pressure balls still show +0.4 dismissals per 100 balls and 1.8 points fewer dots: "
        "batters take more risk. The effect is real but modest.",
        "Simple rules miss most pressure: 70% of pressure balls carry none of the tags below. The tags are reported "
        "for readability only: chase with required rate above 10 in overs 16–20 (mean leverage 2.0), death overs "
        "with win probability 25–75% (1.7), collapse of 3+ wickets for 40 runs or fewer (0.8), knockout match (1.0)."])
    st.append(fig(OUT / "fig_pressure_validation.png", 174,
                  "Figure 5. Batting outcomes by leverage band, per format."))
    st.append(fig(OUT / "fig_example_match.png", 160,
                  "Figure 6. One knockout match: win probability of the side batting first (top) and leverage of "
                  "each ball (bottom; orange = pressure ball)."))

    # ---------------- 3. performance
    st.append(P("3. Measuring performance", "h1"))
    st.append(P("Every metric compares what a player did on a ball with what an average player did on the same kind "
                "of ball: same format, innings, phase, wickets band, required-rate band <i>and</i> the same pressure "
                "status. A pressure ball is compared only with other pressure balls. Sparse situations borrow "
                "strength from the broader format-innings-phase average."))
    st.append(table([
        ["", "Metric (per 100 balls)", "What it captures"],
        ["Batting", "Runs above expected", "Scoring compared with the same situation"],
        ["", "Survival (dismissals below expected)", "Not getting out when it matters"],
        ["", "Win probability added (WPA), situation-centred", "Runs and wickets priced in match terms"],
        ["Bowling", "Runs saved vs expected (bowler-charged runs)", "Containment"],
        ["", "Dot balls above expected (percentage points)", "Building pressure back"],
        ["", "Wickets above expected (run-outs excluded)", "Taking wickets"],
        ["", "WPA taken from the batting side, situation-centred", "Overall match impact"],
    ], [20, 72, 82]))
    st.append(Spacer(1, 4))
    st.append(P("Comparison with the player's usual level", "h2"))
    st.append(P("The brief asks how players perform under pressure <i>compared with their usual level</i>. Every "
                "metric is therefore computed twice, on the player's pressure balls and on their ordinary balls, and "
                "the difference gives a <b>holds-level score</b> (average z-score of pressure minus ordinary for "
                "WPA, runs and survival or wickets). It appears in every evidence card. We tested giving it a rating "
                "weight and chose not to (Section 6.4): a difference of two noisy numbers proved the least repeatable "
                "quantity in the data. Including it at 20% would keep 8 of the 10 batters and 7 of the 10 bowlers."))
    st.append(P("Rating", "h2"))
    st.append(P("Each eligible player's (shrunk) pressure metrics are converted to z-scores among eligible players and "
                "combined with weights chosen for repeatability (weighting C, Section 6.4):"))
    st.append(table([
        ["", "Runs above expected / runs saved", "Survival / dots", "Wickets above expected", "WPA"],
        ["Batting", "40%", "30%", "–", "30%"],
        ["Bowling", "35%", "25%", "15%", "25%"],
    ], [24, 50, 34, 36, 30]))
    st.append(Spacer(1, 3))
    st.append(fig(REP / "fig_components.png", 174,
                  "Figure 7. Each top-10 player's rating split into its weighted components."))

    # ---------------- 4. fair comparisons
    st.append(P("4. Fair comparisons", "h1"))
    st += bullets([
        f"<b>Eligibility.</b> At least 120 pressure balls faced or bowled, and the player must be able to enter the "
        f"IPL 2027 auction: players who have appeared for a Pakistan side in the data are excluded, because Pakistan "
        f"players have not taken part in the IPL since 2008. That leaves {len(rb)} batters and {len(rw)} bowlers. "
        "Lowering the bar to 90 balls changes at most one name; raising it to 150 changes up to three (Table 2).",
        "<b>Small samples.</b> Empirical-Bayes shrinkage pulls each player's rate toward the overall average by "
        "k / (n + k), where n is their balls and k is estimated from the data as ball-to-ball noise divided by the "
        "true spread between players. For pressure WPA, k ≈ 1,150 balls (batting) and 1,360 (bowling): a player with 200 "
        "pressure balls keeps only about one-seventh of their raw WPA. Runs-based metrics are steadier (k ≈ 175–250).",
        "<b>IPL and T20I.</b> Every baseline is format-specific, so a T20I player is never judged against IPL "
        "scoring. The win-probability model adds the Impact Player era and team strength, and each card shows "
        "the IPL share and the average opposition edge (+ = weaker opponents), so weak-opposition samples are visible.",
        "<b>Roles.</b> Phase is part of every baseline, so a death bowler is compared with death bowling and a "
        "finisher with finishing situations. We shrink toward the overall average, not the role average: with this "
        "much noise, a role prior ranks the role instead of the player. Bowler type is not in the baseline; "
        "middle-overs spinners do beat the situation average, and we count that as genuine value."])

    # ---------------- 5. evidence
    st.append(PageBreak())
    st.append(P("5. Evidence for the top 10", "h1"))
    st.append(P("Shrunk values per 100 pressure balls unless marked raw. SR = strike rate. Rank interval and resample "
                "share come from 200 match-level bootstraps; “scenarios” counts how many of 10 alternative settings "
                "keep the player in the top 10."))
    st.append(P("Batters", "h2"))
    st.append(cards(rb, d["mb"], lb, card_bat))
    st.append(P("Bowlers", "h2"))
    st.append(cards(rw, d["mw"], lw, card_bowl))

    # ---------------- 6/7 reliability
    st.append(P("6. Reliability", "h1"))
    st.append(P("6.1 How certain is each rank?", "h2"))
    st.append(P("We resampled whole matches with replacement 200 times (balls within a match are not independent) and "
                "recomputed everything each time, including the shrinkage strength. Figure 8 shows each player's 90% "
                "rank interval and the share of resamples that put them in the top 10."))
    st.append(fig(OUT / "fig_rank_intervals.png", 174,
                  "Figure 8. Rank uncertainty for the current top 20."))
    st.append(P("6.2 Consistency across matches", "h2"))
    st.append(P("Each card reports the share of a player's pressure matches with positive WPA. Top-10 players range "
                "from about 37% to 69%; even the best players have many pressure matches that go against them. "
                "That is why the rating uses per-ball rates over many matches rather than memorable single games."))
    st.append(P("6.3 Does pressure performance persist?", "h2"))
    hb, rows_bh = d["split"]["bat"]
    hw, rows_wh = d["split"]["bowl"]

    def split_rows(rows, labels):
        out = []
        for r in rows:
            parts = r.split()
            name = "2022–24 vs 2025–26" if r.strip().startswith("2022") else "random halves"
            vals = parts[-(len(labels)):]
            out.append([name] + [v.replace("-", "−") for v in vals])
        return out

    lab_b = ["players", "rating", "WPA pressure", "WPA ordinary", "runs pressure", "runs ordinary",
             "survival pressure", "survival ordinary"]
    lab_w = ["players", "rating", "WPA pressure", "WPA ordinary", "runs saved pressure", "runs saved ordinary",
             "wickets pressure", "wickets ordinary", "dots pressure", "dots ordinary"]
    st.append(P("We rated players separately on two halves of the matches (by period, and at random) and correlated "
                "the results for players with 60+ pressure balls in both halves (Spearman). Table 1 is the most "
                "important honesty check in this report."))
    st.append(table([["Batting"] + lab_b] + split_rows(rows_bh, lab_b), [28] + [18.25] * 8))
    st.append(Spacer(1, 4))
    st.append(table([["Bowling"] + lab_w] + split_rows(rows_wh, lab_w), [26] + [14.8] * 10, font=7.5))
    st.append(P("Table 1. Split-half correlations under the final weights. Ordinary-ball skill repeats best; "
                "runs saved and dots under pressure repeat moderately; pressure WPA hardly repeats.", "cap"))
    st.append(P("6.4 Choosing the weights", "h2"))
    st.append(P("Because Table 1 showed that the components differ so much in repeatability, we compared five "
                "weightings on 20 random splits plus the period split, and on bootstrap stability (Figure 9). For "
                "bowlers, moving weight from WPA and the holds-level score to runs saved and dots lifts the period "
                "correlation from +0.17 to +0.46 and the random-halves correlation from +0.01 to +0.17. For batters no weighting makes the rating repeat; we chose "
                "the same principle (C) for consistency."))
    st.append(fig(REP / "fig_weights.png", 174,
                  "Figure 9. A = initial WPA-led weighting with holds-level term; B = A without holds-level; C = "
                  "runs/survival or runs/dots-led (chosen); D = no WPA; E = weights proportional to each component's "
                  "own repeatability."))
    st.append(P("6.5 Sensitivity", "h2"))
    sb, sw = d["sens"]["bat"], d["sens"]["bowl"]
    rows = [["Change", "Batters kept", "Bowlers kept"]]
    for (n1, k1, _), (n2, k2, _) in zip(sb, sw):
        rows.append([n1, f"{k1}/10", f"{k2}/10"])
    st.append(table(rows, [90, 30, 30]))
    st.append(P("Table 2. How many of each final top 10 survive each change. The lists are fairly robust to weights and "
                "eligibility; they are most sensitive to the pressure threshold, the shrinkage target and the "
                "team-strength term in the win-probability model.", "cap"))

    # ---------------- 7 limitations
    st.append(P("7. Limitations", "h1"))
    st += bullets([
        "<b>Batting pressure skill is not separable from noise in this data.</b> The batting rating does not repeat "
        "across periods (−0.21; +0.18 between random halves). The list reflects who performed best under pressure "
        "in 2022–26, after shrinkage, not a proven stable trait. The bowling rating repeats (+0.32 to +0.46).",
        "<b>Rank uncertainty.</b> 90% rank intervals typically span 1 to about 30–50. Ranks 6–10 in each list are "
        "close calls, and several players just outside the lists have similar evidence.",
        "<b>Choices that move names.</b> A stricter threshold (2.0) changes 6 batters and 4 bowlers; removing the "
        "team-strength term changes 5 batters and 4 bowlers; shrinking toward role averages changes 6 batters. We kept both choices fixed in advance rather than tuning to a preferred list.",
        "<b>Missing extras runs</b> were imputed; run margins match exactly in only 55% of matches before "
        "reconciliation. This adds noise to runs needed late in some chases.",
        "<b>Small and uneven samples.</b> Ashutosh Sharma (batting #2) has 125 pressure balls and Mohsin Khan "
        "(bowling #9) 136, with a rank interval of 1–67. Shrinkage limits but cannot remove this.",
        "<b>T20I-only evidence.</b> Towhid Hridoy, Reeza Hendricks and Taskin Ahmed have no IPL matches in the "
        "data; their ratings rest on T20Is, so their IPL value is less certain.",
        "<b>Context we cannot see:</b> pitch and venue, dew, match-ups, injuries, and fielding. Win probability "
        "treats every 200-run target the same at every ground.",
        "<b>Role and opposition.</b> Spinners outperform the situation average in middle-overs pressure; bowler type "
        "is not in the baseline. Elo for associate teams rests on few matches."])

    st += rr_section(rb, rw)

    # ---------------- appendix
    st.append(P("Appendix: reproducing the results", "h1"))
    st.append(P("All code is in <i>src/</i>; run from the project folder with the data pack's CSVs in <i>data/</i> "
                "(Python 3.11, pandas, numpy, scikit-learn, matplotlib, reportlab). Random seeds are fixed."))
    st.append(table([
        ["Step", "Command", "Produces"],
        ["Clean + validate", "python src/clean.py; python src/validate.py", "ball-state table, margin check"],
        ["Win probability + leverage", "python src/winprob.py", "per-ball WP, WPA, leverage, tags"],
        ["Pressure checks", "python src/pressure_check.py", "Figures 4–6, pressure tables"],
        ["Metrics + rating", "python src/metrics.py; python src/rating.py", "player metrics, ranked lists"],
        ["Reliability", "python src/reliability.py; python src/weights_test.py", "Figures 8–9, Tables 1–2"],
        ["Squad fit", "python src/squad_fit.py", "Tables 3–4 (Rajasthan Royals)"],
        ["Report", "python src/report.py", "this PDF and the dropdown order"],
    ], [38, 76, 60]))
    st.append(Spacer(1, 4))
    st.append(P("Key parameters: pressure = leverage ≥ 1.5; eligibility ≥ 120 pressure balls; shrinkage toward the "
                "overall average with data-estimated strength; weighting C as in Section 3; 200 bootstrap resamples."))

    doc = SimpleDocTemplate(str(REP / "SupeRR_S2_Round1_Methodology.pdf"), pagesize=A4, leftMargin=18 * mm,
                            rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=17 * mm,
                            title="SupeRR Selector S2 - Round 1 methodology report")
    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print("built", REP / "SupeRR_S2_Round1_Methodology.pdf")


if __name__ == "__main__":
    build()
