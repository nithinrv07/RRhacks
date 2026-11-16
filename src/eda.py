"""Step 1c: first look at how match situation changes batting outcomes."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
b = pd.read_csv(OUT / "balls_clean.csv.gz", low_memory=False)
b = b[~b.exclude & b.ball_faced]

def rates(df, key):
    g = df.groupby(key, observed=True)
    return pd.DataFrame({"balls": g.size(),
                         "run_rate": g.bat_runs.mean() * 6,
                         "dismissal_pct": g.is_wicket.mean() * 100,
                         "dot_pct": g.is_dot.mean() * 100,
                         "boundary_pct": g.is_boundary.mean() * 100})

chase = b[(b.innings == 2) & (b.balls_left > 0)].copy()
chase["rrr_band"] = pd.cut(chase.rrr, [-99, 6, 8, 10, 12, 15, 99],
                           labels=["<6", "6-8", "8-10", "10-12", "12-15", "15+"])
b["wkts_in_hand"] = 10 - b.wkts_before
tables = {
    "Chase: required run rate": rates(chase, "rrr_band"),
    "Wickets in hand": rates(b[b.wkts_in_hand >= 2], "wkts_in_hand"),
    "Phase x format": rates(b, ["fmt", "phase"]),
    "Knockout vs league": rates(b, ["fmt", "knock_out"]),
}
with open(OUT / "eda_tables.txt", "w") as f:
    for k, t in tables.items():
        s = f"\n=== {k} ===\n{t.round(2).to_string()}\n"
        print(s); f.write(s)

# ---- figure ----
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
t = tables["Chase: required run rate"]
ax[0].bar(t.index.astype(str), t.run_rate, color="#e5e7eb", label="Run rate (per over)")
a2 = ax[0].twinx(); a2.plot(t.index.astype(str), t.dismissal_pct, "o-", color="#d6336c", lw=2.5)
a2.set_ylabel("Dismissal % per ball", color="#d6336c"); ax[0].set_ylabel("Run rate")
ax[0].set_title("As the asking rate climbs, batters\nscore faster but get out far more", loc="left")
ax[0].set_xlabel("Required run rate (2nd innings)")

t = tables["Wickets in hand"]
ax[1].plot(t.index, t.run_rate, "o-", color="#254aa5", lw=2.5, label="Run rate")
a3 = ax[1].twinx(); a3.plot(t.index, t.dismissal_pct, "s--", color="#d6336c", lw=2, label="Dismissal %")
ax[1].invert_xaxis(); ax[1].set_xlabel("Wickets in hand"); ax[1].set_ylabel("Run rate")
a3.set_ylabel("Dismissal %", color="#d6336c")
ax[1].set_title("Losing wickets changes how batters play", loc="left")

t = tables["Phase x format"].reset_index()
for i, (fmt, c) in enumerate([("IPL", "#d6336c"), ("T20I", "#254aa5")]):
    s = t[t.fmt == fmt].set_index("phase").reindex(["powerplay", "middle", "death"])
    ax[2].bar(np.arange(3) + i * .38, s.run_rate, .38, color=c, label=fmt)
ax[2].set_xticks(np.arange(3) + .19, ["Powerplay", "Middle", "Death"])
ax[2].set_ylabel("Run rate"); ax[2].legend(frameon=False)
ax[2].set_title("IPL scores faster than T20I in every phase\n-> baselines must be format-specific", loc="left")
fig.tight_layout(); fig.savefig(OUT / "eda_pressure.png", dpi=160)
print("saved", OUT / "eda_pressure.png")
