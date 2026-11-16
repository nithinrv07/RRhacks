"""
Step 1b: check the rebuilt ball-state table against the official result text
in matches.csv ("Won by N Runs" / "Won by N Wickets").
"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

tot = pd.read_csv(OUT / "innings_totals.csv")
m = pd.read_csv(ROOT / "data" / "matches.csv", keep_default_na=False, na_values=[""])
m = m[m.first_inn_revised_overs.isna() & m.second_inn_revised_overs.isna()]

p = tot.pivot(index="match_id", columns="innings", values=["inn_total", "inn_wkts"])
p.columns = [f"{a}_{b}" for a, b in p.columns]
d = m.set_index("match_id")[["result"]].join(p, how="inner")

rows = []
for mid, r in d.iterrows():
    runs = re.search(r"by (\d+) Run", r.result, re.I)
    wk = re.search(r"by (\d+) Wicket", r.result, re.I)
    if runs:
        rows.append(("runs", int(runs.group(1)), r.inn_total_1 - r.inn_total_2, mid))
    elif wk and pd.notna(r.inn_wkts_2):
        rows.append(("wickets", int(wk.group(1)), 10 - r.inn_wkts_2, mid))
v = pd.DataFrame(rows, columns=["type", "official", "rebuilt", "match_id"])
v["err"] = v.rebuilt - v.official

print("=== Validation vs official result margins ===")
for t, g in v.groupby("type"):
    print(f"\n{t}: n={len(g)}  exact={g.err.eq(0).mean():.1%}  "
          f"within±3={g.err.abs().le(3).mean():.1%}  mean_err={g.err.mean():+.2f}")
    print(g.err.value_counts().sort_index().head(15).to_string())
v.to_csv(OUT / "validation.csv", index=False)
