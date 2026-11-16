"""
SupeRR Selector S2 - interactive dashboard (single self-contained HTML file).

Output: report/SupeRR_S2_Dashboard.html
Usage : python src/dashboard.py   (after report.py; reads only aggregated outputs, never the raw data pack)
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

import report as RP
from rating import WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
OUT, REP = ROOT / "outputs", ROOT / "report"


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def players(kind):
    r = pd.read_csv(OUT / f"rating_{'batters' if kind == 'bat' else 'bowlers'}.csv")
    m = pd.read_csv(OUT / f"{'batting' if kind == 'bat' else 'bowling'}_metrics.csv").set_index("player_id")
    l = pd.read_csv(OUT / f"reliability_{'batters' if kind == 'bat' else 'bowlers'}.csv").set_index("player_id")
    f = pd.read_csv(OUT / "rr_player_fit.csv")
    f = f[f.kind == ("Batter" if kind == "bat" else "Bowler")].set_index("name")
    keep = ["player_id", "rank", "name", "role", "n_p", "matches_p", "ipl_share_p", "opp_edge_p", "rating", "hold",
            "z_hold"]
    if kind == "bat":
        keep += ["rae_p", "surv_p", "wpa_p", "wpa_p_raw", "rae_p_raw", "rae_np_raw", "z_rae_p", "z_surv_p", "z_wpa_p"]
        mk = ["sr_p", "sr_np", "runs_p", "outs_p"]
    else:
        keep += ["rsave_p", "dot_p", "wae_p", "wpa_p", "wpa_p_raw", "rsave_p_raw", "rsave_np_raw", "z_rsave_p", "z_dot_p",
                 "z_wae_p", "z_wpa_p"]
        mk = ["econ_p", "econ_np", "wkts_p"]
    out = []
    for _, p in r[keep].iterrows():
        d = {k: clean(p[k]) for k in keep}
        for k in mk:
            d[k] = clean(m.loc[p.player_id, k])
        if p.player_id in l.index:
            q = l.loc[p.player_id]
            for k in ("rank_p05", "rank_p95", "p_top10", "share_matches_positive", "in_top10_of_scenarios"):
                d[k] = clean(q[k])
        if p["name"] in f.index and int(p["rank"]) <= 10:
            q = f.loc[p["name"]]
            d.update(best_phase=q.best_pressure_phase, best_phase_value=clean(q.best_phase_value),
                     ipl_team=q.latest_ipl_team, country=q.country, availability=q.availability)
        out.append(d)
    return out


def search_index(min_balls=60):
    """Every player with 60+ balls faced or bowled: raw pressure vs ordinary numbers for the search box."""
    from rating import INELIGIBLE
    mb = pd.read_csv(OUT / "batting_metrics.csv").set_index("player_id")
    mw = pd.read_csv(OUT / "bowling_metrics.csv").set_index("player_id")
    mb = mb[(mb.balls_p + mb.balls_np) >= min_balls]
    mw = mw[(mw.balls_p + mw.balls_np) >= min_balls]
    out = []
    for pid in sorted(set(mb.index) | set(mw.index)):
        name = mb.loc[pid, "name"] if pid in mb.index else mw.loc[pid, "name"]
        rec = {"id": int(pid), "name": name, "excluded": pid in INELIGIBLE}
        if pid in mb.index:
            q = mb.loc[pid]
            rec["bat"] = {k: clean(q[k]) for k in ("balls_p", "balls_np", "runs_p", "outs_p", "sr_p", "sr_np",
                                                   "rae_100_p", "rae_100_np", "oae_100_p", "oae_100_np", "wpa_100_p",
                                                   "wpa_100_np", "matches", "matches_p", "ipl_share", "usual_position")}
        if pid in mw.index:
            q = mw.loc[pid]
            rec["bowl"] = {k: clean(q[k]) for k in ("balls_p", "balls_np", "wkts_p", "wkts_np", "econ_p", "econ_np",
                                                    "rsave_100_p", "rsave_100_np", "dotae_pct_p", "dotae_pct_np",
                                                    "wpa_100_p", "wpa_100_np", "matches", "matches_p", "ipl_share",
                                                    "pace_or_spin")}
        out.append(rec)
    return out


def pressure_curve():
    """Mean leverage by over and innings (the hero chart)."""
    parts = []
    for ch in pd.read_csv(OUT / "balls_wp.csv.gz", usecols=["innings", "over_no", "leverage", "exclude"],
                          chunksize=50000):
        parts.append(ch[~ch.exclude.astype(bool)].groupby(["innings", "over_no"]).leverage.agg(["sum", "count"]))
    g = pd.concat(parts).groupby(level=[0, 1]).sum()
    m = (g["sum"] / g["count"]).unstack(0)
    return {"over": [int(o) for o in m.index], "inn1": [round(v, 3) for v in m[1]], "inn2": [round(v, 3) for v in m[2]]}


def img(name, folder):
    return "data:image/png;base64," + base64.b64encode((folder / name).read_bytes()).decode()


def main():
    d = RP.load()
    data = {
        "bat": players("bat"), "bowl": players("bowl"), "all": search_index(), "curve": pressure_curve(),
        "weights": WEIGHTS,
        "sens": {k: [{"change": a, "kept": int(b)} for a, b, _ in v] for k, v in d["sens"].items()},
        "rr": pd.read_csv(OUT / "rr_pressure_profile.csv").round(3).to_dict("records"),
        "img": {"method": img("fig_method.png", REP), "definition": img("fig_pressure_definition.png", OUT),
                "validation": img("fig_pressure_validation.png", OUT), "calibration": img("fig_calibration.png", REP),
                "weights": img("fig_weights.png", REP)},
    }
    html = (ROOT / "src" / "site_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    (REP / "SupeRR_S2_Dashboard.html").write_text(html, encoding="utf-8")
    print("built", REP / "SupeRR_S2_Dashboard.html", f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
