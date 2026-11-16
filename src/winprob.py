"""
SupeRR Selector S2 - Step 2: win-probability model + per-ball leverage (pressure).

Input : outputs/balls_clean.csv.gz   (from clean.py)
Output: outputs/balls_wp.csv.gz      (every ball + wp_before, wp_after, wpa, leverage, tags)
        outputs/wp_metrics.txt       (cross-validated model quality)

Model
  * Two gradient-boosted classifiers (one per innings), target = batting team wins.
      1st innings: score, wickets in hand, balls left, format, IPL impact-player era, team strength
      2nd innings: runs needed, balls left, wickets in hand, required rate, target, format, era, strength
  * Monotonic constraints keep cricket logic intact (more runs needed -> lower WP, etc.).
  * Team strength = pre-match Elo difference (leak-free: uses only earlier results).
    Run with --no-elo to switch it off (sensitivity check for the report).
  * Ties / super-over matches count as half a win for each side.

Leverage (pressure) of a ball = expected absolute swing in win probability
  LI_raw = sum over outcomes o of P(o | situation) * |WP(state after o) - WP(state now)|
  leverage = LI_raw / mean(LI_raw)   ->  1.0 = an average ball, 2.0 = twice the average swing.
Outcomes: wicket, extra (wide/no-ball), and 0/1/2/3/4/6 runs off a legal ball.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"

F1 = ["score", "wih", "balls_left", "ipl", "impact_era", "elo"]
M1 = [1, 1, 1, 0, 0, 1]
F2 = ["runs_needed", "balls_left", "wih", "rrr", "ipl", "impact_era", "elo"]
M2 = [-1, 1, 1, -1, 0, 0, 1]   # no free "target" term: keeps WP strictly monotone in the target
HGB = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
           min_samples_leaf=300, l2_regularization=1.0, random_state=0)


# ---------------------------------------------------------------- team strength
def elo_table(k=24, home_adv=0.0):
    """Pre-match Elo for both teams, computed in date order, separately per format."""
    m = pd.read_csv(DATA / "matches.csv")
    c = pd.read_csv(DATA / "competitions.csv")
    c = c.assign(match_format=np.where(c.comp_name.str.startswith("IPL"), "IPL", "T20I"))   # see clean.py
    m = m.merge(c[["comp_id", "match_format"]], on="comp_id").sort_values(["match_date", "match_id"])
    rating, rows = {}, []
    for r in m.itertuples():
        key_h, key_a = (r.match_format, r.home_team_id), (r.match_format, r.away_team_id)
        rh, ra = rating.get(key_h, 1500.0), rating.get(key_a, 1500.0)
        rows.append((r.match_id, r.home_team_id, rh, r.away_team_id, ra))
        if pd.isna(r.win_team_id):
            continue                                   # no result -> no update
        exp_h = 1 / (1 + 10 ** ((ra - rh - home_adv) / 400))
        s_h = 1.0 if r.win_team_id == r.home_team_id else 0.0 if r.win_team_id == r.away_team_id else 0.5
        rating[key_h] = rh + k * (s_h - exp_h)
        rating[key_a] = ra - k * (s_h - exp_h)
    e = pd.DataFrame(rows, columns=["match_id", "home", "elo_home", "away", "elo_away"])
    return e


def add_features(b, use_elo=True):
    b = b.copy()
    b["wih"] = 10 - b.wkts_before
    # Spread the reconciliation gap (reconciled target - 1 - rebuilt 1st-innings total) evenly over the
    # first innings, so the innings ends exactly where the chase starts (median gap ~3 runs).
    tgt = b[b.innings == 2].groupby("match_id").target.first()
    i1 = b[b.innings == 1].groupby("match_id").agg(tot=("total_runs", "sum"), legal=("legal", "sum"))
    gap = (tgt - 1 - i1.tot).reindex(i1.index).fillna(0)
    frac = b.legal_before / b.match_id.map(i1.legal).clip(lower=1)
    b["score"] = np.where(b.innings == 1, b.score_before + b.match_id.map(gap).fillna(0) * frac, np.nan)
    b["inn1_gap_per_ball"] = np.where(b.innings == 1, b.match_id.map(gap).fillna(0) / b.match_id.map(i1.legal).clip(lower=1), 0)
    b["final1"] = np.where(b.innings == 1, b.match_id.map(tgt) - 1, np.nan)
    b["ipl"] = (b.fmt == "IPL").astype(int)
    b["impact_era"] = ((b.fmt == "IPL") & (b.year >= 2023)).astype(int)
    if use_elo:
        e = elo_table()
        b = b.merge(e, on="match_id", how="left")
        own = np.where(b.batting_team_id == b.home, b.elo_home, b.elo_away)
        opp = np.where(b.batting_team_id == b.home, b.elo_away, b.elo_home)
        b["elo"] = (own - opp) / 100.0            # in units of 100 Elo points
        b = b.drop(columns=["home", "away", "elo_home", "elo_away"])
    else:
        b["elo"] = 0.0
    return b


# ---------------------------------------------------------------- training
def labels(df):
    """Expand tied matches into a half-win and half-loss row (via sample weight)."""
    y = df.bat_team_won.astype(int).to_numpy()
    w = np.ones(len(df))
    tie = df.tie.to_numpy()
    X = pd.concat([df, df[tie]], ignore_index=True)
    y = np.concatenate([np.where(tie, 1, y), np.zeros(tie.sum(), int)])
    w = np.concatenate([np.where(tie, 0.5, w), np.full(tie.sum(), 0.5)])
    return X, y, w


def fit(df, feats, mono):
    X, y, w = labels(df)
    mdl = HistGradientBoostingClassifier(monotonic_cst=mono, **HGB)
    mdl.fit(X[feats], y, sample_weight=w)
    return mdl


def cross_validate(df, feats, mono, name, folds=5):
    """Match-grouped CV: the model never sees any ball of a test match."""
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(folds).split(df, groups=df.match_id):
        mdl = fit(df.iloc[tr], feats, mono)
        oof[te] = mdl.predict_proba(df.iloc[te][feats])[:, 1]
    res = scores(df, oof, name)
    return oof, res


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def platt(oof, df):
    """2-parameter recalibration fitted on out-of-fold predictions (keeps the ranking, fixes over-confidence)."""
    keep = ~df.tie.to_numpy()
    lr = LogisticRegression(C=1e6).fit(logit(oof[keep])[:, None], df.bat_team_won.to_numpy()[keep].astype(int))
    return lambda p: lr.predict_proba(logit(p)[:, None])[:, 1]


def scores(df, p, name):
    keep = ~df.tie.to_numpy()
    y = df.bat_team_won.to_numpy()[keep].astype(int)
    p = p[keep]
    return {"model": name, "balls": int(keep.sum()), "matches": df.match_id.nunique(),
            "log_loss": log_loss(y, p), "log_loss_baseline": log_loss(y, np.full_like(p, y.mean())),
            "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p)}


FR = ["balls_left", "wih", "ipl", "impact_era", "elo"]
MR = [1, 1, 0, 0, 1]
GH_Z = np.array([-2.856970, -1.355626, 0.0, 1.355626, 2.856970])     # Gauss-Hermite nodes for N(0,1)
GH_W = np.array([0.011257, 0.222076, 0.533333, 0.222076, 0.011257])


def fit_projection(t1):
    """Remaining-runs model for the 1st innings: mean and spread of (final total - current score)."""
    d = t1.dropna(subset=["final1"])
    y = d.final1 - d.score
    mu = HistGradientBoostingRegressor(monotonic_cst=MR, **HGB).fit(d[FR], y)
    r2 = (y - mu.predict(d[FR])) ** 2
    var = HistGradientBoostingRegressor(**HGB).fit(d[FR], r2)
    return mu, var


# ---------------------------------------------------------------- WP evaluation
class WP:
    """Win probability of the batting side.
    2nd innings: the chase model directly.
    1st innings: project the final total (mean +/- spread of remaining runs), then ask the chase model how
    likely the opposition is to chase it. At the end of the innings this collapses to the chase model
    exactly, so the two innings join up with no seam."""
    def __init__(self, m2, c2, mu, var):
        self.m2, self.c2, self.mu, self.var = m2, c2, mu, var

    def chase(self, runs_needed, balls_left, wih, target, ipl, era, elo):
        n = len(runs_needed)
        runs_needed, balls_left, wih = map(np.asarray, (runs_needed, balls_left, wih))
        out = np.full(n, np.nan)
        won = runs_needed <= 0
        over = ~won & ((wih <= 0) | (balls_left <= 0))
        out[won] = 1.0
        out[over] = np.where(runs_needed[over] == 1, 0.5, 0.0)   # level scores -> tie
        live = ~(won | over)
        if live.any():
            X = pd.DataFrame({
                "runs_needed": runs_needed[live], "balls_left": balls_left[live], "wih": wih[live],
                "rrr": runs_needed[live] * 6 / np.maximum(balls_left[live], 1),
                "target": np.asarray(target)[live], "ipl": np.asarray(ipl)[live],
                "impact_era": np.asarray(era)[live], "elo": np.asarray(elo)[live]})
            out[live] = self.c2(self.m2.predict_proba(X[F2])[:, 1])
        return out

    def first(self, score, wih, balls_left, ipl, era, elo):
        score, wih, balls_left = (np.asarray(x, float) for x in (score, wih, balls_left))
        ipl, era, elo = (np.asarray(x, float) for x in (ipl, era, elo))
        n = len(score)
        done = (wih <= 0) | (balls_left <= 0)
        X = pd.DataFrame({"balls_left": np.maximum(balls_left, 0), "wih": np.maximum(wih, 0),
                          "ipl": ipl, "impact_era": era, "elo": elo})
        mu = np.where(done, 0.0, np.maximum(self.mu.predict(X[FR]), 0))
        sd = np.where(done, 0.0, np.sqrt(np.maximum(self.var.predict(X[FR]), 1.0)))
        out = np.zeros(n)
        for z, w in zip(GH_Z, GH_W):
            t = score + np.maximum(mu + z * sd, 0) + 1                 # target the chasing side would face
            out += w * (1 - self.chase(t, np.full(n, 120), np.full(n, 10), t, ipl, era, -elo))
        return out

    def state(self, b, d_runs=0, d_wkts=0, d_balls=0):
        """WP of the batting team for each ball's state shifted by the given deltas."""
        d_runs, d_wkts, d_balls = (np.broadcast_to(np.asarray(x), len(b)) for x in (d_runs, d_wkts, d_balls))
        out = np.full(len(b), np.nan)
        i1 = (b.innings == 1).to_numpy()
        i2 = ~i1
        if i1.any():
            s = b[i1]
            out[i1] = self.first(s.score.to_numpy() + d_runs[i1] + s.inn1_gap_per_ball.to_numpy() * d_balls[i1],
                                 s.wih.to_numpy() - d_wkts[i1],
                                 s.balls_left.to_numpy() - d_balls[i1], s.ipl, s.impact_era, s.elo)
        if i2.any():
            s = b[i2]
            out[i2] = self.chase(s.runs_needed.to_numpy() - d_runs[i2], s.balls_left.to_numpy() - d_balls[i2],
                                 s.wih.to_numpy() - d_wkts[i2], s.target, s.ipl, s.impact_era, s.elo)
        return out


# ---------------------------------------------------------------- outcome model for leverage
OUTCOMES = {  # name: (runs, wickets, legal balls used)
    "W": (0, 1, 1), "X": (1, 0, 0), "0": (0, 0, 1), "1": (1, 0, 1), "2": (2, 0, 1),
    "3": (3, 0, 1), "4": (4, 0, 1), "6": (6, 0, 1)}


def outcome_class(b):
    r = b.total_runs.clip(upper=6).replace({5: 4})
    return np.select([b.is_wicket, ~b.legal], ["W", "X"], default=r.astype(int).astype(str))


def outcome_probs(b, prior=500.0):
    """P(outcome | format, innings, phase, wickets band, required-rate band).
    Thin cells are shrunk toward the coarser (format, innings, phase) distribution."""
    oc = pd.Series(outcome_class(b), index=b.index)
    wband = pd.cut(b.wkts_before, [-1, 2, 5, 10], labels=["0-2", "3-5", "6+"]).astype(str)
    rband = np.where(b.innings == 2, pd.cut(b.rrr, [-1, 6, 8, 10, 12, 15, 999]).astype(str), "na")
    coarse = b.fmt + "|" + b.innings.astype(str) + "|" + b.phase.astype(str)
    fine = coarse + "|" + wband + "|" + rband
    tf, tc = pd.crosstab(fine, oc), pd.crosstab(coarse, oc)
    cols = tc.columns
    tf = tf.reindex(columns=cols, fill_value=0)
    Nf = tf.sum(axis=1).reindex(fine).to_numpy()[:, None]
    Pf = tf.div(tf.sum(axis=1), axis=0).reindex(fine).to_numpy()
    Pc = tc.div(tc.sum(axis=1), axis=0).reindex(coarse).to_numpy()
    lam = Nf / (Nf + prior)
    P = lam * Pf + (1 - lam) * Pc
    return pd.DataFrame(P, columns=cols, index=b.index)


# ---------------------------------------------------------------- pressure tags
def collapse_flag(b, wkts=3, runs=40):
    """True if >= `wkts` wickets have fallen for <= `runs` runs, ending at or before this ball."""
    flag = np.zeros(len(b), bool)
    for _, idx in b.groupby(["match_id", "innings"]).indices.items():
        s = b.iloc[idx]
        fow = []                                       # team score at each fall of wicket
        sb = s.score_before.to_numpy()
        wk = s.is_wicket.to_numpy()
        tr = s.total_runs.to_numpy()
        for j in range(len(s)):
            # score when the collapse began = score at the wicket before the last `wkts` (0 at innings start)
            if len(fow) >= wkts:
                start = fow[-wkts - 1] if len(fow) > wkts else 0
                if sb[j] - start <= runs:
                    flag[idx[j]] = True
            if wk[j]:
                fow.append(sb[j] + tr[j])
    return flag


def main(use_elo=True):
    b = pd.read_csv(OUT / "balls_clean.csv.gz", low_memory=False)
    b = add_features(b, use_elo)
    train = b[~b.exclude].reset_index(drop=True)
    t1, t2 = train[train.innings == 1], train[train.innings == 2]

    lines = []
    oof2, r2 = cross_validate(t2, F2, M2, "2nd innings (chase model)")
    c2 = platt(oof2, t2)
    oof2c = c2(oof2)
    # 1st innings, cross-validated by match: projection + chase model, both refitted without the test matches
    oof1 = np.zeros(len(t1))
    for tr, te in GroupKFold(5).split(t1, groups=t1.match_id):
        test_ids = set(t1.match_id.iloc[te])
        tr2 = t2[~t2.match_id.isin(test_ids)]
        mu, var = fit_projection(t1.iloc[tr])
        wpf = WP(fit(tr2, F2, M2), c2, mu, var)
        oof1[te] = wpf.state(t1.iloc[te])
    for r in (scores(t1, oof1, "1st innings (projected total -> chase model)"), r2,
              scores(t2, oof2c, "2nd innings, recalibrated")):
        lines.append(f"{r['model']}: balls={r['balls']:,} matches={r['matches']}  "
                     f"logloss={r['log_loss']:.3f} (no-skill {r['log_loss_baseline']:.3f})  "
                     f"brier={r['brier']:.3f}  AUC={r['auc']:.3f}")
    cal = pd.concat([t1.assign(p=oof1), t2.assign(p=oof2c)])
    cal = cal[~cal.tie]
    cal["bin"] = pd.cut(cal.p, np.linspace(0, 1, 11), include_lowest=True)
    ctab = cal.groupby(["innings", "bin"], observed=True).agg(pred=("p", "mean"), actual=("bat_team_won", "mean"),
                                                              balls=("p", "size"))
    lines += ["", "Calibration (out-of-fold, grouped by match):", ctab.round(3).to_string()]
    ctab.reset_index().to_csv(OUT / f"wp_calibration{'' if use_elo else '_noelo'}.csv", index=False)

    mu, var = fit_projection(t1)
    wp = WP(fit(t2, F2, M2), c2, mu, var)

    # --- WP before / after each ball, and WPA ---
    b["wp_before"] = wp.state(b)
    b["wp_after"] = wp.state(b, b.total_runs, b.is_wicket.astype(int), b.legal.astype(int))
    last2 = (b.innings == 2) & ~b.duplicated(["match_id", "innings"], keep="last")
    b.loc[last2, "wp_after"] = np.where(b.loc[last2, "tie"], 0.5, b.loc[last2, "bat_team_won"].astype(float))
    b["wpa"] = b.wp_after - b.wp_before                     # batting side's gain from this ball

    # --- leverage ---
    P = outcome_probs(b)
    swing = np.zeros(len(b))
    for oc, (r, w, l) in OUTCOMES.items():
        if oc in P:
            swing += P[oc].to_numpy() * np.abs(wp.state(b, r, w, l) - b.wp_before.to_numpy())
    b["li_raw"] = swing
    norm = b.loc[~b.exclude, "li_raw"].mean()
    b["leverage"] = b.li_raw / norm
    b.loc[b.exclude, ["wp_before", "wp_after", "wpa", "li_raw", "leverage"]] = np.nan

    # --- secondary, human-readable tags ---
    b["tag_chase_rrr10"] = (b.innings == 2) & (b.rrr > 10) & (b.over_no >= 16)
    b["tag_collapse"] = collapse_flag(b)
    b["tag_death_close"] = (b.over_no >= 16) & b.wp_before.between(0.25, 0.75)
    b["tag_knockout"] = b.knock_out.eq(1)

    # match-end consistency check: WP at the last first-innings ball vs WP at the start of the chase
    g = b[~b.exclude].groupby(["match_id", "innings"])
    e1 = g.tail(1).query("innings == 1").set_index("match_id").wp_after
    s2 = g.head(1).query("innings == 2").set_index("match_id").wp_before
    seam = (e1 - (1 - s2)).abs()
    lines += ["", f"Innings-break seam |WP end of 1st inns - WP start of chase|: mean {seam.mean():.4f}, max {seam.max():.4f}"]
    lines += ["", f"Leverage normaliser (mean raw swing per ball): {norm:.4f}",
              f"Elo strength term: {'ON' if use_elo else 'OFF'}"]
    q = b.loc[~b.exclude, "leverage"].quantile([.5, .75, .9, .95, .99])
    lines += ["Leverage quantiles: " + "  ".join(f"p{int(k*100)}={v:.2f}" for k, v in q.items())]

    keep_cols = [c for c in b.columns if c not in ("score",)]
    suffix = "" if use_elo else "_noelo"
    b[keep_cols].to_csv(OUT / f"balls_wp{suffix}.csv.gz", index=False)
    (OUT / f"wp_metrics{suffix}.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-elo", action="store_true", help="drop the team-strength term")
    main(use_elo=not ap.parse_args().no_elo)
