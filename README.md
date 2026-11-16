# SupeRR Selector S2 – Round 1 pipeline

Put the data-pack CSVs in `data/` (never commit them – the pack must not be shared).

```
pip install pandas numpy matplotlib scikit-learn
python src/clean.py            # -> outputs/balls_clean.csv.gz (ball state per delivery)
python src/validate.py         # rebuilt totals vs official result margins
python src/eda.py              # -> outputs/eda_tables.txt, outputs/eda_pressure.png
python src/winprob.py          # -> outputs/balls_wp.csv.gz, wp_metrics.txt   (~10 min)
python src/winprob.py --no-elo # sensitivity variant without team strength    (~10 min)
python src/pressure_check.py   # -> fig_pressure_*.png, fig_example_match.png, pressure_tables.txt
python src/metrics.py          # -> batting_metrics.csv, bowling_metrics.csv, metrics_summary.txt
python src/rating.py           # -> rating_batters.csv, rating_bowlers.csv, rating_summary.txt
python src/reliability.py      # -> reliability_*.csv, reliability_summary.txt, fig_rank_intervals.png (~5 min)
python src/weights_test.py     # -> weights_test.txt (compares 5 weightings, ~10 min)
python src/squad_fit.py        # -> rr_pressure_profile.csv, rr_player_fit.csv
python src/report.py           # -> report/SupeRR_S2_Round1_Methodology.pdf, report/submission_order.txt
python src/dashboard.py        # -> report/SupeRR_S2_Dashboard.html (self-contained, aggregated data only)
```

## Day 1 – data findings (report: "Data & limitations")
- `extras1_runs` is an extras TYPE code, not runs. Runs from wides/byes/leg-byes are not recorded.
- Missing extra runs imputed from strike rotation (batters crossing == odd runs).
- Validation: wicket margins 99.7% exact; run margins 55% exact, 77% within ±3 runs.
- 'other' extras behave as no-balls; innings 3–4 are super overs (dropped).

## Day 2 – win probability and pressure
**Data fixes added on Day 2**
- Chase target reconciled with the official result (the rebuilt chase contradicted the winner in 11% of
  matches; 32% of targets moved, median 3 runs). The first-innings gap is spread evenly over its balls.
- Wide runs charged to the bowler use the imputed amount (was always 1).
- Excluded from situation modelling: revised/DLS, no result, and incomplete records (an innings that stops
  early with wickets in hand). 56 of 867 matches excluded; 811 used.

**Win-probability model** (`src/winprob.py`)
- Chase: gradient-boosted classifier on runs needed, balls left, wickets in hand, required rate, format,
  IPL impact-player era (2023+), team strength (pre-match Elo difference, leak-free). Monotonic
  constraints enforce cricket logic; Platt-recalibrated on out-of-fold predictions.
- 1st innings: project the final total (mean ± spread of remaining runs) and ask the chase model how
  likely the opposition is to chase it. This makes the two innings join exactly at the break.
- Match-grouped 5-fold CV: chase AUC 0.90, log-loss 0.40 (no-skill 0.69); 1st innings AUC 0.71,
  log-loss 0.62. Calibration table in `wp_calibration.csv`.

**Pressure = leverage**
- Leverage of a ball = expected absolute WP swing over its possible outcomes (wicket, extra, 0/1/2/3/4/6),
  with outcome probabilities by format/innings/phase/wickets/required rate, divided by the average ball.
- Default pressure ball: leverage ≥ 1.5 → 16.5% of balls (IPL 17%, T20I 16%).
- Secondary tags (for readability): chase RRR > 10 in overs 16–20, collapse (3+ wkts for ≤ 40),
  death overs with WP 25–75%, knockout.

**What the validation shows** (`pressure_tables.txt`)
- Mean leverage rises sharply at the end of a chase (3.4× average in over 20).
- In the top band (≥ 2.5) dismissals rise to ~7.2 per 100 balls (vs ~4.7–5.3 mid-range), dots fall.
- Within the same format/innings/phase, pressure balls still show +0.4 dismissals per 100 balls and
  −2.0 pts dot balls: the effect is real but modest once phase is controlled for.
- 70% of pressure balls carry none of the simple tags: leverage finds pressure rules of thumb miss.
- Eligibility at ≥ 120 pressure balls: 84 batters, 85 bowlers.

## Day 3 – player metrics (`src/metrics.py`)
Every number is relative to an average player in the same situation: format × innings × phase × wickets
band × required-rate band × pressure/ordinary. IPL and T20I each have their own baseline, death
specialists are not punished for death-overs numbers, and pressure balls are compared with other pressure balls.
- Batting: runs above expected /100 balls, dismissals above expected /100, WPA /100 (pct points, centred on
  the situation average), each on pressure and on the player's ordinary balls; drop_* = pressure − ordinary.
- Bowling: runs saved vs expected /100, wickets above expected /100, dot-ball % above expected, WPA /100.
- opp_edge_p: average Elo edge over the opposition on pressure balls (+ = weaker opposition).

## Day 4 – fair comparison and rating (`src/rating.py`)
- Eligibility: ≥ 120 pressure balls (84 batters, 89 bowlers). `--min-balls` to change.
- Empirical-Bayes shrinkage toward the OVERALL average, strength k estimated from the data (method of
  moments). WPA is very noisy per ball (k ≈ 700–900 balls). `--prior role` shrinks toward role averages
  instead; not the default because the ranking then mostly reflects the player's role.
- Roles are reported (batters by usual position; bowlers by pace/spin × main pressure phase). Spinners
  beat the situation average under pressure; bowler type is NOT in the baseline (a judgement call).
- Two biases found and fixed: raw WPA rewarded roles (fixed by centring WPA on the situation), and
  pressure balls were compared with all balls (fixed by adding pressure status to the baseline).

## Day 5 – reliability (`src/reliability.py`, `src/weights_test.py`)
- Match-level bootstrap (whole rating recomputed per resample) → 90% rank interval and P(top 10).
- Split-half (2022–24 vs 2025–26, and random halves of matches): pressure WPA and 'holds level' are the
  least repeatable parts; runs above expected / runs saved / dots under pressure repeat moderately;
  ordinary-ball skill repeats best.
- Sensitivity: other weights, eligibility 90/150, role prior, pressure threshold 1.25/2.0, no-Elo model.
- Five weightings compared (`weights_test.txt`). For bowlers, dropping 'holds level' and leaning on runs
  saved and dots roughly doubles repeatability. For batters no weighting makes the rating repeat.

## Final weighting – C (chosen 2026-11-16)
- Batting: runs above expected 40%, survival 30%, WPA 30%. Bowling: runs saved 35%, dots 25%, wickets
  above expected 15%, WPA 25%. 'Holds own level' is reported per player, not weighted.
- Bowling rating split-half: +0.24 (2022–24 vs 2025–26), +0.39 (random halves), up from +0.07 / +0.08.
- Batting rating does not repeat (about −0.2 across periods): state this as a limitation.
- Final top 10s: `rating_batters.csv`, `rating_bowlers.csv`; rank intervals and scenario counts in
  `reliability_*.csv` and `fig_rank_intervals.png`.

## Day 7 – Rajasthan Royals focus (task: players RR should back at the IPL 2027 auction)
- Bug fixed: format code "T20" in competitions.csv also covers 30 A-team matches (Emerging Teams Asia
  Cup 2024, Asia Cup Rising Stars 2025). IPL is now identified by competition name; A-team matches join
  the T20I group. Whole pipeline re-run (367 IPL, 470 T20I, 30 A-team matches with ball-by-ball).
- Auction eligibility: players who appeared for a Pakistan side in the data are excluded (not available
  to IPL teams). 77 batters and 84 bowlers remain eligible.
- `src/squad_fit.py`: RR's pressure profile by phase vs other IPL sides (`rr_pressure_profile.csv`) and
  each pick's best pressure phase, latest IPL side and country (`rr_player_fit.csv`). Country is the
  national side in the data; 3 picks with no 2022–26 internationals use labelled public info.
- Pre-fix rankings kept in `outputs/before_fmt_fix/` for comparison.

## Day 6 – report (`src/report.py`)
- `report/SupeRR_S2_Round1_Methodology.pdf` (11 pages) built entirely from `outputs/`.
- `report/submission_order.txt` – the exact order to pick in the Unstop dropdowns.
- Before submitting: replace "[Your name / team]", reread in your own voice, and check every dropdown
  name against `submission_order.txt`.
