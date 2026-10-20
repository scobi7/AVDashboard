# Disengagement risk model — handoff

## Reply from Pratheek (15 Oct 2026): revision 3, branch `disengagement-rev3`

Everything below the line is Malachi's original handoff, kept as is. What happened to each item:

- **Generalisation gap (the 8/15 LOCO).** Done two ways.
  - Revision 2 (13 Oct) moved to a negative-binomial GLM with company fixed effects.
  - Revision 3 goes further: the road-type number now comes from **within-company variation in the miles
    each company filed**, not from pooling companies that drive different roads. It is calibrated by
    simulation, because the plain estimator turned out to be biased.
  - Result: per mile driven, freeway-type roads are *not* riskier than surface streets (90% set 0.10–1.40,
    point 0.58). The 1.8× / 5.4× freeway multipliers were per km of road, an artifact of using road length
    as exposure.
  - Leave-one-company-out on the ratio: 0.61–0.79, or 1.10 without AImotive. On the final map: rank
    correlation 0.93 or higher with any one of the 8 largest filers dropped.
- **Caltrans AADT.** Used, but as a feature, not as exposure (AV test miles are not distributed like human
  traffic). Freeway "other vehicle" takeovers (34–52% of freeway reports) scale with AADT; 14,072 Caltrans
  count points cover 72% of freeway km. It stays state-highway only, as you said.
- **Nithya's reframing** is implemented:
  - "predicting disengagement risk", per mile, with a 90% range and a sample size;
  - predicted at 500 m grid cells and named road stretches, shown by county, city and tract;
  - several specialised models (one per road situation the reports name: signals, junctions, crosswalks,
    bikes, merges, curbside activity, curves, ramps, freeway traffic) feeding one aggregated risk.
- **Is it any good?** It is checked against located AV collisions, which played no part in building it.
  DMV-reported autonomous-mode crashes 2019–24, per km of road within county: Spearman 0.45. For comparison:
  - revision 2: 0.04;
  - revision 1: −0.26;
  - crossing density alone: 0.47;
  - the collision-trained crash model: 0.43.

  Caveat: on AV crashes the specialised layers mostly mark busy streets in general. Only the junction layer
  tracks its own kind of crash, and there are only 6–19 located crashes per situation.
- **Each layer against police-reported crashes (added later on 15 Oct).** CCRS 2022–24 has 365,921 located
  crashes in LA, SF, San Mateo and Santa Clara.
  - On the share of a 500 m cell's crashes of the matching type, 7 of 10 layers raise their own kind of crash.
  - Junctions, curbside activity and freeway ramps don't.
  - So most layers do mark where their situation happens. Whether AVs struggle there in particular, the AV
    record is too thin to say.
  - The map is also stable over time: 2019–22 vs 2023–24 reports, rank correlation 0.89.
- **Rebuild:** see README ("What's in this repo"). `build_app.py` now runs from this repo with
  `AVGAP_MODELING` pointing at Project-DSC `Modeling/`. Your template changes are kept; the new pieces only
  appear in the `--disengagement` build.
- **Still open:** interviews with transportation staff; the κ labelling; the outreach emails. A human
  check of the road-situation labels (two annotators, ~100 reports) would be the cheapest next step on
  the model.

Full notebook: `notebooks/17_Disengagement_Risk_per_Mile.ipynb`.

---

For whoever (Claude or Pratheek) picks this up next. Read `disengagement_risk.py` first — this
doc is context, not a replacement for reading the code.

## What exists

A second, separate dashboard from the collision-risk map: `disengagement_risk.py` scores every CA
tract by road-type mix (freeway/arterial/local street) against how much CA DMV disengagement
reports concentrate on each road type statewide. Live at
https://av-validation-gap-site-disengagemen.vercel.app, source on the `main` branch of
`github.com/scobi7/AVDashboard`.

**Data**: 52,626 disengagement reports, 39 companies, CA DMV 2020-24 (`avgap_data.load_disengagements()`).
Waymo is 1.8% of it — this is an industry-wide pattern, not Waymo-specific.

**Method**: `disengagement_risk.py`, step 4 (`compute_multipliers`). Pools every company together,
computes `(share of disengagements on a road type) / (share of CA road length that's that type)`
per road type, with a cluster bootstrap over companies for the 90% range. Tract score = tract's own
road-length mix weighted against those multipliers.

**Results right now**: freeway 1.82× (0.23–4.83), arterial 0.39× (0.09–1.28), local street 1.03×
(0.69–1.20). Fleet base rate 2,519 disengagements / million miles.

## The problem

Leave-one-company-out validation (run inline, not saved as a script — rerun if needed, it's ~30
lines, see the chat log or ask): held out each of the 15 companies with ≥200 rows, rebuilt the
multipliers from the rest, checked whether that predicts the held-out company's road-type
distribution better than a naive road-length-only baseline.

**Result: beats the baseline on 8 of 15 companies — close to chance.**

Why: companies differ by *where they operate*, not by risk. From the real data
(`dis.groupby(["company","road_type"]).size()`):

| Company | Freeway | Highway | Interstate | Parking | Street |
|---|---|---|---|---|---|
| Apple | 3,781 | 248 | 1,458 | 0 | 4,767 |
| AImotive | 1,244 | 114 | 0 | 1 | 436 |
| Mercedes-Benz | 64 | 30 | 0 | 0 | 1,387 |
| Beep | 0 | 0 | 0 | 4,705 | 4,454 |
| Ghost Autonomy | 0 | 1,794 | 0 | 0 | 39 |
| May Mobility | 0 | 0 | 0 | 0 | 20,694 |

Beep, Ghost Autonomy, and May Mobility each operate on essentially one road type. Pooling them in
mostly teaches the model "who drives where," not "what's dangerous."

## The fix to try

Company-fixed-effects design, not pooled-then-held-out:

1. Only companies with real internal road-type spread can inform a within-company comparison —
   from the table above, that's **Apple, AImotive, Mercedes-Benz** (maybe a couple more; check the
   full `dis.company.value_counts()` for others with ≥2 road types at meaningful volume).
2. For each of those companies, compute *its own* rate per road type relative to *its own* average
   rate (not the fleet average). This isolates "does this company see more disengagements on
   freeways than its own baseline" from "which company is this."
3. Check whether that within-company deviation points the same direction across companies (e.g.,
   does every company with freeway exposure show elevated freeway risk relative to its own
   average?). If yes, consistently, that's a real road-type effect. If not, the signal may not
   exist at the road-type level with this data at all — worth knowing either way.
4. Re-run the same leave-one-company-out validation on the new design before shipping it. Don't
   skip this — it's the only way we caught the original design's problem.

## Also real, lower priority

- **Caltrans AADT** (traffic volume by road segment) would give real exposure data instead of the
  road-length proxy, but only for the state highway system — freeway/arterial, not local streets
  (72% of the disengagement data is on local streets). Confirmed via Caltrans' own documentation:
  Traffic Census Program is state-highway-only. Worth doing *after* the fixed-effects fix, not
  instead of it.
- **CA DMV 2025 reporting year**: requested directly, declined. Dead end, don't re-pursue through
  that channel.

## How to actually rebuild this

This repo now has `build_app.py`, `run_explain.py`, `source/template.html` (current, not the stale
copy from the first push), and the two CSVs these commands need
(`outputs/tract_scores_disengagement.csv`, `outputs/tract_drivers.csv`).

What it still needs from your own copy of the Modeling pipeline (not duplicated here, since it's
git-ignored and large): the `avgap_*.py` modules, the rest of `outputs/*.csv`, and `av_data/`. Drop
this repo's `build_app.py`, `run_explain.py`, and `source/template.html` into your Modeling folder
(overwriting yours), copy the two CSVs into your `outputs/`, then:

```
python disengagement_risk.py   # only if you've changed the model - rewrites tract_scores_disengagement.csv
python build_app.py --disengagement           # full variant
python build_app.py --disengagement --simple  # map-only variant, this is what's deployed
```

Output lands in `app/av-validation-gap-mvp-disengagement*.html`. Copy the `--simple` one over this
repo's `index.html` to update the live site.

## Wiring, once a better model exists

`build_app.py --disengagement` reads `outputs/tract_scores_disengagement.csv` (columns:
`GEOID, county, population, risk_multiplier, risk_multiplier_lo, risk_multiplier_hi,
disengagements_per_million_miles, dpmm_lo, dpmm_hi, n_sample, risk_percentile, pct_freeway,
pct_arterial, pct_local`). Keep that schema and the rest of the pipeline (tooltip, detail panel,
legend) needs no changes — it's already wired to read from this CSV, not hardcoded to the current
model. Rerun `disengagement_risk.py`, rebuild with `build_app.py --disengagement` and
`--disengagement --simple`, redeploy.

## Other open items (not about this model specifically)

- Interviews with transportation staff — still haven't confirmed the actual target user wants
  this product. Biggest unvalidated assumption in the whole project.
- Two-annotator κ labeling for the disengagement cause taxonomy.
- CPUC and DMV data request emails are drafted (`outreach/`) but not sent.
