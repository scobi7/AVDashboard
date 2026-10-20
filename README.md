# AV Validation Gap — Dashboard

**Live:** https://av-validation-gap-site-disengagemen.vercel.app
**All dashboards:** https://av-validation-gap-landing.vercel.app (collision-risk model, full view, pooled-data experiment, and this one)

UC Berkeley MIDS 210 capstone. This is the **disengagement-risk** dashboard. Framed as *predicting
disengagement risk*, not predicting disengagements. It answers: per mile driven, how likely is an
autonomous vehicle to need a human takeover here, relative to an average California road mile? It is
predicted at 500 m grid cells and named road stretches, and shown by county, city and tract, with a 90%
range and the number of reports behind each part of the estimate.

It is industry-wide: 52,626 CA DMV disengagement reports from 39 companies, 2020–24. Waymo is under 2% of
them.

## Revision 3 (15 Oct 2026): what changed and why

The full method, every number, and the checks are in `notebooks/17_Disengagement_Risk_per_Mile.ipynb`
(source: Project-DSC `Modeling/nb_17.py`, module `avgap_disengage.py`).

1. **Freeways are not the tricky terrain per mile driven.** Revisions 1–2 split each company's miles
   across road types like California's *road length*. That put freeway-type roads at 5–8× surface streets.
   - Splitting miles like California *traffic* (FHWA VM-2: freeway-type roads carry 48% of vehicle-miles)
     gives 0.71×.
   - A new estimator from the miles companies actually filed assumes no split. It uses months when a
     company's reports shift toward freeways, against that month's miles, and is calibrated by
     simulation because it is biased. It puts the ratio at 0.10–1.40 (90% set), point 0.58.
   - Ratios of 2+ are ruled out.
2. **Specialised models → one aggregated risk** (the staff engineer's suggestion). Two local LLMs (Qwen3.5-9B,
   Gemma 4 12B; checked against Qwen with reasoning and Cloudflare's Clef-flash, κ 0.73–0.74) read every
   report for the road situation. Each situation with a mappable feature is its own model, and its rate per
   mile scales with that feature's density in OpenStreetMap:
   - signals and stop signs; junctions; crosswalks; bike lanes; multi-lane streets;
   - shops and services (curbside activity → double-parked and stopped vehicles); road curvature;
   - freeway ramps; and freeway traffic from Caltrans AADT (Malachi's suggestion, used as a feature, not as
     exposure).

   Together they give the risk per mile.
3. **Uncertainty and sample size, as asked.** The 90% range is a Monte Carlo over:
   - the company bootstrap of the situation shares;
   - the calibrated freeway ratio;
   - how many unnamed reports follow the named mix;
   - the dose-response exponent.

   Each specialised model shows its sample size: the reports that name its situation.
4. **Checked against where AVs actually crashed** (nothing from collisions is used to build the map). On
   DMV-reported autonomous-mode crashes, 2019–24, in the tracts AVs drove, the map ranks crashes per km of
   road within county at Spearman **0.45**:
   - revision 2: 0.04; revision 1: −0.26;
   - crossing density alone: 0.47;
   - the collision-trained crash model: 0.43.

   At the 500 m cells it predicts, the riskiest tenth holds 56% of the crashes in them on 13% of the road km.
5. **Each specialised model checked against police-reported crashes of its own kind.** The AV record has
   6–19 located crashes per situation, too few to test the layers one by one. CHP's crash records (CCRS
   2022–24, LA, SF, San Mateo, Santa Clara) have 365,921 located crashes with a collision type.
   - The test is the share of a 500 m cell's crashes of the matching type (crosswalks → pedestrian, curves →
     left the lane, Caltrans traffic → freeway rear-ends, ...). It uses all layers in one model and a 5 km
     block bootstrap.
   - 7 of the 10 layers raise their own crash type. Junctions, curbside activity and freeway ramps don't.
   - The tract card marks each bar ✓ or ✗ accordingly.
6. **The 8/15 leave-one-company-out problem.** The road-type multiplier no longer comes from pooling
   companies that drive different roads. It comes from within-company variation, and the map holds when any
   one of the eight largest filers is dropped (rank correlation 0.93 or higher).
7. **Stable over time.** Rebuilt from 2019–22 reports alone (Apple 47%) or 2023–24 alone (May Mobility
   55%), the tract maps agree at rank correlation 0.89.

**Read it as:** a relative risk per mile (1 = an average California road mile), plus reports per 1,000
miles at the 2024 median company's rate (0.50). Company rates differ by four orders of magnitude, so use the
ranking and the ratio. Do not read it as a forecast for any one company, or as a rating of whether AVs are
safe.

## What's in this repo

- **`index.html`**: the dashboard (the `--disengagement --simple` build), self-contained.
- **`build_app.py`**: builds it. It reads the model outputs from the Project-DSC `Modeling/` folder:
  ```
  $env:AVGAP_MODELING = "C:\path\to\Project-DSC\Modeling"     # PowerShell (bash: export AVGAP_MODELING=...)
  python build_app.py --disengagement --simple   # -> app/av-validation-gap-mvp-disengagement-simple.html (copy to index.html)
  python build_app.py --disengagement            # the same with the header, ranked list and method notes
  ```
- **`source/template.html`**: the shared template. The disengagement build adds:
  - the per-mile tooltip;
  - a tract card ("what makes it tricky" with each model's sample size and its police-crash check, and the
    trickiest named stretches);
  - a county card with a city table;
  - a "500 m grid" map mode;
  - a counties/cities table.
- **`avgap_disengage.py`, `disengagement_risk.py`**: the revision 3 model and its runner (run from
  `Modeling/`, which has the data and the other `avgap_*` modules). `disengagement_risk_rev2.py` and
  `disengagement_validation.py` reproduce revision 2 and its leave-one-company-out check.
- **`outputs/`**: `tract_scores_disengagement.csv` (revision 1's columns plus the new ones),
  `disengagement_areas.csv` (county and city), `disengagement_cells.csv` (500 m cells in SF, LA,
  Santa Clara), `disengagement_stretches.csv`, `disengagement_model.json` (the numbers the page quotes),
  `tract_drivers.csv`.
- **`notebooks/17_Disengagement_Risk_per_Mile.ipynb`** and **`figures/DIS_*.png`**: the analysis.
- **`REPORT.md`, `VALIDATION_REPORT.md`, `PROFESSOR_RESPONSE.md`**: the collision-risk model's results and
  review (still the other dashboards).

The **`dashboards`** branch has all four variants as separate files plus the landing page.

## The two models, in one paragraph each

**Collision risk** (the original model): negative binomial regression on expected collisions per
census tract per quarter, from land area, housing density, service maturity, and marked-crosswalk
density. Validated four ways — spatial hold-out (41%), temporal hold-out (39%), an out-of-time test
against a different operator's crash data (Spearman 0.39), and an NHTSA cross-check (0.74). Waymo-trained.

**Disengagement risk** (this dashboard, revision 3): specialised models for each road situation the
disengagement reports name, each tied to an OpenStreetMap or Caltrans feature, aggregated into one relative
risk per mile. The freeway/surface split comes from the miles companies filed, calibrated by simulation.
Cross-company, with a Monte Carlo 90% range and a sample size per model. It ranks where AVs crashed about as
well as crossing density alone, without using any crash data.
