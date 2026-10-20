"""
Disengagement-risk model, revision 3 (15 Oct 2026): risk per mile driven, predicted at 500 m grid cells and
named road stretches, shown by county, city and tract. The model lives in avgap_disengage.py and is explained,
with every number, in notebook 17 (17_Disengagement_Risk_per_Mile.ipynb). This script only runs it.

What changed from revision 2 (disengagement_risk_rev2.py):
  * Road type per mile: revision 2 split each company's miles like California's road length, which put
    freeway-type roads at ~8x surface streets. Splitting them like California traffic (FHWA VM-2) gives ~0.7x,
    and an estimate from the miles companies actually filed (no split assumed; simulation-calibrated) puts
    the ratio at or below 1. Freeways are not the tricky terrain per mile driven.
  * Specialised models: two local LLMs read every report for the road situation (traffic control,
    intersection turns, pedestrians, cyclists, merges, ...). Each situation with a mappable feature is a model
    whose rate per mile scales with that OpenStreetMap feature (signals and stop signs, junctions, crossings,
    bike lanes, multi-lane streets, ramps); they add up to one risk per mile.
  * Uncertainty and sample size: a Monte Carlo 90% range (company bootstrap of the situation shares, the
    calibrated freeway ratio, and how many unnamed reports follow the named mix); reports behind each model.
  * Checked against located AV collisions (DMV OL 316, CPUC), which played no part in building it.

Run from Project-DSC/Modeling (it reads av_data/ and the avgap_* modules):
    python disengagement_risk.py           # cached fit (av_data/disengage/fit_v3.pkl) -> outputs/
    python disengagement_risk.py --refit   # refit everything, ~30 minutes
Writes outputs/tract_scores_disengagement.csv (revision 1/2 schema plus new columns), disengagement_areas.csv
(county and city), disengagement_cells.csv (500 m cells in SF, LA, Santa Clara), disengagement_stretches.csv
and disengagement_model.json (headline numbers for the dashboard).
"""
import sys

import avgap_disengage as D

if __name__ == "__main__":
    fit = D.fit_all(refresh="--refit" in sys.argv)
    res = D.score_all(fit)
    c = res["county"].sort_values("risk_multiplier", ascending=False)
    print(f"freeway/surface rate per mile {fit['rho']:.2f} (calibrated 90% set {fit['confidence_set']['lo']:.2f}-"
          f"{fit['confidence_set']['hi']:.2f}); scale {fit['reference_rate']:.2f} reports per 1,000 miles = 1x")
    print("highest counties:", ", ".join(f"{n} {m:.2f}x" for n, m in c.risk_multiplier.head(5).items()))
