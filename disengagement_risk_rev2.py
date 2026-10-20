"""
Revision 2 (13 Oct 2026), kept to reproduce its numbers. Revision 3 (avgap_disengage.py, notebook 17) replaced it.

Disengagement-risk model: a second, separate dashboard from the Waymo collision-risk map.

Per the professor's note: usable if framed as *risk*, not event prediction, reported per mile
with confidence/uncertainty and sample size. Cross-company (39 companies, Waymo is only 1.8%
of the data) - this is an industry-wide testing pattern, not a Waymo-specific estimate, and the
dashboard says so.

Design (revision 2; the leave-one-company-out check is disengagement_validation.py):

  1. Fleet base rate = total disengagements / total miles, across all companies/months.
     Real exposure, no proxy.
  2. Road types: two classes, because the DMV form has no arterial box (Street covers every
     surface road) and companies use "Highway" for freeway-type driving (Ghost Autonomy, a freeway
     system, files 98% of its reports there):
       Freeway, Interstate, Highway           -> freeway-type: OSM motorway/trunk (km_freeway)
       Street, Rural Road, Parking Facility   -> surface: OSM primary and below
     Revision 1 mapped Highway onto OSM arterials, a third class the form does not have.
  3. Road-type multiplier: a negative-binomial GLM (the crash model's family, notebooks 06/14) with
     a fixed effect per company, n[company, type] ~ company + type, offset log(share of CA road
     length), fitted on the road types each company reports on (5+ reports). Each multiplier is a
     within-company comparison: companies that report on one road type, or report far more per mile
     (May Mobility files 39% of all reports), no longer set it. Scaled so that CA's road-length-
     weighted average is 1. Revision 1 pooled every report, so it mostly learned who drives where.
  4. Tract score = fleet base rate x the tract's road-length mix weighted by the multipliers.
     Exposure is still road length, not miles driven on each road type (no public source has the
     latter by company), so the multiplier says where reports concentrate per km of road.
  5. Uncertainty: cluster bootstrap over companies (200 resamples), refitting the GLM each time;
     each tract's 90% range comes from its own score in every resample. Sample size (reports per
     road type, weighted by the tract's mix) is reported alongside, per the professor's ask.

Run: python disengagement_risk_rev2.py
Writes: outputs/tract_scores_disengagement_rev2.csv (revision 3 replaced it: avgap_disengage.py, notebook 17)
"""
import numpy as np
import pandas as pd

import avgap_data as A
from avgap_coverage import tract_table
from avgap_osm import tract_roads
from disengagement_validation import SCHEMES, counts, fe_glm, road_km

SCHEME = "two-class"
N_BOOT = 200


def percentile(x):
    """1-100, ties sharing their average rank"""
    from scipy.stats import rankdata
    x = np.round(np.asarray(x, float), 6)        # equal road mixes must tie exactly, not by float noise
    return ((rankdata(x, method="average") - 1) / len(x) * 99 + 1).round()


def bootstrap(N, s, n_boot=N_BOOT, seed=0):
    """log multipliers refitted on companies resampled with replacement (each draw a distinct cluster)"""
    rng = np.random.default_rng(seed)
    comps = N.index.to_numpy()
    out = []
    for _ in range(n_boot):
        pick = rng.choice(comps, size=len(comps), replace=True)
        Nb = N.loc[pick].copy()
        Nb.index = [f"{c}#{i}" for i, c in enumerate(pick)]
        out.append(fe_glm(Nb, s))
    return pd.DataFrame(out)


def main():
    pd.set_option("display.width", 200)
    print("1. load disengagements + mileage, map road_type -> two classes")
    dis = A.load_disengagements()
    mil_long, _ = A.load_mileage()
    N = counts(dis, SCHEME)
    print(f"   {int(N.values.sum()):,} reports kept, {(~dis.road_type.isin(SCHEMES[SCHEME]['dmv'])).sum()} "
          f"'Other (non-conforming)' dropped; {len(N)} companies")
    print(dis[dis.road_type.isin(SCHEMES[SCHEME]["dmv"])].groupby(
        [dis.road_type.map(SCHEMES[SCHEME]["dmv"]), "road_type"]).size())

    print("\n2. fleet base rate (real exposure, no proxy)")
    total_miles = mil_long.miles.sum()
    base_rate = N.values.sum() / total_miles
    print(f"   {int(N.values.sum()):,} disengagements / {total_miles:,.0f} miles = {base_rate * 1e6:.2f} per million miles")

    print("\n3. statewide road length by class (OSM, every CA tract)")
    roads = tract_roads()
    km = road_km(roads, SCHEME)
    s = km.sum() / km.values.sum()
    print(pd.DataFrame({"km": km.sum().round(0), "share": s.round(3)}))

    print("\n4. NB GLM with company fixed effects -> multiplier, cluster bootstrap over companies")
    lm, model, used = fe_glm(N, s, return_model=True)
    mult = np.exp(lm)
    boot = bootstrap(N, s)
    mb = np.exp(boot)
    print(f"   fitted on {used.company.nunique()} companies that report on both road types "
          f"({int(used.n.sum()):,} reports); dispersion alpha {model.model.family.alpha:.2f}")
    print(pd.DataFrame({"reports": N.sum(axis=0), "multiplier": mult.round(2),
                        "lo (5%)": mb.quantile(0.05).round(2), "hi (95%)": mb.quantile(0.95).round(2)}))

    print("\n5. score every tract: road-mix-weighted base rate")
    t, _ = tract_table()
    t = t.merge(roads, on="GEOID", how="left")
    k3 = pd.DataFrame({"pct_freeway": t.km_freeway, "pct_arterial": t.km_arterial,
                       "pct_local": t.km_collector + t.km_local}).fillna(0)
    total = k3.sum(axis=1)
    ok = total > 0
    for c in k3:
        t.loc[ok, c] = k3.loc[ok, c] / total[ok]
    mix = pd.DataFrame({"freeway": t.loc[ok, "pct_freeway"],
                        "surface": t.loc[ok, "pct_arterial"] + t.loc[ok, "pct_local"]})[list(N.columns)]
    t.loc[ok, "risk_multiplier"] = mix.to_numpy() @ mult.to_numpy()
    draws = mix.to_numpy() @ mb.to_numpy().T                   # tracts x resamples
    t.loc[ok, "risk_multiplier_lo"] = np.percentile(draws, 5, axis=1)
    t.loc[ok, "risk_multiplier_hi"] = np.percentile(draws, 95, axis=1)
    t.loc[ok, "disengagements_per_million_miles"] = base_rate * 1e6 * t.loc[ok, "risk_multiplier"]
    t.loc[ok, "dpmm_lo"] = base_rate * 1e6 * t.loc[ok, "risk_multiplier_lo"]
    t.loc[ok, "dpmm_hi"] = base_rate * 1e6 * t.loc[ok, "risk_multiplier_hi"]
    t.loc[ok, "n_sample"] = (mix.to_numpy() @ N.sum(axis=0).to_numpy()).round(0)
    # tied tracts share their average rank: over half have no freeway-type road and tie at the lowest
    # score, which a lowest-rank rule would label "1st percentile"
    t.loc[ok, "risk_percentile"] = percentile(t.loc[ok, "risk_multiplier"].to_numpy())

    out = t.loc[ok, ["GEOID", "county", "population", "risk_multiplier", "risk_multiplier_lo",
                     "risk_multiplier_hi", "disengagements_per_million_miles", "dpmm_lo", "dpmm_hi",
                     "n_sample", "risk_percentile", "pct_freeway", "pct_arterial", "pct_local"]]
    out.to_csv("outputs/tract_scores_disengagement_rev2.csv", index=False)
    print(f"\n   wrote outputs/tract_scores_disengagement_rev2.csv ({len(out):,} tracts)")
    print(f"\nSUMMARY: fleet base rate {base_rate * 1e6:.2f}/million miles, {len(N)} companies "
          f"(Waymo {100 * N.loc['Waymo'].sum() / N.values.sum():.1f}% of reports); multipliers "
          + ", ".join(f"{c} {mult[c]:.2f} ({mb[c].quantile(0.05):.2f}-{mb[c].quantile(0.95):.2f})" for c in N.columns))


if __name__ == "__main__":
    main()
