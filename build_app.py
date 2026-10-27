"""
Builds the MVP page: injects the notebook outputs into app/template.html.

    python build_app.py              ->  app/av-validation-gap-mvp.html  (San Francisco + Los Angeles)
    python build_app.py --statewide  ->  every California tract

Inputs (written by notebooks 02, 03, 05, 06, 07, 08, 13, 17 and 19):
    outputs/tract_coverage_equity.csv
    outputs/tract_scores.csv            (crash-risk model trained on CPUC 2025-26 collisions)
    outputs/tract_scores_dmv_model.csv  (second crash-risk model, DMV collision reports 2019-24)
    outputs/tract_crash_history.csv
    outputs/company_risk_forecast.csv
    outputs/coverage_series.csv
    outputs/tract_explanations.csv      (nb 13: model features, observed vs expected collisions,
                                         closest listed tract)
    outputs/tract_analogues.csv         (nb 13: the 25 analogues behind each Watch / Flag status)
    outputs/place_model_explain.json    (nb 13: coefficients, bootstrap refits, typical listed tract)
    outputs/tract_scores_disengagement.csv  (nb 17: disengagement risk per mile, 90% range, reports behind it,
                                             one term per specialised layer)
    outputs/disengagement_areas.csv     (nb 17: the same per county)
    outputs/disengagement_model.json    (nb 17: the headline numbers the page quotes, police-crash check per layer)
    outputs/synthetic_futures.json      (nb 19, optional: what a newly served county's own record would show)
    av_data/tl_2023_06_tract.zip        (TIGER 2023 tracts: map shapes and the state outline)

Map shapes (M. Nguyen's site, deployed 2026-10-03): each tract is drawn as its outline when zoomed
in (largest part, simplified to ~40 m) and as a dot when zoomed out; the state outline is the
union of every tract, simplified to ~1 km.

Explanations (notebook 13): the page recomputes each tract's factors from its 13 model features
and the coefficients, so the "why" panel, comparisons and model-input layers need no per-tract
precomputation. The build checks that the factors multiply back to the risk index.

Framing (team decision 2026-10-14, after a staff engineer's feedback relayed by Nithya): the page leads with
disengagement risk per mile (notebook 17), a relative index with a 90% range shown with the reports behind it;
the crash-risk map is the second, independent signal and the basis of the validation statuses.

Product scope (team decision 2026-09-29, following the professor's feedback and M. Nguyen's site):
tract-level crash risk is shown for San Francisco and Los Angeles, the two counties with enough
driving history to support it. Percentiles are ranked within the scope, so "riskier than 62% of
tracts" refers to tracts actually on the map. Statewide context (the authorisation gap) is passed
in as numbers.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SCOPE_COUNTIES = ["San Francisco", "Los Angeles"]


def r(v, n):
    return None if pd.isna(v) else round(float(v), n)


def ip(v):
    return None if pd.isna(v) else int(v)


def percentile(ref_values, values):
    """1-100 rank of `values` within the sorted reference distribution (same rule as nb 06)."""
    ref = np.sort(pd.Series(ref_values).dropna().to_numpy())
    out = pd.Series(np.nan, index=values.index)
    ok = values.notna()
    out[ok] = (np.searchsorted(ref, values[ok].to_numpy(), side="left") / len(ref) * 99 + 1).round()
    return out


def shapes(geoids):
    """Tract outlines for `geoids` (None where a tract has no shape) and the state outline."""
    from shapely.ops import unary_union
    from avgap_osm import TRACT_SHP, read_polygons
    ids, geoms, _ = read_polygons(TRACT_SHP, "GEOID")

    def parts(g):
        return list(g.geoms) if g.geom_type == "MultiPolygon" else [g]

    def ring(p, nd):
        return [[round(x, nd), round(y, nd)] for x, y in p.exterior.coords]

    by_id = dict(zip(ids, geoms))
    polys = [None if gid not in by_id else
             ring(max(parts(by_id[gid].simplify(0.0004, preserve_topology=True)), key=lambda p: p.area), 5)
             for gid in geoids]
    state = sorted(parts(unary_union([g.buffer(0) for g in geoms])), key=lambda p: -p.area)
    outline = [ring(q, 4) for p in state for q in parts(p.simplify(0.01, preserve_topology=True))
               if not q.is_empty]
    return polys, outline


NEAR_FEATURES = ["log_density", "pct_multiunit", "pct_transit", "pct_walk_bike", "pct_no_vehicle", "log_income",
                 "log_road_density", "share_arterial", "log_crossing_density", "log_signal_density"]


def closest_listed(ex, served, pos, k=3):
    """
    For every tract not in a filing: its k closest listed tracts, [index on the map, distance], in the
    standardised census + road space that the look-alikes use (geography is not a feature). Distances
    are in units of the median nearest-neighbour distance among listed tracts, so 1 = a typical gap.
    These are the places to test in to learn about the tract.
    """
    from sklearn.neighbors import NearestNeighbors
    X = ex[NEAR_FEATURES]
    ok = X.notna().all(axis=1).to_numpy() & served.notna().to_numpy()
    Z = ((X - X[ok].mean()) / X[ok].std()).to_numpy()
    listed = np.where(ok & (served.fillna(False).astype(bool).to_numpy()))[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z[listed])
    scale = np.median(nn.kneighbors(Z[listed])[0][:, 1])
    out = [None] * len(ex)
    todo = np.where(ok & ~np.isin(np.arange(len(ex)), listed))[0]
    dist, idx = nn.kneighbors(Z[todo], n_neighbors=k)
    for row, d, ii in zip(todo, dist, idx):
        out[row] = [[int(listed[j]), round(float(x / scale), 2)] for x, j in zip(d, ii)]
    return out


def explanations(t, geoids):
    """
    Notebook 13's exports, aligned with the map's tracts:
      feat     the 13 model features per tract (model scale), from which the page rebuilds each factor
      obs      [reported collisions 2025Q1-26Q2, expected, shrunk ratio, 90% lo, hi] for listed tracts
      anal     for Watch / Flag tracts: [25 analogues, closest listed tract, 3 largest feature gaps].
               An analogue on the map is its index; one off the map is -(1 + 2 * county + listed),
               county indexing offCounties and listed = ever in a filing. The closest listed tract
               is its index, or [GEOID, county] when it is off the map.
    """
    meta = json.load(open(os.path.join(OUT, "place_model_explain.json"), encoding="utf-8"))
    F = meta["features"]
    ex = pd.read_csv(os.path.join(OUT, "tract_explanations.csv"),
                     dtype={"GEOID": str, "nearest_listed": str}).set_index("GEOID")
    an = pd.read_csv(os.path.join(OUT, "tract_analogues.csv"), dtype={"GEOID": str}).set_index("GEOID")
    ex = ex.reindex(geoids)

    feat = [None if r.isna().any() else [round(float(v), 4) for v in r] for _, r in ex[F].iterrows()]
    # the factors must multiply back to the published risk index
    c, b, ref = meta["constant"], pd.Series(meta["coef"]), pd.Series(meta["reference"])
    ri = t.set_index("GEOID").risk_index.reindex(geoids)
    rebuilt = np.exp(c + ((ex[F] - ref[F]) * b[F]).sum(axis=1, min_count=len(F)))
    err = (rebuilt / ri - 1).abs().max()
    assert err < 1e-2, f"factors do not rebuild the risk index (max relative error {err:.3g})"

    obs_cols = ["cpuc_collisions_2025_26", "expected_2025_26", "excess", "excess_lo", "excess_hi"]
    obs = [None if pd.isna(r.cpuc_collisions_2025_26) else
           [int(r.cpuc_collisions_2025_26)] + [round(float(r[k]), 2) for k in obs_cols[1:]]
           for _, r in ex[obs_cols].iterrows()]

    pos = {g: i for i, g in enumerate(geoids)}
    allt = pd.read_csv(os.path.join(OUT, "tract_coverage_equity.csv"), dtype={"GEOID": str}).set_index("GEOID")
    off_counties = sorted(allt.county.dropna().unique())

    def ref_of(g):
        if g in pos:
            return pos[g]
        return -(1 + 2 * off_counties.index(allt.county[g]) + int(allt.months_served[g] > 0))

    gap_names = sorted({v for k in ("gap1_feature", "gap2_feature", "gap3_feature") for v in ex[k].dropna()})
    gap_feats = [f for f in meta["labels"] if f in gap_names]
    anal = []
    for g, r in ex.iterrows():
        if g not in an.index or pd.isna(r.nearest_listed):
            anal.append(None)
            continue
        gaps = [[gap_feats.index(r[f"gap{k}_feature"]), round(float(r[f"gap{k}_z"]), 1)] for k in (1, 2, 3)]
        near = r.nearest_listed
        anal.append([[ref_of(a) for a in an.loc[g, "analogues"].split()],
                     pos[near] if near in pos else [near, allt.county[near]], gaps])
    meta["gapFeatures"] = gap_feats
    return meta, {"feat": feat, "obs": obs, "anal": anal, "offCounties": off_counties,
                  "near3": closest_listed(ex, t.set_index("GEOID").served_now.reindex(geoids), pos)}


def disengagement(geoids):
    """
    Notebook 17's layer, aligned with the map's tracts: [risk per mile (x an average California road
    mile), 90% lo, hi, largest term, reports per 1,000 miles at the 2024 median company, reports behind
    the tract's road mix, [one term per specialised layer], pct of this tract's road length that is
    freeway, the surface-road multiplier, the freeway multiplier], plus the numbers the page quotes and,
    per layer, its feature, the reports naming its situation and the police-crash check (nb 17 section 5).
    It is a separate model from the supervised-testing era and uses no crash data.
    """
    from avgap_disengage import FEATURE_LABEL, TIED, situation_of
    d = pd.read_csv(os.path.join(OUT, "tract_scores_disengagement.csv"), dtype={"GEOID": str}).set_index("GEOID")
    m = json.load(open(os.path.join(OUT, "disengagement_model.json"), encoding="utf-8"))
    d = d.reindex(geoids)
    keys = [c[5:] for c in d.columns if c.startswith("term:")]
    tv = d[["term:" + k for k in keys]].to_numpy()
    rows = [None if pd.isna(x.risk_multiplier) else
            [round(float(x.risk_multiplier), 3), round(float(x.risk_multiplier_lo), 3), round(float(x.risk_multiplier_hi), 3),
             x.top_feature if isinstance(x.top_feature, str) else None, round(float(x.per_1k_miles), 3),
             ip(x.n_sample), [r(v, 3) for v in tv[j]],
             r(x.pct_freeway * 100, 1), r(x.surface_multiplier, 3), r(x.freeway_multiplier, 3)]
            for j, x in enumerate(d.itertuples())]
    v = m["validation"]
    pol = v["police_by_layer"]
    layers = [{"key": k, "feature": FEATURE_LABEL[k], "road": TIED[k][0], "situation": situation_of(k),
               "reports": int(m["situation_reports"][TIED[k][0]].get(situation_of(k), 0)),
               "crashType": pol[k]["crash_type"], "crashes": int(pol[k]["crashes"]), "or": pol[k]["odds_ratio"],
               "lo": pol[k]["lo"], "hi": pol[k]["hi"], "raises": bool(pol[k]["raises_own"])} for k in keys]
    areas = pd.read_csv(os.path.join(OUT, "disengagement_areas.csv"))
    counties = {x.name: [round(float(x.risk_multiplier), 3), round(float(x.risk_multiplier_lo), 3),
                         round(float(x.risk_multiplier_hi), 3), x.top_feature if isinstance(x.top_feature, str) else None]
                for x in areas[areas.level == "county"].itertuples()}
    meta = {"counties": counties, "reports": m["reports"], "companies": m["companies"], "rhoCrashPerKm": v["spearman_per_km_rev3"],
            "rhoCrossingsPerKm": v["spearman_per_km_crossings"], "crashes": int(v["crashes"]),
            "policeUp": v["police_layers_own_up"], "policeLayers": v["police_layers"],
            "periodRho": v["period_map_spearman"], "freewayRatio": m["rho"], "freewaySet": m["rho_set"],
            "rhoRev2PerKm": v["spearman_per_km_rev2"], "cellTopCrashes": v["cell_top_tenth_crashes"],
            "cellTopKm": v["cell_top_tenth_km"], "cells": v["cells"], "policeCrashes": v["police_crashes"],
            "cpucBase": v["cpuc_base"], "cpucWith": v["cpuc_rev3_offset"], "locoMin": v["map_loco_min_spearman"],
            "referenceRate": m["reference_rate"], "freewayRoadLength": m["rho_road_length"],
            "freewayVehicleMiles": m["rho_vehicle_miles"], "layers": layers}
    return rows, meta, keys


def disengagement_cells(geoids, keys):
    """
    Notebook 17's 500 m grid cells (`outputs/disengagement_cells.csv`), kept to the same tracts as
    the map: [lon, lat, risk per mile, 90% lo, hi, reports per 1,000 miles at the 2024 median
    company, largest specialised layer]. Lets the page show the disengagement layer at its native
    grid resolution instead of averaged over a tract, with a switch to flip between the two.
    """
    from avgap_disengage import FEATURE_LABEL
    gset = set(geoids)
    c = pd.read_csv(os.path.join(OUT, "disengagement_cells.csv"), dtype={"GEOID": str})
    c = c[c.GEOID.isin(gset)].reset_index(drop=True)
    tv = c[["term:" + k for k in keys]].to_numpy()
    rows = []
    for j, x in enumerate(c.itertuples()):
        terms = tv[j]
        top = None
        if np.isfinite(terms).any():
            ix = int(np.nanargmax(terms))
            if terms[ix] > 0:
                top = FEATURE_LABEL[keys[ix]]
        rows.append([r(x.lon, 4), r(x.lat, 4), r(x.risk_multiplier, 3), r(x.risk_multiplier_lo, 3),
                     r(x.risk_multiplier_hi, 3), r(x.per_1k_miles, 3), top])
    return rows


def main(statewide=False):
    t = pd.read_csv(os.path.join(OUT, "tract_coverage_equity.csv"), dtype={"GEOID": str})
    sc = pd.read_csv(os.path.join(OUT, "tract_scores.csv"), dtype={"GEOID": str})
    t = t.merge(sc[["GEOID", "status", "risk_index", "analogue_coverage", "ood_distance",
                    "risk_reliable", "risk_lo", "risk_hi", "city"]], on="GEOID", how="left")
    t = t.merge(pd.read_csv(os.path.join(OUT, "tract_crash_history.csv"), dtype={"GEOID": str}),
                on="GEOID", how="left")
    t = t.merge(pd.read_csv(os.path.join(OUT, "tract_scores_dmv_model.csv"), dtype={"GEOID": str})
                [["GEOID", "risk316_q"]], on="GEOID", how="left")

    # statewide context, computed before any scoping
    auth = sorted(t.loc[t.authorised_2026, "county"].unique())
    by = t.groupby("county").agg(listed=("served_now", "sum"), pop=("population", "sum"))
    no_filing = [c for c in auth if by.loc[c, "listed"] == 0]
    context = {"authorised": len(auth), "noFiling": len(no_filing),
               "noFilingResidents": int(by.loc[no_filing, "pop"].sum()),
               "statewideTracts": len(t), "statewideListed": int(t.served_now.sum()),
               "scope": "California" if statewide else " + ".join(SCOPE_COUNTIES)}

    if not statewide:
        t = t[t.county.isin(SCOPE_COUNTIES)].copy()
    # percentiles within the tracts on the map; range bounds looked up in the same distribution
    ref = t.risk_index
    t["pct"], t["pct_lo"], t["pct_hi"] = (percentile(ref, t.risk_index), percentile(ref, t.risk_lo),
                                          percentile(ref, t.risk_hi))
    t["pct316"] = percentile(t.risk316_q, t.risk316_q)

    status = {"Covered": 0, "Watch": 1, "Flag": 2, "Not authorised": 3}
    counties = sorted(t.county.dropna().unique())
    cidx = {c: i for i, c in enumerate(counties)}
    tracts = [[r(x.lon, 4), r(x.lat, 4), cidx.get(x.county, -1), int(x.population or 0),
               r(x.median_income, -2), r(x.pct_hispanic, 3), r(x.pct_nh_black, 3),
               r(x.pct_no_vehicle, 3), int(bool(x.served_now)), x.GEOID,
               status.get(x.status, 3), r(x.risk_index, 2), r(x.analogue_coverage, 2),
               r(x.ood_distance, 2), int(x.risk_reliable == True),  # noqa: E712 (numpy bool)
               r(x.risk_lo, 2), r(x.risk_hi, 2), x.city if isinstance(x.city, str) else None,
               r(x.ol316_autonomous_2019_24, 1), r(x.ol316_all_2019_24, 1),
               ip(x.pct), ip(x.pct_lo), ip(x.pct_hi), ip(x.pct316),
               int(x.months_served), x.first_served if isinstance(x.first_served, str) else None]
              for x in t.itertuples()]

    explain, extra = explanations(t, [x[9] for x in tracts])
    geoids = [x[9] for x in tracts]
    extra["dis"], extra["disMeta"], dis_keys = disengagement(geoids)
    extra["disCells"] = disengagement_cells(geoids, dis_keys)
    fut = os.path.join(OUT, "synthetic_futures.json")
    extra["futures"] = json.load(open(fut, encoding="utf-8")) if os.path.exists(fut) else None

    risk = pd.read_csv(os.path.join(OUT, "company_risk_forecast.csv"))
    risk = risk.rename(columns={risk.columns[0]: "company"})
    risk = [{k: (None if pd.isna(v) else (int(v) if isinstance(v, (int, np.integer)) else v))
             for k, v in row.items()} for row in risk.to_dict("records")]

    cs = pd.read_csv(os.path.join(OUT, "coverage_series.csv"))
    series = [{"m": row.month, "listed": int(row["Any operator"]),
               "trips": None if pd.isna(row.share_listed_with_trips)
               else round(row["Any operator"] * row.share_listed_with_trips)}
              for _, row in cs.iterrows()]

    polys, outline = shapes([x[9] for x in tracts])
    filed = sorted(by.index[by.listed > 0])                     # counties with any tract in a filing
    data = {"counties": counties, "authorised": [c for c in auth if c in counties], "tracts": tracts,
            "filedCounties": filed,
            "tractPolys": polys, "risk": risk, "series": series, "asOf": series[-1]["m"],
            "context": context, "outline": outline, "explain": explain, **extra}
    blob = json.dumps(data, separators=(",", ":"))
    tpl = open(os.path.join(HERE, "app", "template.html"), encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/null", blob)
    dst = os.path.join(HERE, "app", "av-validation-gap-mvp.html")
    open(dst, "w", encoding="utf-8").write(html)
    print(f"wrote {dst} ({os.path.getsize(dst)/1e6:.2f} MB, {len(tracts):,} tracts, scope: {context['scope']})")


if __name__ == "__main__":
    main(statewide="--statewide" in sys.argv)
