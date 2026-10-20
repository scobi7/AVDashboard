"""
Disengagement risk per mile (notebook 17): the team's reframing of 2026-10-14.

Nithya's staff-engineer feedback: frame the product as *predicting disengagement risk*, not predicting
disengagements; make it useful for spotting tricky terrain; predict at road-segment or grid-cell level
and display at city or county level; report risk per mile with its uncertainty and sample size; and,
if there is time, let several specialised models feed one aggregated risk. Malachi's dashboard
(github.com/scobi7/AVDashboard) scores tracts by road-type mix; this module is its third revision.

What the DMV data can and cannot say. A disengagement report has a date, a company, a road-type box
(Street, Freeway, Highway, Interstate, Rural Road, Parking Facility) and a free-text description, but
no location. Each company also files its autonomous miles per vehicle per month. So:

  * per-mile rates are real at the company x month level (miles are filed);
  * the split of a company's miles across road types is not filed. Revision 2 (disengagement_risk_rev2.py)
    used each road type's share of California road length as that split; vehicle-miles (FHWA VM-2) is the
    other natural choice, and the two give opposite answers for freeways;
  * revision 3 estimates the freeway / surface ratio from filed miles instead: in months when a company
    files more of its reports on freeways, its miles were more freeway-heavy, so if freeway miles
    produce reports at a different rate, its total reports per mile move with that share
    (`ratio_from_miles`). On simulated panels with known answers that estimator is biased under the
    data's overdispersion (`simulate_ratio`), so the reported answer is a simulation-calibrated set
    (`confidence_set`) and a median-unbiased point (`median_unbiased`);
  * the description says, for some reports, what road situation the vehicle was handling (a traffic
    light, a pedestrian, a merge...). avgap_llm.scen_* tags every distinct description; each situation
    with a mappable feature becomes a specialised model whose rate per mile scales with that feature
    (TIED: OpenStreetMap signals and stops, junctions, crossings, bike lanes, multi-lane streets, POIs,
    curvature, ramps; Caltrans AADT on freeways), to the power BETA.

Layers, in the order notebook 17 runs them (fit_all caches 1-4, score_all does 6):

  1. panel(): company x month counts by road class (freeway-type / surface) and miles
  2. ratio_from_miles(), ratio_by_company(), random_effects(), ratio_loco(), simulate_ratio(),
     confidence_set(), median_unbiased(): the freeway / surface rate per mile from filed miles
  3. ratio_from_exposure(): the same ratio under road-length or vehicle-mile exposure (revision 2's
     company fixed-effects NB GLM), for comparison
  4. report_table(), situation_shares(), situation_bootstrap(): share of each road situation within
     each road class, as stated and imputed, cluster-bootstrapped over companies
  5. terrain(): OpenStreetMap (+ Caltrans AADT) features per 500 m grid cell, tract and named stretch
  6. multipliers(), monte_carlo(), score_all(): relative risk per mile (1 = an average California road
     mile) per cell, tract, city, county and stretch, with a Monte Carlo 90% range; outputs/ CSVs + JSON
  7. crash_check(), panel_check(), situation_check(), map_loco(): checks against located AV collisions
     (DMV OL 316 re-geocoded in notebook 07; Waymo's CPUC panel of notebook 06) and map stability
"""
import os
from array import array

import numpy as np
import pandas as pd

import avgap_data as A
from avgap_data import CACHE

DIS_DIR = os.path.join(CACHE, "disengage")
TERRAIN_CELLS = os.path.join(DIS_DIR, "terrain_cells.csv.gz")
TERRAIN_STRETCH = os.path.join(DIS_DIR, "terrain_stretches.csv.gz")
TERRAIN_TRACTS = os.path.join(DIS_DIR, "terrain_tracts.csv.gz")
FREEWAY_BOXES = {"Freeway", "Interstate", "Highway"}          # revision 2's two-class mapping
SURFACE_BOXES = {"Street", "Rural Road", "Parking Facility"}

# ------------------------------------------------------------------ 1. company x month panel
def panel(dis=None, mil=None, min_each=20):
    """
    Company x month: reports on freeway-type roads (n_fw), all reports with a road box (n), and
    autonomous miles filed. Kept: months with miles > 0, companies with at least `min_each` reports
    on each road class (the only ones whose month-to-month road mix can say anything), and
    company-years with at least one report. p = the month's share of reports on freeway-type roads
    (the company's overall share in months with no report).
    """
    dis = A.load_disengagements() if dis is None else dis
    mil = A.load_mileage()[0] if mil is None else mil
    d = dis[dis.road_type.isin(FREEWAY_BOXES | SURFACE_BOXES)].assign(fw=lambda x: x.road_type.isin(FREEWAY_BOXES))
    g = d.groupby(["company", "month"]).agg(n_fw=("fw", "sum"), n=("fw", "size")).reset_index()
    m = mil.groupby(["company", "month"]).miles.sum().reset_index()
    p = m.merge(g, on=["company", "month"], how="outer").fillna({"n_fw": 0, "n": 0, "miles": 0})
    p = p[p.miles > 0].copy()
    tot = p.groupby("company")[["n_fw", "n"]].sum()
    keep = tot[(tot.n_fw >= min_each) & (tot.n - tot.n_fw >= min_each)].index
    p = p[p.company.isin(keep)].copy()
    p["year"] = p.month.dt.year.astype(str)
    p = p[p.groupby(["company", "year"]).n.transform("sum") > 0].copy()
    share = tot.n_fw / tot.n
    p["p"] = np.where(p.n > 0, p.n_fw / p.n.where(p.n > 0, 1), p.company.map(share))
    p["t"] = (p.month.dt.to_timestamp() - pd.Timestamp("2020-01-01")).dt.days / 365.25
    return p.reset_index(drop=True)


# ------------------------------------------------------------------ 2. freeway / surface ratio from real miles
# If a company's surface miles produce reports at rate L and its freeway miles at rho * L, and a share
# e of a month's miles is on freeways, the month's reports per mile are L * (1 - e + rho e) and the
# share of its reports on freeways is p = rho e / (1 - e + rho e). Eliminating e:
#     E[reports] = miles * L / (1 - p (1 - 1/rho))
# so for a candidate rho the monthly totals are an NB GLM with offset log miles - log(1 - p(1 - 1/rho))
# and a level per company-year (absorbing each company's own rate and its year-to-year improvement).
# rho is chosen by profile likelihood. Only within-company, within-year variation in where the
# company drove identifies it. Noise in p biases rho toward 1 (simulate_ratio measures how much).
RATIO_FORMS = {"company-year": "n ~ C(company):C(year)", "company": "n ~ C(company)",
               "company trend": "n ~ C(company) + C(company):t"}


def _offset(p, rho):
    return np.log(p.miles.to_numpy()) - np.log(1 - p.p.to_numpy() * (1 - 1 / rho))


class _Failed:
    llf = -np.inf


def _glm(p, rho, form, alpha):
    """The GLM at a candidate ratio; a fit that fails numerically (extreme ratios on resampled panels)
    scores -inf, so the profile simply moves away from it."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    try:
        return smf.glm(form, p, offset=_offset(p, rho), family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    except (ValueError, np.linalg.LinAlgError):
        return _Failed()


def _best_alpha(p, rho, form):
    from scipy.optimize import minimize_scalar
    o = minimize_scalar(lambda la: -_glm(p, rho, form, np.exp(la)).llf, bounds=(np.log(0.01), np.log(10)),
                        method="bounded", options={"xatol": 0.05})
    return float(np.exp(o.x))


def ratio_from_miles(p, form="company-year", alpha=None, grid=np.geomspace(0.05, 20, 49)):
    """
    Profile likelihood of the freeway / surface ratio. Returns dict: rho, lo, hi (95% profile
    interval), alpha, profile (DataFrame), and the fitted GLM at rho (its company-year levels are each
    company's surface rate per mile).
    """
    import warnings
    from scipy.optimize import minimize_scalar
    f = RATIO_FORMS.get(form, form)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = alpha or _best_alpha(p, 1.0, f)
        ll = np.array([_glm(p, r, f, a).llf for r in grid])
        k = int(ll.argmax())
        lo_b, hi_b = np.log(grid[max(k - 1, 0)]), np.log(grid[min(k + 1, len(grid) - 1)])
        o = minimize_scalar(lambda lr: -_glm(p, np.exp(lr), f, a).llf, bounds=(lo_b, hi_b), method="bounded",
                            options={"xatol": 1e-3})
        rho = float(np.exp(o.x))
        if alpha is None:                       # one refinement of the dispersion at the estimate
            a = _best_alpha(p, rho, f)
            ll = np.array([_glm(p, r, f, a).llf for r in grid])
        fit = _glm(p, rho, f, a)
    ok = grid[ll >= fit.llf - 1.92]
    prof = pd.DataFrame({"rho": grid, "llf": ll - fit.llf})
    return {"rho": rho, "lo": float(ok.min()), "hi": float(ok.max()), "alpha": a, "profile": prof,
            "fit": fit, "companies": p.company.nunique(), "months": len(p), "reports": int(p.n.sum()),
            "edge": bool(ok.min() <= grid[0] or ok.max() >= grid[-1])}


def ratio_bootstrap(p, B=200, form="company-year", seed=0):
    """log rho refitted on companies resampled with replacement (each draw a distinct cluster);
    the dispersion is held at its full-data value."""
    import warnings
    from scipy.optimize import minimize_scalar
    f = RATIO_FORMS.get(form, form)
    a = ratio_from_miles(p, form)["alpha"]
    rng = np.random.default_rng(seed)
    comps = p.company.unique()
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(B):
            pick = rng.choice(comps, size=len(comps), replace=True)
            pb = pd.concat([p[p.company == c].assign(company=f"{c}#{j}") for j, c in enumerate(pick)],
                           ignore_index=True)
            o = minimize_scalar(lambda lr: -_glm(pb, np.exp(lr), f, a).llf, bounds=(np.log(0.05), np.log(20)),
                                method="bounded", options={"xatol": 1e-3})
            out.append(float(o.x))
    return np.array(out)


def ratio_loco(p, form="company-year"):
    """rho without each company in turn: does any single company carry the estimate?"""
    rows = []
    for c in p.groupby("company").n.sum().sort_values(ascending=False).index:
        r = ratio_from_miles(p[p.company != c], form)
        rows.append({"left out": c, "reports left out": int(p[p.company == c].n.sum()),
                     "rho": r["rho"], "lo": r["lo"], "hi": r["hi"]})
    return pd.DataFrame(rows)


def ratio_by_company(p):
    """Each company on its own (a level per year): the spread a pooled ratio averages over."""
    rows = []
    for c, x in p.groupby("company"):
        r = ratio_from_miles(x, "n ~ C(year)", grid=np.geomspace(0.01, 100, 61))
        rows.append({"company": c, "months": len(x), "reports": int(x.n.sum()),
                     "freeway share of reports": x.n_fw.sum() / x.n.sum(),
                     "sd of monthly freeway share": (x.n_fw / x.n)[x.n >= 10].std(),
                     "rho": r["rho"], "lo": r["lo"], "hi": r["hi"], "at grid edge": r["edge"]})
    return pd.DataFrame(rows).sort_values("reports", ascending=False).reset_index(drop=True)


def random_effects(byc):
    """
    DerSimonian-Laird random-effects pool of the per-company log ratios (SE from each company's 95%
    profile interval). Returns the pooled ratio, its 95% interval, tau (between-company sd of log rho)
    and the 95% prediction interval for a new company. Companies whose interval hits the grid edge are
    left out (no usable SE).
    """
    from scipy.stats import t as tdist
    x = byc[~byc["at grid edge"]]
    y = np.log(x.rho.to_numpy())
    se = (np.log(x.hi.to_numpy()) - np.log(x.lo.to_numpy())) / (2 * 1.96)
    w = 1 / se ** 2
    ybar = (w * y).sum() / w.sum()
    Q = (w * (y - ybar) ** 2).sum()
    tau2 = max(0.0, (Q - (len(y) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    mu = (ws * y).sum() / ws.sum()
    se_mu = np.sqrt(1 / ws.sum())
    q = tdist.ppf(0.975, max(len(y) - 2, 1))
    return {"companies": len(y), "used": list(x.company), "rho": float(np.exp(mu)),
            "lo": float(np.exp(mu - 1.96 * se_mu)), "hi": float(np.exp(mu + 1.96 * se_mu)),
            "tau": float(np.sqrt(tau2)), "Q": float(Q),
            "pred_lo": float(np.exp(mu - q * np.sqrt(tau2 + se_mu ** 2))),
            "pred_hi": float(np.exp(mu + q * np.sqrt(tau2 + se_mu ** 2)))}


def _sim_setup(p, form="company-year"):
    """What the simulator needs from the real panel: each company-year's surface rate per mile and the
    dispersion at the fitted ratio, and each month's report share shrunk toward the company's."""
    base = ratio_from_miles(p, form)
    L = base["fit"].fittedvalues.to_numpy() / p.miles.to_numpy() * (1 - p.p.to_numpy() * (1 - 1 / base["rho"]))
    tot = p.groupby("company")[["n_fw", "n"]].sum()
    ps = ((p.n_fw + 5.0 * p.company.map(tot.n_fw / tot.n)) / (p.n + 5.0)).to_numpy()
    return {"L": L, "alpha": base["alpha"], "ps": ps, "rho_obs": base["rho"]}


def _sim_panel(p, S, rho, shock, rng):
    """
    One synthetic panel with a known ratio: the real company-months and miles, exposure shares implied
    by the (shrunk) observed shares under that ratio, and NB noise either as one shock per month shared
    by both road classes ('common') or as separate shocks per class ('independent').
    """
    e = S["ps"] / (rho - S["ps"] * (rho - 1))                 # invert p = rho e / (1 - e + rho e)
    M, L, a = p.miles.to_numpy(), S["L"], S["alpha"]
    if shock == "common":
        u = rng.gamma(1 / a, a, len(p))
        n_fw, n_sf = rng.poisson(M * e * rho * L * u), rng.poisson(M * (1 - e) * L * u)
    else:
        n_fw = rng.poisson(rng.gamma(1 / a, a * np.maximum(M * e * rho * L, 1e-12)))
        n_sf = rng.poisson(rng.gamma(1 / a, a * np.maximum(M * (1 - e) * L, 1e-12)))
    q = p[["company", "year", "miles", "t"]].copy()
    q["n_fw"], q["n"] = n_fw, n_fw + n_sf
    q = q[q.groupby(["company", "year"]).n.transform("sum") > 0].copy()
    sh = q.groupby("company").n_fw.sum() / q.groupby("company").n.sum()
    q["p"] = np.where(q.n > 0, q.n_fw / q.n.where(q.n > 0, 1), q.company.map(sh))
    return q


def simulate_ratio(p, true_rhos=(0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 8.0), reps=30,
                   shocks=("independent", "common"), form="company-year", seed=0):
    """
    The estimator on synthetic panels with a known ratio, under two noise structures. Returns
    (summary per noise model and true ratio: median estimate, 5-95% of estimates, coverage of the 95%
    profile interval; every replicate). The profile interval assumes the estimator is unbiased; the
    simulation shows it is not once months are overdispersed, so `confidence_set` is what to report.
    """
    rng = np.random.default_rng(seed)
    S = _sim_setup(p, form)
    rows = []
    for shock in shocks:
        for rho in true_rhos:
            for r in range(reps):
                est = ratio_from_miles(_sim_panel(p, S, rho, shock, rng), form, alpha=S["alpha"],
                                       grid=np.geomspace(0.02, 50, 31))
                rows.append({"noise": shock, "true rho": rho, "rep": r, "rho": est["rho"], "lo": est["lo"], "hi": est["hi"]})
    s = pd.DataFrame(rows)
    s["covered"] = (s.lo <= s["true rho"]) & (s.hi >= s["true rho"])
    summ = s.groupby(["noise", "true rho"]).agg(median=("rho", "median"), q05=("rho", lambda v: v.quantile(.05)),
                                                q95=("rho", lambda v: v.quantile(.95)), coverage=("covered", "mean"))
    return summ, s


def median_unbiased(summ, rho_obs):
    """
    The true ratio whose simulated median estimate equals the observed one, per noise model (log-linear
    interpolation along the simulated median curve), and their geometric mean: the bias-corrected point.
    """
    out = {}
    for noise, g in summ.reset_index().groupby("noise"):
        g = g.sort_values("true rho")
        med = np.maximum.accumulate(g["median"].to_numpy())          # the median curve is increasing
        out[noise] = float(np.exp(np.interp(np.log(rho_obs), np.log(med), np.log(g["true rho"].to_numpy()))))
    out["point"] = float(np.exp(np.mean(np.log([v for k, v in out.items()]))))
    return out


def confidence_set(summ, rho_obs):
    """
    Simulation-calibrated 90% confidence set (a Neyman construction): every true ratio whose simulated
    5-95% band of estimates contains the observed estimate, under each noise model; the union across
    noise models is reported. Valid whatever the estimator's bias, at the resolution of the grid.
    """
    ok = summ[(summ.q05 <= rho_obs) & (summ.q95 >= rho_obs)].reset_index()
    by = ok.groupby("noise")["true rho"].agg(["min", "max"])
    return {"lo": float(ok["true rho"].min()) if len(ok) else np.nan,
            "hi": float(ok["true rho"].max()) if len(ok) else np.nan, "by_noise": by}


# ------------------------------------------------------------------ 3. the ratio under an exposure proxy
# FHWA Highway Statistics 2023, table VM-2 (released Nov 2024), California, millions of vehicle-miles.
# Freeway-type = Interstate + Other Freeways and Expressways (rural + urban), matching OSM motorway/trunk.
VM2_CA_2023 = {"rural": {"interstate": 16964, "other freeway": 5915, "other principal arterial": 11288,
                         "minor arterial": 8487, "major collector": 9801, "minor collector": 1441, "local": 5786},
               "urban": {"interstate": 70833, "other freeway": 58096, "other principal arterial": 50946,
                         "minor arterial": 41194, "major collector": 22097, "minor collector": 153, "local": 13612}}
VM2_URL = "https://www.fhwa.dot.gov/policyinformation/statistics/2023/vm2.cfm"


def vmt_share():
    fw = sum(v[k] for v in VM2_CA_2023.values() for k in ("interstate", "other freeway"))
    tot = sum(sum(v.values()) for v in VM2_CA_2023.values())
    return pd.Series({"freeway": fw / tot, "surface": 1 - fw / tot})


def length_share(tracts=None):
    t = terrain()[1] if tracts is None else tracts
    fw, sf = t.km_fw.sum(), t.km_sf.sum()
    return pd.Series({"freeway": fw / (fw + sf), "surface": sf / (fw + sf)})


def ratio_from_exposure(dis, share):
    """Revision 2's NB GLM with company fixed effects under a fixed exposure split -> freeway/surface ratio."""
    from disengagement_validation import counts, fe_glm
    N = counts(dis, "two-class")
    b = fe_glm(N, share[N.columns])
    return float(np.exp(b["freeway"] - b["surface"]))


# ------------------------------------------------------------------ 4. road situations -> specialised models
# Each situation the narratives name is tied to the OpenStreetMap feature it happens at. Its share of a
# road class's reports scales with that feature's density in a place (per km of that road class,
# relative to California's average): twice the junctions per km, twice the intersection-turn share.
# Situations without a mappable feature (another vehicle's behaviour, lane keeping, obstructions,
# construction) and reports that name no situation are spread evenly over every mile of the class.
SITUATIONS = ["traffic control", "intersection turn", "pedestrian", "cyclist", "lane change or merge",
              "lane keeping", "other vehicle", "obstruction", "construction or emergency", "none stated"]
# situation -> (road class, feature numerator columns, denominator column). Keys ending "(freeway)" are the
# freeway-class version of a situation. A place's feature ratio is its numerator per unit of denominator,
# relative to California's.
TIED = {
    "traffic control": ("surface", ["n_signal", "n_stop", "n_rail"], "km_sf"),
    "intersection turn": ("surface", ["n_junction"], "km_sf"),
    "pedestrian": ("surface", ["n_crossing"], "km_sf"),
    "cyclist": ("surface", ["km_bike"], "km_sf"),
    "lane change or merge": ("surface", ["km_multilane"], "km_sf"),
    "obstruction": ("surface", ["n_poi"], "km_sf"),
    "lane keeping": ("surface", ["turn_sf"], "km_sf"),
    "lane change or merge (freeway)": ("freeway", ["km_ramp"], "km_fw"),
    "lane keeping (freeway)": ("freeway", ["turn_fw"], "km_fw"),
    "other vehicle (freeway)": ("freeway", ["vkm_fw"], "km_counted_fw"),
}
# the five ties of the first version (mapped features only); the extended set adds curbside activity,
# curvature and Caltrans traffic counts
TIED_CORE = ["traffic control", "intersection turn", "pedestrian", "cyclist", "lane change or merge",
             "lane change or merge (freeway)"]
FEATURE_LABEL = {"traffic control": "signals & stop signs", "intersection turn": "junctions",
                 "pedestrian": "crosswalks", "cyclist": "bike lanes", "lane change or merge": "multi-lane streets",
                 "obstruction": "shops & services (curbside)", "lane keeping": "curves",
                 "lane change or merge (freeway)": "freeway ramps", "lane keeping (freeway)": "freeway curves",
                 "other vehicle (freeway)": "freeway traffic (AADT)"}


def situation_of(key):
    return key.replace(" (freeway)", "")


FAULTS = ["Hardware", "Software / system fault"]
SCEN_MODELS = ("q35-9b", "g4-12b")


def report_table(dis=None, scen_models=SCEN_MODELS, cause_models=("q35-9b-zs", "g4-12b-zs")):
    """
    One row per disengagement report with a road box: company, year, road class, and the consensus
    probabilities of each road situation (columns s:<situation>) and of a hardware/software fault
    (p_fault), read from the narratives by the local LLMs (avgap_llm caches).
    """
    import avgap_llm as L
    import avgap_taxonomy as T
    dis = A.load_disengagements() if dis is None else dis
    Lc = T.label_corpus(dis)
    it = L.items()
    item_of = it.set_index("norm").item_id
    d = dis.assign(norm=Lc.norm.values, item_id=Lc.norm.map(item_of).values)
    d = d[d.road_type.isin(FREEWAY_BOXES | SURFACE_BOXES)].copy()
    d["cls"] = np.where(d.road_type.isin(FREEWAY_BOXES), "freeway", "surface")
    cons = L.scen_consensus(L.scen_labels(list(scen_models)), scen_models)
    S = cons[[f"situation:{s}" for s in SITUATIONS]]
    S.columns = [f"s:{s}" for s in SITUATIONS]
    cause = []
    for v in cause_models:
        r = L.load_run(v)
        P = np.vstack(r.probs.values)
        cause.append(pd.Series(P[:, [T.CAT_NAMES.index(f) for f in FAULTS]].sum(axis=1), index=r.item_id))
    pf = pd.concat(cause, axis=1).mean(axis=1).rename("p_fault")
    d = d.join(S, on="item_id").join(pf, on="item_id")
    d["setting"] = d.item_id.map(cons["setting"]) if "setting" in cons else ""
    return d[["company", "month", "cls", "road_type", "item_id", "setting", "p_fault"] + list(S.columns)]


def situation_shares(R, how="stated"):
    """
    Share of each road class's reports in each situation (rows: freeway, surface).
      stated     the narratives as they are: 'none stated' stays unexplained (spread evenly) -> the
                 smallest plausible terrain effect
      imputed    reports that name no situation and are not hardware/software faults are given the
                 class's mix among reports that do name one -> the largest plausible terrain effect
    """
    cols = [f"s:{s}" for s in SITUATIONS]
    out = {}
    for k, x in R.groupby("cls"):
        P = x[cols].to_numpy()
        if how == "imputed":
            none = P[:, -1] * (1 - x.p_fault.fillna(0).to_numpy())
            stated = P[:, :-1].sum(axis=0)
            mix = stated / stated.sum()
            P = P.copy()
            P[:, :-1] += none[:, None] * mix[None, :]
            P[:, -1] -= none
        out[k] = P.mean(axis=0)
    return pd.DataFrame(out, index=SITUATIONS).T


def situation_bootstrap(R, B=200, seed=0):
    """Cluster bootstrap over companies of both share estimates -> arrays (B, class, situation)."""
    rng = np.random.default_rng(seed)
    comps = R.company.unique()
    groups = {c: R[R.company == c] for c in comps}
    out = {"stated": [], "imputed": []}
    for _ in range(B):
        pick = rng.choice(comps, size=len(comps), replace=True)
        Rb = pd.concat([groups[c] for c in pick], ignore_index=True)
        for how in out:
            out[how].append(situation_shares(Rb, how).reindex(["freeway", "surface"]).fillna(0).to_numpy())
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------ 6. aggregated risk per mile
PRIOR_KM = 1.0          # feature densities are shrunk toward California's average with 1 km of prior road
# Dose-response: a situation's rate per mile scales with (feature density ratio) ** BETA. Linear (1) puts
# downtown tracts at 20x+ (their signal and shop densities are 50-100x the statewide road mile) and fails
# the CPUC calibration check; 0.5 (square root: diminishing returns) keeps the ranking almost unchanged
# on both crash records (notebook 17, section 5). The 90% range draws BETA from BETA_RANGE.
BETA = 0.5
BETA_RANGE = (0.25, 1.0)


def feature_ratios(T_, ref, prior_km=PRIOR_KM, tied=None, beta=None):
    """
    Each tied feature's density in every place relative to California's average, per unit of its
    denominator (km of the road class it lives on; counted freeway km for traffic), shrunk with `prior_km`
    of average road (a 50 m stub with one signal must not read as 20 signals per km). `ref` is the statewide
    terrain table that sets the averages. Columns a place lacks are treated as average (ratio 1).
    """
    out = {}
    for s, (k, cols, den) in (tied or TIED).items():
        if not all(c in T_.columns and c in ref.columns for c in cols + [den]):
            out[s] = np.ones(len(T_))
            continue
        avg = ref[cols].to_numpy().sum() / ref[den].sum()
        num, d = T_[cols].sum(axis=1).to_numpy(), T_[den].to_numpy()
        out[s] = (((num + prior_km * avg) / (d + prior_km)) / avg) ** (BETA if beta is None else beta)
    return pd.DataFrame(out, index=T_.index)


def multipliers(T_, ref, rho, shares, tied=None, beta=None):
    """
    Relative risk per mile of each place, before scaling (score_all divides by California's road-km mean):
        surface part:  sum over situations of share x (its surface feature ratio, or 1 if untied)
        freeway part:  rho x the same with the freeway shares and freeway feature ratios
        place:         the two weighted by the place's surface and freeway road km
    Returns (place multiplier, surface multiplier, freeway multiplier, per-situation surface terms).
    `shares` is a (class x situation) frame from situation_shares; `tied` defaults to TIED.
    """
    tied = TIED if tied is None else tied
    fr = feature_ratios(T_, ref, tied=tied, beta=beta)
    sf, fw = shares.loc["surface"], shares.loc["freeway"]
    terms = pd.DataFrame(index=T_.index)
    for s in SITUATIONS:
        terms[s] = sf[s] * (fr[s] if s in fr else 1.0)
    m_sf = terms.sum(axis=1)
    m_fw = sum(fw[s] * (fr[f"{s} (freeway)"] if f"{s} (freeway)" in fr else 1.0) for s in SITUATIONS)
    m_fw = pd.Series(np.broadcast_to(np.asarray(m_fw, float), len(T_)), index=T_.index) * rho
    ksf, kfw = T_.km_sf.to_numpy(), T_.km_fw.to_numpy()
    tot = ksf + kfw
    place = pd.Series(np.where(tot > 0, (ksf * m_sf + kfw * m_fw) / np.where(tot > 0, tot, 1), np.nan), index=T_.index)
    return place, m_sf, m_fw, terms


def monte_carlo(T_, ref, log_rho_draws, share_draws, seed=0, tied=None, beta_range=None):
    """
    The place multiplier under every draw: the ratio from the company bootstrap of ratio_from_miles,
    the situation shares from the company bootstrap of situation_shares, and how much of the unnamed
    reports follow the named mix (uniform between 'stated' and 'imputed'). -> (places x draws) array.
    """
    rng = np.random.default_rng(seed)
    B = min(len(log_rho_draws), len(share_draws["stated"]))
    lo_b, hi_b = BETA_RANGE if beta_range is None else beta_range
    out = np.empty((len(T_), B))
    for b in range(B):
        w = rng.uniform()
        beta = rng.uniform(lo_b, hi_b)
        sh = (1 - w) * share_draws["stated"][b] + w * share_draws["imputed"][b]
        sh = pd.DataFrame(sh, index=["freeway", "surface"], columns=SITUATIONS)
        out[:, b] = multipliers(T_, ref, float(np.exp(log_rho_draws[b])), sh, tied, beta)[0].to_numpy()
    return out


# ------------------------------------------------------------------ 7. validation against located AV collisions
# Disengagements have no location, so the map cannot be checked against them. The located AV outcomes we
# have are collisions: the DMV OL 316 reports of every company, 2019-24, re-geocoded in notebook 07, and
# Waymo's CPUC filings per tract and quarter, 2025-26 (notebook 06). Neither was used to build the
# disengagement map, so agreement is out-of-source evidence that it marks places where AVs struggle.
def located_crashes(refresh=False):
    """DMV OL 316 crashes re-geocoded against OSM (notebook 07) and their boundary-aware tract split
    (cached in av_data/disengage/)."""
    import pickle
    import avgap_external as X
    path = os.path.join(DIS_DIR, "ol316_located.pkl")
    if os.path.exists(path) and not refresh:
        return pickle.load(open(path, "rb"))
    ol = X.locate(X.geocode_crashes(X.load_ol316()))
    out = (ol, X.allocate(ol))
    pickle.dump(out, open(path, "wb"))
    return out


def _spearman_per_km(x, col, y, km, min_crashes=10):
    """Within-county Spearman of a score with crashes per km of road, weighted by each county's crashes."""
    from scipy.stats import spearmanr
    r = [(spearmanr(g[col], g[y] / g[km])[0], g[y].sum()) for _, g in x.groupby("county")
         if g[y].sum() >= min_crashes and g[col].nunique() > 1]
    return float(np.average([a for a, _ in r], weights=[b for _, b in r])) if r else np.nan


def crash_check(L, scores, y="crashes", km="road_km"):
    """
    Listed tracts (where AVs drove), located crashes: for each score, the within-county Spearman with
    crash counts, and the NB elasticity of crashes per km of road with respect to the score (county fixed
    effects, offset log road km; > 0 = more crashes per km where the score is higher).
    """
    import avgap_external as X
    rows = []
    for name, col in scores.items():
        x = L[[y, col, "county", km]].dropna()
        x = x[(x[col] > 0) & (x[km] > 0)].copy()
        rho, _ = X.within_county_spearman(x, col, y=y)
        rho_km = _spearman_per_km(x, col, y, km)
        x["ls"] = np.log(x[col]) - np.log(x[col]).mean()
        x = X._fe_counties(x, y)
        m = X.fit_nb(f"{y} ~ ls + C(county)", x, offset=np.log(x[km]))
        lo, hi = m.conf_int().loc["ls"]
        rows.append({"score": name, "within-county Spearman": rho, "Spearman per km": rho_km,
                     "elasticity per km": m.params["ls"],
                     "95% CI": f"{lo:.2f} to {hi:.2f}", "AIC": m.aic, "tracts": len(x)})
    out = pd.DataFrame(rows).set_index("score")
    out["dAIC vs best"] = out.AIC - out.AIC.min()
    return out.drop(columns="AIC")


def panel_check(d, extra, k=8):
    """
    Waymo's CPUC collisions per tract-quarter (notebook 06's panel), spatial K-fold, held-out deviance
    reduction over the quarter mean. Base model: service maturity + log road km. Each score enters two
    ways: as an offset with no fitted weight (collisions proportional to road km x the score: the map
    used exactly as published, nothing learned from collisions), and as a fitted covariate (its log,
    standardised). Returns a frame: % deviance reduction per score and way.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import avgap_place as P
    base = ["log_months_listed", "log_road_km"]
    d = d.dropna(subset=base + list(extra.values())).copy()
    d["fold"] = P.spatial_folds(d, k)
    for c in extra.values():
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    specs = {("maturity + road km", "-"): (base, None)}
    for n, c in extra.items():
        specs[(n, "as published (offset)")] = (base, c)
        specs[(n, "fitted weight")] = (base + [c + "_z"], None)
    preds = {key: np.zeros(len(d)) for key in specs}
    null = np.zeros(len(d))
    for f in range(k):
        tr, te = d[d.fold != f], d[d.fold == f]
        idx = (d.fold == f).to_numpy()
        null[idx] = te.quarter.map(tr.groupby("quarter").collisions.mean()).to_numpy()
        for key, (feats, off) in specs.items():
            m = smf.glm("collisions ~ C(quarter) + " + " + ".join(feats), tr,
                        offset=None if off is None else tr[off], family=sm.families.NegativeBinomial(alpha=0.5)).fit()
            preds[key][idx] = m.predict(te, offset=None if off is None else te[off]).to_numpy()
    dev0 = P.poisson_dev(d.collisions, null)
    out = pd.Series({key: 100 * (1 - P.poisson_dev(d.collisions, p) / dev0) for key, p in preds.items()})
    return out.unstack().rename_axis(index="score", columns="held-out deviance reduction %")


def map_loco(R, T_, ref, rho, companies):
    """Leave each company out of the situation shares: Spearman of the tract map with the full one."""
    from scipy.stats import spearmanr
    full = multipliers(T_, ref, rho, 0.5 * situation_shares(R, "stated") + 0.5 * situation_shares(R, "imputed"))[0]
    rows = []
    for c in companies:
        Rc = R[R.company != c]
        m = multipliers(T_, ref, rho, 0.5 * situation_shares(Rc, "stated") + 0.5 * situation_shares(Rc, "imputed"))[0]
        top = set(full.nlargest(len(full) // 10).index)
        rows.append({"left out": c, "reports left out": int((R.company == c).sum()),
                     "Spearman with full map": spearmanr(full, m, nan_policy="omit")[0],
                     "top tenth kept": len(top & set(m.nlargest(len(m) // 10).index)) / len(top)})
    return pd.DataFrame(rows)


def period_check(fit, T_, ref, scope, split=2022):
    """
    Out-of-time stability: the situation shares (midpoint of stated and imputed) and the map rebuilt from
    the reports of 2019-`split` and of the years after, separately, and the raw filed-miles ratio on each
    period's panel (uncalibrated: the simulation calibration was run on the full panel only).
    `scope` = the tract ids the dashboard shows. Returns (shares by period, comparison table).
    """
    from scipy.stats import spearmanr
    R, p = fit["report_table"], fit["panel"]
    yr = R.month.dt.year
    parts = {f"2019-{str(split)[2:]}": R[yr <= split], f"{str(split + 1)[2:]}-24": R[yr > split]}
    sh = {k: 0.5 * situation_shares(x, "stated") + 0.5 * situation_shares(x, "imputed") for k, x in parts.items()}
    sh["all years"] = _mid_shares(fit)
    shares = pd.concat({k: v.T for k, v in sh.items()}, axis=1)
    maps = {k: multipliers(T_, ref, fit["rho"], v)[0] for k, v in sh.items()}
    (a, ma), (b, mb) = [(k, maps[k]) for k in parts]
    sc = [g for g in scope if g in ma.index]
    top = lambda m: set(m.reindex(sc).nlargest(len(sc) // 10).index)
    py = p.year.astype(int)
    vc = {k: x.company.value_counts(normalize=True) for k, x in parts.items()}
    rows = {"reports": [len(parts[a]), len(parts[b]), ""],
            "companies": [parts[a].company.nunique(), parts[b].company.nunique(), ""],
            "largest filer": [f"{v.index[0]} ({v.iloc[0]:.0%})" for v in vc.values()] + [""],
            "raw freeway / surface ratio (filed miles)": [ratio_from_miles(p[py <= split])["rho"],
                                                          ratio_from_miles(p[py > split])["rho"], ""],
            "map Spearman, all CA tracts": ["", "", spearmanr(ma, mb, nan_policy="omit")[0]],
            "map Spearman, dashboard tracts": ["", "", spearmanr(ma.reindex(sc), mb.reindex(sc), nan_policy="omit")[0]],
            "dashboard top tenth kept": ["", "", len(top(ma) & top(mb)) / len(top(ma))]}
    comp = pd.DataFrame(rows, index=[a, b, f"{a} vs {b}"], dtype=object).T
    return shares, comp


def crash_situations(variants=SCEN_MODELS):
    """Consensus road situation of each DMV OL 316 collision narrative (avgap_llm scen-crash caches),
    indexed like located_crashes()'s rows (crash item dmv-<i> is row i)."""
    import avgap_llm as L
    frames = [pd.read_json(L.scen_path(v, "crash"), lines=True) for v in variants
              if os.path.exists(L.scen_path(v, "crash"))]
    if not frames:
        return pd.Series(dtype=object)
    lab = pd.concat(frames, ignore_index=True)
    P = lab.groupby("item_id").probs.apply(lambda x: np.mean(np.vstack(x.values), axis=0))
    out = P.map(lambda v: SITUATIONS[int(np.argmax(v))])
    out = out[out.index.str.startswith("dmv-")]
    out.index = out.index.str[4:].astype(int)
    return out


def situation_check(Lt, ol, alloc, layers, km="road_km"):
    """
    Specificity of the specialised models: each situation's layer against located autonomous-mode crashes
    whose narrative names that situation, and against all other crashes (within-county Spearman per km of
    road, counties with 5+ / 10+ such crashes). A layer that tracks its own kind of crash better than the
    rest is measuring that situation, not just busy streets.
    """
    import avgap_external as X
    sit = crash_situations().reindex(ol.index)
    rows = []
    for s, col in layers.items():
        x = Lt.copy()
        x["own"] = X.crash_counts(ol[sit == s], alloc, Lt, mode="autonomous").to_numpy()
        x["other"] = X.crash_counts(ol[sit != s], alloc, Lt, mode="autonomous").to_numpy()
        rows.append({"situation": s, "located crashes naming it": round(float(x.own.sum()), 1),
                     "Spearman per km: its own crashes": _spearman_per_km(x, col, "own", km, min_crashes=5),
                     "Spearman per km: all other crashes": _spearman_per_km(x, col, "other", km)})
    return pd.DataFrame(rows).set_index("situation")


# Each layer claims its feature marks where a road situation is common. Police-reported crashes of all
# vehicles (CCRS 2022-24, notebook 12) carry a collision type and what was hit, so every layer can be held
# against crashes of its own kind, where that situation happened and went wrong for a human driver: hundreds
# of thousands of located crashes where the AV record has a handful per situation. Busy streets have more of
# every crash, so the test is on the *mix*: the share of a cell's crashes that are of the layer's type, in a
# joint quasi-binomial GLM with every layer of the road class and county fixed effects. A layer passes if it
# raises its own type's share beyond what the other layers explain.
POLICE_TYPES = {
    "traffic control": "at a signal or stop sign (control device present)",
    "intersection turn": "broadside, vehicle vs. vehicle",
    "pedestrian": "vehicle vs. pedestrian",
    "cyclist": "vehicle vs. bicycle",
    "lane change or merge": "sideswipe, vehicle vs. vehicle",
    "obstruction": "hit a parked vehicle",
    "lane keeping": "hit an object or overturned (left the lane)",
    "lane change or merge (freeway)": "freeway sideswipe",
    "lane keeping (freeway)": "freeway: hit an object or overturned",
    "other vehicle (freeway)": "freeway rear-end",
}
POLICE_COUNTIES = ["Los Angeles", "San Francisco", "San Mateo", "Santa Clara"]     # notebook 12's counties


def police_crashes(refresh=False):
    """CCRS crashes 2022-24 in notebook 12's four counties, located (avgap_context.locate_ccrs), with a
    boolean column per POLICE_TYPES key and the road class ('surface' = not highway-related, 'freeway' =
    IsFreeway). Cached in av_data/disengage/ccrs_located.pkl (~2 minutes to build)."""
    import pickle
    import avgap_context as J
    path = os.path.join(DIS_DIR, "ccrs_located.pkl")
    if os.path.exists(path) and not refresh:
        c = pickle.load(open(path, "rb"))
    else:
        c = J.locate_ccrs(J.load_ccrs(counties=POLICE_COUNTIES))
        pickle.dump(c, open(path, "wb"))
    c = c[c.loc_source != "unlocated"].copy()
    ct, mv = c["Collision Type Description"].fillna(""), c.MotorVehicleInvolvedWithDesc.fillna("")
    sf, fw = ~c.highway, c.IsFreeway.eq("True")
    left_lane = ct.isin(["HIT OBJECT", "OVERTURNED"]) | mv.eq("FIXED OBJECT")
    c["cls"] = np.select([fw, sf], ["freeway", "surface"], "other")
    for k, m in {"traffic control": sf & c.TrafficControlDeviceCode.isin(["A", "B", "C"]),   # D = no control
                 "intersection turn": sf & ct.eq("BROADSIDE") & mv.eq("OTHER MOTOR VEHICLE"),
                 "pedestrian": sf & mv.eq("PEDESTRIAN"), "cyclist": sf & mv.eq("BICYCLE"),
                 "lane change or merge": sf & ct.eq("SIDE SWIPE") & mv.eq("OTHER MOTOR VEHICLE"),
                 "obstruction": sf & mv.eq("PARKED MOTOR VEHICLE"), "lane keeping": sf & left_lane,
                 "lane change or merge (freeway)": fw & ct.eq("SIDE SWIPE"),
                 "lane keeping (freeway)": fw & left_lane,
                 "other vehicle (freeway)": fw & ct.eq("REAR END")}.items():
        c["t:" + k] = m.to_numpy()
    r, cc = cell_of(c.loc_lon, c.loc_lat)
    c["cell"] = [f"{a}_{b}" for a, b in zip(r, cc)]
    return c


def _joint_glm(x, X, k, n):
    """Quasi-binomial GLM of type k's share of the cell's n crashes on every layer (standardised log
    density) + county fixed effects -> coefficients of the layers."""
    import statsmodels.api as sm
    Z = pd.get_dummies(x.county, drop_first=True, dtype=float)
    exog = sm.add_constant(pd.concat([X, Z], axis=1), has_constant="add")
    y = np.column_stack([x["t:" + k], x[n] - x["t:" + k]])
    return sm.GLM(y, exog, family=sm.families.Binomial()).fit(scale="X2").params[X.columns]


def police_check(cells, ref, county_of, min_n=20, min_km=0.2, B=200, block=10, seed=0):
    """
    Specificity of the layers against police-reported crashes (see POLICE_TYPES), at 500 m cells in
    POLICE_COUNTIES with min_km+ of the road class and min_n+ crashes on it.
      share Spearman   within-county Spearman of each layer with each type's share of the cell's crashes
      joint GLM        odds ratio per sd of log feature density, all layers of the class together; the
                       90% range is a block bootstrap (block x block cells, ~5 km squares, resampled within
                       county), since neighbouring cells are not independent
    Returns (summary per layer, matrix of joint odds ratios [crash type x layer], cell table).
    """
    from scipy.stats import spearmanr
    c = police_crashes()
    cl = cells.assign(cell=[f"{a}_{b}" for a, b in zip(cells.row, cells.col)]).set_index("cell")
    cl["county"] = cl.GEOID.astype(str).map(county_of)
    cl = cl[cl.county.isin(POLICE_COUNTIES)].copy()
    for k in ("surface", "freeway"):
        cl["n_" + k] = c[c.cls == k].groupby("cell").size().reindex(cl.index).fillna(0).to_numpy()
    for k in POLICE_TYPES:
        cl["t:" + k] = c[c["t:" + k]].groupby("cell").size().reindex(cl.index).fillna(0).to_numpy()
    fr = np.log(feature_ratios(cl, ref, beta=1.0))
    rng = np.random.default_rng(seed)
    rows, mat = [], {}
    for cls, km in (("surface", "km_sf"), ("freeway", "km_fw")):
        lay = [l for l in POLICE_TYPES if l.endswith("(freeway)") == (cls == "freeway")]
        n = "n_" + cls
        x = cl[(cl[n] >= min_n) & (cl[km] >= min_km)].copy()
        X = fr.loc[x.index, lay]
        X = (X - X.mean()) / X.std()
        blk = x.county + ":" + (x.row // block).astype(str) + "_" + (x.col // block).astype(str)
        groups = {cty: [np.flatnonzero((blk == b).to_numpy()) for b in blk[x.county == cty].unique()]
                  for cty in x.county.unique()}
        boot = []
        for _ in range(B):
            idx = np.concatenate([gs[i] for gs in groups.values() for i in rng.integers(0, len(gs), len(gs))])
            boot.append({k: _joint_glm(x.iloc[idx], X.iloc[idx], k, n).to_numpy() for k in lay})
        for k in lay:
            b = _joint_glm(x, X, k, n)
            mat[k] = np.exp(b)
            bk = np.array([d[k] for d in boot])
            j = lay.index(k)
            own_first = (bk.argmax(axis=1) == j).mean()
            sh = x["t:" + k] / x[n]
            rho = {l: np.average([spearmanr(g[l], sh[g.index])[0] for _, g in X.assign(county=x.county).groupby("county")],
                                 weights=[x.loc[g.index, "t:" + k].sum() for _, g in X.assign(county=x.county).groupby("county")])
                   for l in lay}
            other = max((l for l in lay if l != k), key=lambda l: b[l])
            rows.append({"layer": FEATURE_LABEL[k], "situation": k, "police-reported crash type": POLICE_TYPES[k],
                         "crashes": int(x["t:" + k].sum()), "share of the class": x["t:" + k].sum() / x[n].sum(),
                         "cells": len(x), "own odds ratio per sd": float(np.exp(b[k])),
                         "lo": float(np.exp(np.percentile(bk[:, j], 5))), "hi": float(np.exp(np.percentile(bk[:, j], 95))),
                         "rank among layers": int((b > b[k]).sum() + 1), "of": len(lay),
                         "own first in bootstrap": float(own_first),
                         "strongest other layer": FEATURE_LABEL[other], "its odds ratio": float(np.exp(b[other])),
                         "share Spearman: own": float(rho[k]),
                         "share Spearman: best other": float(max(v for l, v in rho.items() if l != k))})
    M = pd.DataFrame(mat).T.reindex(columns=list(POLICE_TYPES)).rename(columns=FEATURE_LABEL)
    return pd.DataFrame(rows).set_index("layer"), M, cl


def cell_check(cell_scores, cells, ol, footprint, county_of, min_km=0.2):
    """
    The check at the prediction unit: located autonomous-mode DMV crashes counted per 500 m cell, for the
    cells (min_km+ of road) inside the deployment footprint that every score covers. For each
    score: within-county Spearman with crashes per km, and the share of crashes in the riskiest tenth of
    cells against that tenth's share of road km (lift). `cell_scores` maps a name to a Series indexed
    "row_col".
    """
    from scipy.stats import spearmanr
    x = ol[(ol["mode"] == "autonomous") & ol.loc_lon.notna()]
    r, c = cell_of(x.loc_lon, x.loc_lat)
    n = pd.Series(1, index=[f"{a}_{b}" for a, b in zip(r, c)]).groupby(level=0).sum()
    cl = cells.assign(cell=[f"{a}_{b}" for a, b in zip(cells.row, cells.col)]).set_index("cell")
    cl["road_km"] = cl.km_sf + cl.km_fw
    cl = cl[cl.GEOID.isin(footprint) & (cl.road_km >= min_km)].copy()
    cl["county"] = cl.GEOID.map(county_of)
    cl["crashes"] = n.reindex(cl.index).fillna(0).to_numpy()
    cl["per_km"] = cl.crashes / cl.road_km
    common = set.intersection(*[set(v.dropna().index) for v in cell_scores.values()])
    cl = cl[cl.index.isin(common)].copy()                      # every score on the same cells
    rows = []
    for name, sc in cell_scores.items():
        v = sc.reindex(cl.index)
        ok = v.notna()
        rho = [(spearmanr(v[ok & (cl.county == k)], cl.per_km[ok & (cl.county == k)])[0], cl.crashes[cl.county == k].sum())
               for k in cl.county.unique() if cl.crashes[ok & (cl.county == k)].sum() >= 10]
        top = v[ok].rank(pct=True, method="average") > 0.9          # ties stay together
        top = top.reindex(cl.index).fillna(False).astype(bool)
        rows.append({"score": name, "cells": int(ok.sum()), "crashes": float(cl.crashes[ok].sum()),
                     "within-county Spearman per km": float(np.average([a for a, _ in rho], weights=[b for _, b in rho])) if rho else np.nan,
                     "top tenth: share of crashes": cl.crashes[top].sum() / cl.crashes[ok].sum(),
                     "top tenth: share of road km": cl.road_km[top].sum() / cl.road_km[ok].sum()})
    out = pd.DataFrame(rows).set_index("score")
    out["lift"] = out["top tenth: share of crashes"] / out["top tenth: share of road km"]
    return out, cl


def situation_sheet(per_label=10, seed=0, write=True):
    """
    A blind check of the LLMs' road-situation labels for two human annotators: up to `per_label`
    distinct descriptions per consensus situation (100 in all), shuffled. The sheet has the text and an
    empty answer column; the key has the LLM consensus, its confidence and whether the two models agreed.
    Written to outputs/situation_check_sheet.csv and situation_check_key.csv.
    """
    import avgap_llm as L
    it = L.items().set_index("item_id")
    cons = L.scen_consensus(L.scen_labels(list(SCEN_MODELS)), SCEN_MODELS)
    rng = np.random.default_rng(seed)
    pick = []
    for lab, g in cons.groupby("situation"):
        ids = g.index.to_numpy()
        pick += list(rng.choice(ids, size=min(per_label, len(ids)), replace=False))
    pick = list(rng.permutation(pick))
    key = pd.DataFrame({"check_id": [f"S{i + 1:03d}" for i in range(len(pick))], "item_id": pick,
                        "company": it.loc[pick, "company"].to_numpy(), "reports": it.loc[pick, "rows"].astype(int).to_numpy(),
                        "llm_situation": cons.loc[pick, "situation"].to_numpy(),
                        "llm_confidence": cons.loc[pick, "situation_conf"].round(3).to_numpy(),
                        "llm_models_agree": cons.loc[pick, "situation_agree"].to_numpy()})
    sheet = pd.DataFrame({"check_id": key.check_id, "description": it.loc[pick, "text"].to_numpy(),
                          "situation (one of: " + "; ".join(SITUATIONS) + ")": "", "annotator": "", "notes": ""})
    if write:
        sheet.to_csv(os.path.join(OUT, "situation_check_sheet.csv"), index=False, encoding="utf-8-sig")
        key.to_csv(os.path.join(OUT, "situation_check_key.csv"), index=False, encoding="utf-8-sig")
    return sheet, key


# ------------------------------------------------------------------ outputs for the dashboard
def area_table(T_, ref, rho, shares, draws=None):
    """Point multiplier, 90% range (from Monte Carlo draws), road mix and per-situation terms of places."""
    m, m_sf, m_fw, terms = multipliers(T_, ref, rho, shares)
    out = pd.DataFrame({"risk_multiplier": m, "surface_multiplier": m_sf, "freeway_multiplier": m_fw}, index=T_.index)
    if draws is not None:
        out["risk_multiplier_lo"] = np.nanpercentile(draws, 5, axis=1)
        out["risk_multiplier_hi"] = np.nanpercentile(draws, 95, axis=1)
    tot = (T_.km_sf + T_.km_fw).replace(0, np.nan)
    out["road_km"] = tot
    out["pct_freeway"] = T_.km_fw / tot
    out["pct_arterial"] = T_.km_arterial / tot
    out["pct_local"] = (T_.km_sf - T_.km_arterial) / tot
    # each tied situation's extra (or reduced) risk relative to an average place, in multiplier units
    fr = feature_ratios(T_, ref)
    for key, (k, _, _) in TIED.items():
        base = shares.loc[k, situation_of(key)]
        w = (T_.km_sf if k == "surface" else T_.km_fw * rho) / tot
        out[f"term:{key}"] = w * base * (fr[key] - 1)
    return out


# ------------------------------------------------------------------ the whole pipeline (cached)
FIT_CACHE = os.path.join(DIS_DIR, "fit_v3.pkl")
SCOPE_COUNTIES = ["San Francisco", "Los Angeles", "Santa Clara"]     # the dashboard's counties
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def company_rates(dis=None, mil=None, min_miles=1000):
    """Reports per 1,000 autonomous miles per company and year (companies with min_miles+ that year)."""
    dis = A.load_disengagements() if dis is None else dis
    mil = A.load_mileage()[0] if mil is None else mil
    n = dis.groupby(["company", dis.month.dt.year]).size().rename("reports")
    m = mil.groupby(["company", mil.month.dt.year]).miles.sum().rename("miles")
    r = pd.concat([n, m], axis=1).fillna(0).rename_axis(["company", "year"]).reset_index()
    r = r[r.miles >= min_miles].copy()
    r["per_1k_miles"] = r.reports / r.miles * 1000
    return r


def fit_all(refresh=False, B=200, sim_reps=40):
    """Every fitted quantity notebook 17 and the dashboard need; ~15 minutes, cached in FIT_CACHE."""
    import pickle
    import warnings
    if os.path.exists(FIT_CACHE) and not refresh:
        return pickle.load(open(FIT_CACHE, "rb"))
    warnings.simplefilter("ignore")
    dis, mil = A.load_disengagements(), A.load_mileage()[0]
    p = panel(dis, mil)
    f = {}
    for form in RATIO_FORMS:
        r = ratio_from_miles(p, form)
        f[form] = {k: v for k, v in r.items() if k != "fit"}
    byc = ratio_by_company(p)
    out = {"panel": p, "forms": f, "rho_estimate": f["company-year"]["rho"], "by_company": byc,
           "random_effects": random_effects(byc), "loco": ratio_loco(p),
           "log_rho_boot": ratio_bootstrap(p, B=B)}
    out["sim_summary"], out["sims"] = simulate_ratio(p, reps=sim_reps)
    out["confidence_set"] = confidence_set(out["sim_summary"], out["rho_estimate"])
    out["median_unbiased"] = median_unbiased(out["sim_summary"], out["rho_estimate"])
    # the map uses the bias-corrected point, and draws spread evenly (in log) across the calibrated set
    lo, hi = out["confidence_set"]["lo"], out["confidence_set"]["hi"]
    out["rho"] = out["median_unbiased"]["point"]
    out["log_rho_draws"] = np.random.default_rng(1).uniform(np.log(lo), np.log(hi), B)
    _, tr, _ = terrain()
    out["exposure"] = pd.DataFrame({
        "freeway share of exposure": [length_share(tr)["freeway"], vmt_share()["freeway"], np.nan],
        "freeway / surface rate per mile": [ratio_from_exposure(dis, length_share(tr)),
                                            ratio_from_exposure(dis, vmt_share()), f["company-year"]["rho"]]},
        index=["road length (OSM), revision 2", "vehicle-miles (FHWA VM-2 2023)", "filed miles, within company (revision 3)"])
    R = report_table(dis)
    out["report_table"] = R
    out["shares"] = {h: situation_shares(R, h) for h in ("stated", "imputed")}
    out["share_boot"] = situation_bootstrap(R, B=B)
    out["rates"] = company_rates(dis, mil)
    out["reference_rate"] = float(out["rates"].query("year == 2024").per_1k_miles.median())
    out["reports_by_class"] = R.groupby("cls").size()
    pickle.dump(out, open(FIT_CACHE, "wb"))
    return out


def _mid_shares(fit):
    return 0.5 * fit["shares"]["stated"] + 0.5 * fit["shares"]["imputed"]


def _stretch_terrain(st, ref):
    """Stretches in the terrain layout; features the stretch table doesn't carry (bike lanes, ramps, nearby
    shops) are set to California's average for its road class, so they neither raise nor lower it."""
    sf = (st.cls > 0).to_numpy()
    km = st.km.to_numpy()
    avg = lambda c, d: ref[c].sum() / ref[d].sum()
    out = {"km_sf": np.where(sf, km, 0.0), "km_fw": np.where(sf, 0.0, km),
           "km_bike": np.where(sf, km * avg("km_bike", "km_sf"), 0.0),
           "km_multilane": np.where(sf & (st.lanes >= 4).to_numpy(), km, 0.0),
           "km_ramp": np.where(sf, 0.0, km * avg("km_ramp", "km_fw")), "km_arterial": np.where(st.cls == 1, km, 0.0),
           "n_junction": st.n_junction, "n_signal": st.n_signal, "n_stop": st.n_stop,
           "n_crossing": st.n_crossing, "n_rail": st.n_rail}
    if "n_poi" in ref:
        out["n_poi"] = np.where(sf, km * avg("n_poi", "km_sf"), 0.0)
    if "turn_deg" in st:
        out["turn_sf"] = np.where(sf, st.turn_deg, 0.0)
        out["turn_fw"] = np.where(sf, 0.0, st.turn_deg)
    if "vkm" in st:
        out["vkm_fw"] = np.where(sf, 0.0, st.vkm)
        out["km_counted_fw"] = np.where(sf, 0.0, st.km_counted)
    return pd.DataFrame(out, index=st.index)


def score_all(fit=None, write=True):
    """
    Score cells, tracts, cities, counties and road stretches. Multipliers are normalised so that
    California's road-km-weighted average is 1 (an average California road mile); per 1,000 miles uses
    the 2024 median company's rate. Writes the dashboard's CSVs to outputs/.
    """
    from avgap_coverage import tract_table
    fit = fit_all() if fit is None else fit
    cells, tr, st = terrain()
    ref = tr
    rho = fit["rho"]
    shares = _mid_shares(fit)
    lrb, sb = fit["log_rho_draws"], fit["share_boot"]
    tr = tr.set_index("GEOID")
    tt, _ = tract_table()
    city = pd.read_csv(os.path.join(OUT, "tract_scores.csv"), dtype={"GEOID": str}).set_index("GEOID").city
    tr["county"] = tt.set_index("GEOID").county.reindex(tr.index)
    tr["population"] = tt.set_index("GEOID").population.reindex(tr.index)
    tr["city"] = city.reindex(tr.index)
    kcols = [c for c in ref.columns if c.startswith(("km_", "n_"))]
    cty = tr.groupby("county")[kcols].sum()
    cit = tr[tr.city.notna()].groupby(["county", "city"])[kcols].sum()
    cit.index = [f"{c}|{n}" for c, n in cit.index]
    # normaliser: California's road-km-weighted mean multiplier, point and per draw
    state = pd.DataFrame([ref[kcols].sum()], index=["CA"])
    norm = float(multipliers(state, ref, rho, shares)[0].iloc[0])
    norm_d = monte_carlo(state, ref, lrb, sb)[0]
    res = {}
    for name, T_ in {"county": cty, "city": cit, "tract": tr[kcols]}.items():
        d = monte_carlo(T_, ref, lrb, sb) / norm_d[None, :]
        a = area_table(T_, ref, rho, shares, d)
        a[["risk_multiplier", "surface_multiplier", "freeway_multiplier"]] /= norm
        for c in [c for c in a.columns if c.startswith("term:")]:
            a[c] /= norm
        res[name] = a.join(tr[["county", "city", "population"]]) if name == "tract" else a
    rate = fit["reference_rate"]
    n_fw, n_sf = fit["reports_by_class"]["freeway"], fit["reports_by_class"]["surface"]
    for a in res.values():
        a["per_1k_miles"] = rate * a.risk_multiplier
        a["per_1k_lo"], a["per_1k_hi"] = rate * a.risk_multiplier_lo, rate * a.risk_multiplier_hi
        a["n_sample"] = (a.pct_freeway * n_fw + (1 - a.pct_freeway) * n_sf).round(0)
        terms = a[[c for c in a.columns if c.startswith("term:")]]
        none = terms.isna().all(axis=1)
        a["top_feature"] = terms.fillna(-np.inf).idxmax(axis=1).str[5:].map(FEATURE_LABEL).where(~none)
        a["top_feature_term"] = terms.max(axis=1)
    # cells in the dashboard's counties
    sc_ids = set(tr.index[tr.county.isin(SCOPE_COUNTIES)])
    cs = cells[cells.GEOID.astype(str).isin(sc_ids)].copy()
    cs = cs[(cs.km_sf + cs.km_fw) > 0.05]
    cs.index = [f"{r}_{c}" for r, c in zip(cs.row, cs.col)]
    dcell = monte_carlo(cs[kcols], ref, lrb, sb) / norm_d[None, :]
    ac = area_table(cs[kcols], ref, rho, shares, dcell)
    ac[["risk_multiplier", "surface_multiplier", "freeway_multiplier"]] /= norm
    ac = ac.join(cs[["row", "col", "lon", "lat", "GEOID"]])
    ac["per_1k_miles"] = rate * ac.risk_multiplier
    res["cell"] = ac
    # stretches: named streets inside the dashboard's counties, 200 m or more
    ss = st[st.GEOID.astype(str).isin(sc_ids) & (st.km >= 0.2)].copy()
    sm = multipliers(_stretch_terrain(ss, ref), ref, rho, shares)[0] / norm
    ss["risk_multiplier"] = sm.to_numpy()
    ss["per_1k_miles"] = rate * ss.risk_multiplier
    res["stretch"] = ss
    if write:
        _write(res, fit)
    return res


def _write(res, fit):
    t = res["tract"]
    pct = lambda x: ((x.rank(method="average") - 1) / max(len(x) - 1, 1) * 99 + 1).round()
    out = t.reset_index().rename(columns={"index": "GEOID"})
    out["disengagements_per_million_miles"] = out.per_1k_miles * 1000
    out["dpmm_lo"], out["dpmm_hi"] = out.per_1k_lo * 1000, out.per_1k_hi * 1000
    out["risk_percentile"] = pct(out.risk_multiplier.round(6))
    keep = ["GEOID", "county", "city", "population", "risk_multiplier", "risk_multiplier_lo", "risk_multiplier_hi",
            "disengagements_per_million_miles", "dpmm_lo", "dpmm_hi", "n_sample", "risk_percentile",
            "pct_freeway", "pct_arterial", "pct_local", "per_1k_miles", "per_1k_lo", "per_1k_hi", "road_km",
            "surface_multiplier", "freeway_multiplier", "top_feature", "top_feature_term"] + \
           [c for c in out.columns if c.startswith("term:")]
    out[keep].to_csv(os.path.join(OUT, "tract_scores_disengagement.csv"), index=False)
    areas = []
    for lvl in ("county", "city"):
        a = res[lvl].copy()
        a.index.name = "name"
        a = a.reset_index()
        a.insert(0, "level", lvl)
        if lvl == "city":
            a["county"], a["name"] = a.name.str.split("|").str[0], a.name.str.split("|").str[1]
        else:
            a["county"] = a.name
        areas.append(a)
    pd.concat(areas, ignore_index=True).to_csv(os.path.join(OUT, "disengagement_areas.csv"), index=False)
    c = res["cell"]
    c[["row", "col", "lon", "lat", "GEOID", "risk_multiplier", "risk_multiplier_lo", "risk_multiplier_hi",
       "per_1k_miles", "road_km", "pct_freeway"] + [x for x in c.columns if x.startswith("term:")]].round(4).to_csv(
        os.path.join(OUT, "disengagement_cells.csv"), index_label="cell")
    s = res["stretch"]
    s[["GEOID", "street", "cls", "km", "lanes", "n_junction", "n_signal", "n_stop", "n_crossing", "n_rail",
       "lon", "lat", "risk_multiplier", "per_1k_miles"]].round(4).to_csv(
        os.path.join(OUT, "disengagement_stretches.csv"), index=False)
    model_json(fit)


MODEL_JSON = os.path.join(OUT, "disengagement_model.json")


def model_json(fit, validation=None):
    """Headline numbers for the dashboard text (outputs/disengagement_model.json); notebook 17 adds
    the validation block after it runs the checks."""
    import json
    R = fit["report_table"]
    mid = _mid_shares(fit)
    old = json.load(open(MODEL_JSON, encoding="utf-8")) if os.path.exists(MODEL_JSON) else {}
    m = {"revision": 3, "reports": int(len(R)), "companies": int(R.company.nunique()),
         "reports_freeway": int(fit["reports_by_class"]["freeway"]), "reports_surface": int(fit["reports_by_class"]["surface"]),
         "reference_rate": round(fit["reference_rate"], 4),
         "rho": round(fit["rho"], 3), "rho_estimate": round(fit["rho_estimate"], 3),
         "rho_set": [round(fit["confidence_set"]["lo"], 3), round(fit["confidence_set"]["hi"], 3)],
         "rho_set_lower_at_grid_edge": bool(fit["confidence_set"]["lo"] <= fit["sim_summary"].index.get_level_values(1).min()),
         "beta": BETA, "beta_range": list(BETA_RANGE),
         "rho_road_length": round(float(fit["exposure"].iloc[0, 1]), 3), "rho_vehicle_miles": round(float(fit["exposure"].iloc[1, 1]), 3),
         "tied_surface": {h: round(float(fit["shares"][h].loc["surface", [s for s in TIED if not s.endswith("(freeway)")]].sum()), 4)
                          for h in ("stated", "imputed")},
         "tied_freeway": {h: round(float(fit["shares"][h].loc["freeway", [situation_of(s) for s in TIED if s.endswith("(freeway)")]].sum()), 4)
                          for h in ("stated", "imputed")},
         "shares_mid": {k: {s: round(float(v), 4) for s, v in mid.loc[k].items()} for k in mid.index},
         # sample size of each specialised model: reports that name its situation (LLM probabilities summed)
         "situation_reports": {k: {s: int(round(float(R.loc[R.cls == k, f"s:{s}"].sum()))) for s in SITUATIONS}
                               for k in ("freeway", "surface")},
         "validation": validation if validation is not None else old.get("validation")}
    json.dump(m, open(MODEL_JSON, "w", encoding="utf-8"), indent=1)
    return m


# ------------------------------------------------------------------ 5. terrain from OpenStreetMap
CELL_DEG = 0.0045            # grid rows are 0.0045 deg of latitude (500 m); columns are 500 m wide too
DRIVE = {"motorway": 0, "motorway_link": 0, "trunk": 0, "trunk_link": 0,
         "primary": 1, "primary_link": 1, "secondary": 1, "secondary_link": 1,
         "tertiary": 2, "tertiary_link": 2, "unclassified": 2,
         "residential": 3, "living_street": 3}
POINT_KINDS = ["signal", "stop", "crossing", "rail"]   # highway=traffic_signals|stop|crossing, railway=level_crossing
POI_AMENITY = {"restaurant", "cafe", "fast_food", "bar", "pub", "bank", "pharmacy", "marketplace", "post_office",
               "ice_cream", "food_court", "clinic", "doctors", "dentist", "school", "college", "university",
               "kindergarten", "place_of_worship", "theatre", "cinema", "community_centre", "library"}


AADT_URL = "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/CHhighway/Traffic_AADT/MapServer/0/query"
AADT_CSV = os.path.join(CACHE, "external", "caltrans_aadt", "aadt.csv")


def caltrans_aadt(refresh=False):
    """
    Caltrans Traffic Census: annual average daily traffic at ~14,000 count points on the state highway
    system (latest count year in the layer), from the Caltrans GIS REST service, cached. One row per point:
    lon, lat, route, aadt (mean of the back and ahead counts where both exist).
    """
    import json
    import urllib.request
    if os.path.exists(AADT_CSV) and not refresh:
        return pd.read_csv(AADT_CSV)
    os.makedirs(os.path.dirname(AADT_CSV), exist_ok=True)
    rows, off = [], 0
    while True:
        q = (f"{AADT_URL}?where=1%3D1&outFields=RTE,RTE_SFX,CNTY,PM,DESCRIPTION,BACK_AADT,AHEAD_AADT"
             f"&outSR=4326&resultOffset={off}&resultRecordCount=2000&orderByFields=OBJECTID&f=json")
        d = json.load(urllib.request.urlopen(q, timeout=120))
        feats = d.get("features", [])
        for f in feats:
            a, g = f["attributes"], f.get("geometry") or {}
            rows.append({"lon": g.get("x"), "lat": g.get("y"), "route": a.get("RTE"), "county": a.get("CNTY"),
                         "pm": a.get("PM"), "desc": a.get("DESCRIPTION"),
                         "back_aadt": a.get("BACK_AADT"), "ahead_aadt": a.get("AHEAD_AADT")})
        if len(feats) < 2000 and not d.get("exceededTransferLimit"):
            break
        off += len(feats)
    a = pd.DataFrame(rows).dropna(subset=["lon", "lat"])
    for c in ("back_aadt", "ahead_aadt"):
        a[c] = pd.to_numeric(a[c].astype(str).str.replace(",", ""), errors="coerce")
    a["aadt"] = a[["back_aadt", "ahead_aadt"]].replace(0, np.nan).mean(axis=1)
    a = a.dropna(subset=["aadt"])
    a.to_csv(AADT_CSV, index=False)
    return a


def cell_of(lon, lat):
    """(row, col) of the ~500 m grid cell containing each point."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    row = np.floor(lat / CELL_DEG).astype(np.int64)
    width = CELL_DEG / np.cos(np.radians((row + 0.5) * CELL_DEG))
    col = np.floor(lon / width).astype(np.int64)
    return row, col


def cell_polygon(row, col):
    """[[lon, lat], ...] ring of a grid cell."""
    lat0, lat1 = row * CELL_DEG, (row + 1) * CELL_DEG
    w = CELL_DEG / np.cos(np.radians((row + 0.5) * CELL_DEG))
    lon0, lon1 = col * w, (col + 1) * w
    return [[round(lon0, 5), round(lat0, 5)], [round(lon1, 5), round(lat0, 5)],
            [round(lon1, 5), round(lat1, 5)], [round(lon0, 5), round(lat1, 5)], [round(lon0, 5), round(lat0, 5)]]


def _parse_ways():
    """
    One pass over the California PBF. Returns
      ways:  one row per drivable way (class, ramp, bike lane, lanes, name code)
      refs:  every node reference of every drivable way (node id, lon, lat, way index, endpoint flag)
      tagged: node ids of signals, stops, crossings and rail level crossings
      names: the street-name codebook
    """
    import osmium
    from avgap_osm import PBF
    tag_ids = [array("q") for _ in POINT_KINDS]
    poi_x, poi_y = array("d"), array("d")       # shops, food and drink, services: curbside activity
    w_cls, w_link, w_bike, w_lanes, w_name = array("b"), array("b"), array("b"), array("b"), array("i")
    r_id, r_x, r_y, r_w, r_end = array("q"), array("d"), array("d"), array("i"), array("b")
    names = {}
    fp = (osmium.FileProcessor(PBF)
          .with_locations()
          .with_filter(osmium.filter.KeyFilter("highway", "railway", "shop", "amenity")))
    for o in fp:
        if o.is_node() and ("shop" in o.tags or o.tags.get("amenity") in POI_AMENITY):
            if o.location.valid():
                poi_x.append(o.location.lon); poi_y.append(o.location.lat)
        if o.is_node():
            h, r = o.tags.get("highway"), o.tags.get("railway")
            if h == "traffic_signals":
                tag_ids[0].append(o.id)
            elif h == "stop":
                tag_ids[1].append(o.id)
            elif h == "crossing" or o.tags.get("crossing") not in (None, "no"):
                tag_ids[2].append(o.id)
            if r in ("level_crossing", "crossing"):
                tag_ids[3].append(o.id)
            continue
        if not o.is_way():
            continue
        h = o.tags.get("highway")
        c = DRIVE.get(h)
        if c is None or o.tags.get("access") in ("private", "no") or o.tags.get("area") == "yes":
            continue
        try:
            nodes = [(n.ref, n.lon, n.lat) for n in o.nodes]
        except osmium.InvalidLocationError:
            continue
        if len(nodes) < 2:
            continue
        wi = len(w_cls)
        w_cls.append(c)
        w_link.append(1 if h.endswith("_link") else 0)
        w_bike.append(1 if any(o.tags.get(k, "no") in ("lane", "track", "shared_lane")
                               for k in ("cycleway", "cycleway:both", "cycleway:right", "cycleway:left")) else 0)
        try:
            w_lanes.append(min(int(str(o.tags.get("lanes", "0")).split(";")[0]), 12))
        except ValueError:
            w_lanes.append(0)
        nm = o.tags.get("name") or o.tags.get("ref") or ""
        w_name.append(names.setdefault(nm, len(names)))
        last = len(nodes) - 1
        for k, (ref, x, y) in enumerate(nodes):
            r_id.append(ref); r_x.append(x); r_y.append(y); r_w.append(wi)
            r_end.append(1 if k in (0, last) else 0)
    ways = pd.DataFrame({"cls": np.frombuffer(w_cls, np.int8), "link": np.frombuffer(w_link, np.int8),
                         "bike": np.frombuffer(w_bike, np.int8), "lanes": np.frombuffer(w_lanes, np.int8),
                         "name": np.frombuffer(w_name, np.int32)})
    refs = pd.DataFrame({"id": np.frombuffer(r_id, np.int64), "x": np.frombuffer(r_x), "y": np.frombuffer(r_y),
                         "way": np.frombuffer(r_w, np.int32), "end": np.frombuffer(r_end, np.int8)})
    tagged = {k: np.unique(np.frombuffer(a, np.int64)) for k, a in zip(POINT_KINDS, tag_ids)}
    codebook = pd.Series(list(names), index=list(names.values()))
    pois = pd.DataFrame({"x": np.frombuffer(poi_x), "y": np.frombuffer(poi_y)})
    return ways, refs, tagged, codebook, pois



def _hav_km_vec(x1, y1, x2, y2):
    p = np.pi / 180
    a = (np.sin((y2 - y1) * p / 2) ** 2 + np.cos(y1 * p) * np.cos(y2 * p) * np.sin((x2 - x1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def terrain(refresh=False):
    """
    Road terrain per grid cell, per census tract and per named road stretch (a street's length inside
    one cell), cached in av_data/disengage/. Columns (fw = freeway-type, sf = surface):
      km_fw, km_sf, km_ramp (motorway/trunk links), km_bike (surface km with a bike lane),
      km_multilane (surface km with 4+ lanes), km_arterial, and point counts on surface roads:
      n_junction (street-graph nodes of degree 3+), n_signal, n_stop, n_crossing, n_rail; n_junction_fw
      counts freeway junctions (ramp merges and splits). Tracts get each segment by its midpoint and
      each node by its location, so tract totals are exact rather than summed from cells.
    Returns (cells, tracts, stretches).
    """
    if all(os.path.exists(f) for f in (TERRAIN_CELLS, TERRAIN_TRACTS, TERRAIN_STRETCH)) and not refresh:
        return (pd.read_csv(TERRAIN_CELLS, dtype={"GEOID": str}), pd.read_csv(TERRAIN_TRACTS, dtype={"GEOID": str}),
                pd.read_csv(TERRAIN_STRETCH, keep_default_na=False, dtype={"GEOID": str}))
    from avgap_osm import TRACT_SHP, assign, read_polygons
    os.makedirs(DIS_DIR, exist_ok=True)
    ways, refs, tagged, names, pois = _parse_ways()
    print(f"parsed {len(ways):,} drivable ways, {len(refs):,} node references", flush=True)
    ids, geoms, _ = read_polygons(TRACT_SHP, "GEOID")
    ids = np.asarray(ids, dtype=object)
    # segments: consecutive references of the same way
    same = refs.way.to_numpy()[1:] == refs.way.to_numpy()[:-1]
    i = np.nonzero(same)[0]
    x1, y1, x2, y2 = refs.x.to_numpy()[i], refs.y.to_numpy()[i], refs.x.to_numpy()[i + 1], refs.y.to_numpy()[i + 1]
    seg = pd.DataFrame({"way": refs.way.to_numpy()[i], "km": _hav_km_vec(x1, y1, x2, y2),
                        "x": (x1 + x2) / 2, "y": (y1 + y2) / 2})
    # curvature: the turn (degrees) between consecutive segments of the same way, credited to the later one
    hdg = np.degrees(np.arctan2(y2 - y1, (x2 - x1) * np.cos(np.radians((y1 + y2) / 2))))
    turn = np.abs((np.diff(hdg) + 180) % 360 - 180)
    same_way = seg.way.to_numpy()[1:] == seg.way.to_numpy()[:-1]
    seg["turn_deg"] = np.concatenate([[0.0], np.where(same_way & (seg.km.to_numpy()[1:] > 0.005), turn, 0.0)])
    seg["row"], seg["col"] = cell_of(seg.x, seg.y)
    t = assign(seg.x.to_numpy(), seg.y.to_numpy(), geoms)
    seg["GEOID"] = np.where(t >= 0, ids[np.clip(t, 0, None)], "")
    seg = seg.join(ways, on="way")
    fw = seg.cls == 0
    seg["km_fw"] = np.where(fw, seg.km, 0.0)
    seg["km_sf"] = np.where(~fw, seg.km, 0.0)
    seg["km_ramp"] = np.where(fw & (seg.link == 1), seg.km, 0.0)
    seg["km_bike"] = np.where(~fw & (seg.bike == 1), seg.km, 0.0)
    seg["km_multilane"] = np.where(~fw & (seg.lanes >= 4), seg.km, 0.0)
    seg["km_arterial"] = np.where(seg.cls == 1, seg.km, 0.0)
    seg["turn_fw"] = np.where(fw, seg.turn_deg, 0.0)
    seg["turn_sf"] = np.where(~fw, seg.turn_deg, 0.0)
    # traffic: Caltrans AADT of the nearest count point on the state highway system, x km,
    # i.e. vehicle-km per day on the segment; segments off the state system carry none
    seg["vkm_fw"], seg["vkm_sf"], seg["km_counted_fw"], seg["km_counted_sf"] = 0.0, 0.0, 0.0, 0.0
    try:
        from scipy.spatial import cKDTree
        a = caltrans_aadt()
        kx = np.cos(np.radians(37.0))
        tree = cKDTree(np.c_[a.lon.to_numpy() * kx, a.lat.to_numpy()])
        cand = (seg.cls <= 1).to_numpy()                    # freeways and arterials (state routes)
        dist, j = tree.query(np.c_[seg.x.to_numpy()[cand] * kx, seg.y.to_numpy()[cand]], k=1)
        # count points sit at postmile breaks, often km apart: a freeway segment takes the nearest one within
        # 1.5 km; an arterial only within 300 m, so it never borrows a crossing freeway's count
        ok = dist < np.where(seg.cls.to_numpy()[cand] == 0, 1500, 300) / 111_000
        aadt = np.where(ok, a.aadt.to_numpy()[np.minimum(j, len(a) - 1)], 0.0)
        idx = np.nonzero(cand)[0]
        fwc = (seg.cls.to_numpy()[idx] == 0)
        km_c = seg.km.to_numpy()[idx]
        seg.loc[seg.index[idx[fwc]], "vkm_fw"] = aadt[fwc] * km_c[fwc]
        seg.loc[seg.index[idx[~fwc]], "vkm_sf"] = aadt[~fwc] * km_c[~fwc]
        seg.loc[seg.index[idx[fwc]], "km_counted_fw"] = np.where(ok[fwc], km_c[fwc], 0.0)
        seg.loc[seg.index[idx[~fwc]], "km_counted_sf"] = np.where(ok[~fwc], km_c[~fwc], 0.0)
        print(f"AADT matched to {ok.mean():.0%} of freeway/arterial segment km candidates", flush=True)
    except Exception as e:                                   # the network is optional
        print("AADT skipped:", e, flush=True)
    kcols = ["km_fw", "km_sf", "km_ramp", "km_bike", "km_multilane", "km_arterial", "turn_fw", "turn_sf",
             "vkm_fw", "vkm_sf", "km_counted_fw", "km_counted_sf"]

    # nodes: degree in the street graph (an interior reference adds 2 edges, an endpoint 1)
    deg = refs.assign(d=np.where(refs.end == 1, 1, 2)).groupby("id").agg(
        d=("d", "sum"), x=("x", "first"), y=("y", "first"))
    deg["cls"] = refs.join(ways.cls, on="way").groupby("id").cls.min()    # highest road class at the node
    deg["row"], deg["col"] = cell_of(deg.x, deg.y)
    t = assign(deg.x.to_numpy(), deg.y.to_numpy(), geoms)
    deg["GEOID"] = np.where(t >= 0, ids[np.clip(t, 0, None)], "")
    deg["junction"] = deg.d >= 3
    for k in POINT_KINDS:
        deg[k] = np.isin(deg.index.to_numpy(), tagged[k])
    surf = deg.cls > 0
    pts = pd.DataFrame({"row": deg.row, "col": deg.col, "GEOID": deg.GEOID,
                        "n_junction": (deg.junction & surf).astype(int),
                        "n_junction_fw": (deg.junction & ~surf).astype(int),
                        **{f"n_{k}": (deg[k] & surf).astype(int) for k in POINT_KINDS}})
    pr, pc_ = cell_of(pois.x, pois.y)
    t = assign(pois.x.to_numpy(), pois.y.to_numpy(), geoms)
    pts = pd.concat([pts, pd.DataFrame({"row": pr, "col": pc_, "GEOID": np.where(t >= 0, ids[np.clip(t, 0, None)], ""),
                                        "n_poi": 1})], ignore_index=True).fillna({"n_poi": 0})
    for c in pts.columns:
        if c.startswith("n_"):
            pts[c] = pts[c].fillna(0).astype(int)
    pcols = [c for c in pts.columns if c.startswith("n_")]

    cells = (seg.groupby(["row", "col"])[kcols].sum()
             .join(pts.groupby(["row", "col"])[pcols].sum(), how="outer").fillna(0).reset_index())
    lon_c, lat_c = _cell_centre(cells.row.to_numpy(), cells.col.to_numpy())
    cells["lon"], cells["lat"] = lon_c.round(5), lat_c.round(5)
    t = assign(lon_c, lat_c, geoms)
    cells["GEOID"] = np.where(t >= 0, ids[np.clip(t, 0, None)], "")
    cells.to_csv(TERRAIN_CELLS, index=False)
    tracts = (seg[seg.GEOID != ""].groupby("GEOID")[kcols].sum()
              .join(pts[pts.GEOID != ""].groupby("GEOID")[pcols].sum(), how="outer").fillna(0).reset_index())
    tracts.to_csv(TERRAIN_TRACTS, index=False)

    # stretches: (cell, street name, road class) with the point features on that street's nodes
    named = seg[seg.name.map(names) != ""]
    named = named.assign(vkm=named.vkm_fw + named.vkm_sf, km_counted=named.km_counted_fw + named.km_counted_sf)
    st = named.assign(xk=named.x * named.km, yk=named.y * named.km).groupby(["row", "col", "name", "cls"]).agg(
        km=("km", "sum"), xk=("xk", "sum"), yk=("yk", "sum"), lanes=("lanes", "max"), turn_deg=("turn_deg", "sum"),
        vkm=("vkm", "sum"), km_counted=("km_counted", "sum"),
        GEOID=("GEOID", "first")).reset_index()
    st["lon"], st["lat"] = (st.xk / st.km).round(5), (st.yk / st.km).round(5)
    r2 = refs.join(ways[["name", "cls"]], on="way")
    r2 = r2[r2.name.map(names) != ""]
    r2["row"], r2["col"] = cell_of(r2.x, r2.y)
    r2 = r2.drop_duplicates(["id", "name"]).join(deg[["junction"] + POINT_KINDS], on="id")
    agg = r2.groupby(["row", "col", "name", "cls"])[["junction"] + POINT_KINDS].sum()
    agg.columns = ["n_junction"] + [f"n_{k}" for k in POINT_KINDS]
    st = st.join(agg, on=["row", "col", "name", "cls"]).fillna({c: 0 for c in agg.columns})
    st["street"] = st.name.map(names)
    st = st.drop(columns=["name", "xk", "yk"])
    st.to_csv(TERRAIN_STRETCH, index=False)
    print(f"wrote {len(cells):,} cells, {len(tracts):,} tracts and {len(st):,} stretches to {DIS_DIR}")
    return cells, tracts, st


def _cell_centre(row, col):
    lat = (row + 0.5) * CELL_DEG
    w = CELL_DEG / np.cos(np.radians(lat))
    return (col + 0.5) * w, lat


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "terrain":
        terrain(refresh=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit_all(refresh=True)
        score_all()
