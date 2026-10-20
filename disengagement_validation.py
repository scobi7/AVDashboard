"""
Leave-one-company-out check of the disengagement road-type model (disengagement_risk.py).

    python disengagement_validation.py

Road-type schemes (what a DMV road-type box means, and which OSM road length is its exposure):

  shipped     Freeway, Interstate -> freeway (OSM motorway/trunk); Highway -> arterial (OSM
              primary/secondary); Street, Rural Road, Parking Facility -> local (the rest)
  two-class   the DMV form has no arterial box: Street covers every surface road. Freeway, Interstate,
              Highway -> freeway-type (OSM motorway/trunk, which includes California's state-highway
              expressways); Street, Rural Road, Parking Facility -> surface (OSM primary and below)

Designs for the multiplier, each fitted without the held-out company:

  pooled      the shipped design: share of all reports on a road type / its share of CA road length
  NB FE       a negative-binomial GLM (the family of the crash model, notebooks 06 and 14) with a fixed
              effect per company: n[company, type] ~ company + type, offset log(road-length share),
              fitted on the road types each company reports on (5+ reports). A company that drives one
              road type tells it nothing, and no company's reporting level leaks into the multipliers.

Tests per held-out company (the 15 with 200+ reports), each against road length alone:

  all types        the original test: how the company's reports split across road types
                   (log-likelihood per report). Companies that drive one road type dominate it, and no
                   road-type model can predict where a company chooses to drive.
  types it drives  the same, restricted to the road types the company reports on (20+ reports) and
                   renormalised: given where a company drives, does the model predict how its reports
                   split? Single-type companies drop out (every model predicts them exactly).

Plus the direction check the handoff asks for: within each company, each road type's rate relative to
the company's own average (exposure: road length).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import avgap_data as A
from avgap_osm import tract_roads

SCHEMES = {
    "shipped": {
        "dmv": {"Freeway": "freeway", "Interstate": "freeway", "Highway": "arterial", "Street": "local",
                "Rural Road": "local", "Parking Facility": "local"},
        "osm": {"freeway": ["km_freeway"], "arterial": ["km_arterial"], "local": ["km_collector", "km_local"]}},
    "two-class": {
        "dmv": {"Freeway": "freeway", "Interstate": "freeway", "Highway": "freeway", "Street": "surface",
                "Rural Road": "surface", "Parking Facility": "surface"},
        "osm": {"freeway": ["km_freeway"], "surface": ["km_arterial", "km_collector", "km_local"]}},
}


def road_km(roads, scheme):
    return pd.DataFrame({c: roads[cols].sum(axis=1) for c, cols in SCHEMES[scheme]["osm"].items()})


def counts(dis, scheme):
    rmap = SCHEMES[scheme]["dmv"]
    classes = list(SCHEMES[scheme]["osm"])
    d = dis[dis.road_type.isin(rmap)].assign(cls=lambda x: x.road_type.map(rmap))
    return (d.pivot_table(index="company", columns="cls", values="vin", aggfunc="size", fill_value=0)
             .reindex(columns=classes, fill_value=0))


def pooled(N, s):
    """log multiplier per class, the shipped design"""
    with np.errstate(divide="ignore"):
        return np.log(N.sum(axis=0) / N.values.sum() / s)


def fe_glm(N, s, min_n=5, return_model=False):
    """
    log multiplier per class from the NB GLM with company fixed effects; the dispersion alpha is
    chosen by profile likelihood. Multipliers are scaled so that their road-length-weighted mean is 1,
    as in the pooled design.
    """
    classes = list(N.columns)
    ref = classes[-1]
    long = N.rename_axis(index="company", columns="cls").stack().rename("n").reset_index()
    long = long[long.n >= min_n]
    multi = long.groupby("company").cls.nunique()
    long = long[long.company.isin(multi[multi >= 2].index)].copy()
    long["off"] = np.log(long.cls.map(s))
    form = f"n ~ C(company) + C(cls, Treatment('{ref}'))"
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in np.geomspace(0.01, 20, 25):
            m = smf.glm(form, long, offset=long.off, family=sm.families.NegativeBinomial(alpha=a)).fit()
            if best is None or m.llf > best.llf:
                best = m
    b = pd.Series({c: 0.0 if c == ref else best.params.get(f"C(cls, Treatment('{ref}'))[T.{c}]", -np.inf)
                   for c in classes})
    b = b - np.log((s * np.exp(b)).sum())
    return (b, best, long) if return_model else b


DESIGNS = {"pooled": pooled, "NB FE": fe_glm}


def predict(logm, s, keep=None):
    p = s * np.exp(logm)
    if keep is not None:
        p = p[keep]
    return p / p.sum()


def loglik(n, p):
    return float((n * np.log(np.clip(p, 1e-12, None))).sum())


def loco(N, s, min_rows=200, min_type=20):
    rows = []
    for h in N[N.sum(axis=1) >= min_rows].sum(axis=1).sort_values(ascending=False).index:
        tr, n = N.drop(index=h), N.loc[h]
        keep = n.index[n >= min_type]
        r = {"company": h, "reports": int(n.sum()),
             "mix %": "/".join(f"{100 * v:.0f}" for v in n / n.sum()), "types driven": len(keep)}
        for name, fn in DESIGNS.items():
            lm = fn(tr, s)
            r[f"{name}: all types"] = (loglik(n, predict(lm, s)) - loglik(n, s)) / n.sum()
            if len(keep) >= 2:
                nk = n[keep]
                r[f"{name}: types driven"] = (loglik(nk, predict(lm, s, keep)) - loglik(nk, s[keep] / s[keep].sum())) / nk.sum()
        rows.append(r)
    return pd.DataFrame(rows).set_index("company")


def score(R):
    """per design and test: companies where it beats road length, and the mean gain (nats per report)"""
    out = {}
    for c in [c for c in R.columns if ":" in c]:
        v = R[c].dropna()
        out[c] = {"beats road length": f"{int((v > 0).sum())}/{len(v)}", "mean gain": round(v.mean(), 3)}
    return pd.DataFrame(out).T


def directions(N, s, min_type=20):
    out = []
    for c, n in N.iterrows():
        ok = n >= min_type
        if ok.sum() < 2:
            continue
        rel = (n[ok] / n[ok].sum()) / s[ok]
        rel = rel / ((s[ok] / s[ok].sum()) * rel).sum()      # 1 = the company's own average
        out.append({"company": c, "reports": int(n.sum()), **rel.reindex(N.columns).to_dict()})
    return pd.DataFrame(out).set_index("company").sort_values("reports", ascending=False)


def main():
    pd.set_option("display.width", 220)
    dis = A.load_disengagements()
    roads = tract_roads()
    for scheme in SCHEMES:
        km = road_km(roads, scheme).sum()
        s = km / km.sum()
        N = counts(dis, scheme)
        print(f"\n=== {scheme}: {len(N)} companies, {int(N.values.sum()):,} reports | road-length share "
              + ", ".join(f"{k} {v:.3f}" for k, v in s.items()) + " ===")
        for name, fn in DESIGNS.items():
            print(f"  multipliers, all companies, {name:7s}", np.exp(fn(N, s)).round(2).to_dict())
        R = loco(N, s)
        print(R.round(3).to_string())
        print(score(R).to_string())
        W = directions(N, s)
        print("\n  within-company rate relative to the company's own average (road-length exposure)")
        print(W.round(2).to_string())
        for k in N.columns:
            v = W[k].dropna()
            print(f"  {k:8s} above the company's own average in {int((v > 1).sum())} of {len(v)} companies")


if __name__ == "__main__":
    main()
