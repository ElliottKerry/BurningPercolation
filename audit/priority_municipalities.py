"""Natural experiment: did the 2008 priority-municipalities list ("lista de municipios prioritarios", Decreto
6.321/2007, Portaria MMA 28 of 24 Jan 2008) change fire in the listed municipalities?

Treated: the 36 municipalities of Portaria MMA 28/2008 (verbatim list from LegisWeb, id=205491).
Excluded from the controls (listed later, so not clean controls): the 2009 additions (Portaria MMA 102/2009:
Amarante do Maranhao, Mucajai, Pacaja, Maraba, Itupiranga, Tailandia, Feliz Natal), Boca do Acre (on the list
by 2017), and the 2017 additions (Apui, Manicore, Novo Aripuana, Itaituba, Portel, Buritis, Candeias do Jamari,
Cujubim). Any other later additions we could not identify stay in the controls, which biases effects toward zero.
Controls: every other Legal Amazon municipality (9 states; Maranhao west of 44 W by municipality centroid).
Similar controls: controls whose 2003-2007 mean fire count is at least the 10th percentile of the treated.

Outcomes per municipality-year (MODIS Aqua fire events, 2 km / 2 days, start year):
  log(1 + events)
  median start day of year of the events (municipality-years with >= MIN_TIMING events)
Event study: y_it = sum_k b_k [treated_i x year_k] (k != 2007) + municipality FE + state-by-year FE,
SE clustered by municipality. b_k = treated minus control difference in year k, relative to 2007.
Pooled: one coefficient for 2009-2015 (2008, the first year, reported in the event study only).

    python audit/priority_municipalities.py
"""
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES, YEARS
from audit.percolation_stability import events_with_dates
from firms import config

LA_STATES = ["AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"]
LIST_2008 = {
    "AM": ["Lábrea"],
    "MT": ["Alta Floresta", "Aripuanã", "Brasnorte", "Colniza", "Confresa", "Cotriguaçu", "Gaúcha do Norte", "Juara",
           "Juína", "Marcelândia", "Nova Bandeirantes", "Nova Maringá", "Nova Ubiratã", "Paranaíta",
           "Peixoto de Azevedo", "Porto dos Gaúchos", "Querência", "São Félix do Araguaia", "Vila Rica"],
    "PA": ["Altamira", "Brasil Novo", "Cumaru do Norte", "Dom Eliseu", "Novo Progresso", "Novo Repartimento",
           "Paragominas", "Rondon do Pará", "Santa Maria das Barreiras", "Santana do Araguaia", "São Félix do Xingu",
           "Ulianópolis"],
    "RO": ["Nova Mamoré", "Porto Velho", "Machadinho D'Oeste", "Pimenta Bueno"],
}
LATER = {
    "MA": ["Amarante do Maranhão"], "RR": ["Mucajaí"], "MT": ["Feliz Natal"],
    "PA": ["Pacajá", "Marabá", "Itupiranga", "Tailândia", "Itaituba", "Portel"],
    "AM": ["Boca do Acre", "Apuí", "Manicoré", "Novo Aripuanã"],
    "RO": ["Buritis", "Candeias do Jamari", "Cujubim"],
}
BASE = 2007
POOL = range(2009, 2016)
MIN_TIMING = 20


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum())


def municipalities():
    import geopandas as gpd
    m = gpd.read_file(config.RAW_DIR / "boundaries" / "BR_Municipios_2025.shp")
    m = m[m.SIGLA_UF.isin(LA_STATES)].to_crs("EPSG:4674")
    c = m.geometry.representative_point()
    m = m[(m.SIGLA_UF != "MA") | (c.x < -44)].copy()
    key = m.SIGLA_UF + "|" + m.NM_MUN.map(norm)
    t08 = {f"{uf}|{norm(n)}" for uf, ns in LIST_2008.items() for n in ns}
    later = {f"{uf}|{norm(n)}" for uf, ns in LATER.items() for n in ns}
    missing = (t08 | later) - set(key)
    if missing:
        raise SystemExit(f"names not matched: {missing}")
    m["group"] = np.where(key.isin(t08), "listed_2008", np.where(key.isin(later), "listed_later", "control"))
    return m[["CD_MUN", "NM_MUN", "SIGLA_UF", "AREA_KM2", "group", "geometry"]]


def panel(m):
    import geopandas as gpd
    rows = []
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        g = gpd.GeoDataFrame({"doy": ev.start.dt.dayofyear.to_numpy()},
                             geometry=gpd.points_from_xy(ev.lon, ev.lat), crs="EPSG:4674")
        j = gpd.sjoin(g, m[["CD_MUN", "geometry"]], predicate="within")
        a = j.groupby("CD_MUN").agg(events=("doy", "size"), median_doy=("doy", "median"))
        a["year"] = y
        rows.append(a.reset_index())
        print(y, len(j), flush=True)
    p = pd.concat(rows)
    full = pd.MultiIndex.from_product([m.CD_MUN, YEARS], names=["CD_MUN", "year"]).to_frame(index=False)
    p = full.merge(p, on=["CD_MUN", "year"], how="left").fillna({"events": 0})
    p = p.merge(pd.DataFrame(m.drop(columns="geometry")), on="CD_MUN")
    p.loc[p.events < MIN_TIMING, "median_doy"] = np.nan
    return p


def demean(df, cols, fes, tol=1e-10, it=500):
    """Remove several fixed effects by alternating projections."""
    x = df[cols].to_numpy(float).copy()
    codes = [pd.factorize(df[f])[0] for f in fes]
    for _ in range(it):
        old = x.copy()
        for c in codes:
            cnt = np.bincount(c)
            for j in range(x.shape[1]):
                x[:, j] -= (np.bincount(c, weights=x[:, j]) / cnt)[c]
        if np.abs(x - old).max() < tol:
            break
    return x


def fe_ols(df, y, xs, fes=("CD_MUN", "st_yr"), cluster="CD_MUN"):
    """OLS with absorbed fixed effects; SE clustered (CR1, as in Stata reghdfe with FE nested in clusters)."""
    z = demean(df, [y, *xs], list(fes))
    Y, X = z[:, 0], z[:, 1:]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    e = Y - X @ b
    g = pd.factorize(df[cluster])[0]
    G, n, k = g.max() + 1, len(Y), X.shape[1]
    S = np.zeros((G, k))
    np.add.at(S, g, X * e[:, None])
    V = XtX_inv @ (S.T @ S) @ XtX_inv * G / (G - 1) * (n - 1) / (n - k)
    return pd.Series(b, xs), pd.DataFrame(V, xs, xs)


def wald_p(b, V, names):
    from scipy import stats
    bb, VV = b[names].to_numpy(), V.loc[names, names].to_numpy()
    w = float(bb @ np.linalg.pinv(VV) @ bb)
    return float(stats.chi2.sf(w, len(names)))


def event_study(p, y):
    from scipy import stats
    d = p.dropna(subset=[y]).copy()
    d["treated"] = (d.group == "listed_2008").astype(int)
    ks = [k for k in YEARS if k != BASE]
    for k in ks:
        d[f"t{k}"] = d.treated * (d.year == k)
    d["st_yr"] = d.SIGLA_UF + d.year.astype(str)
    names = [f"t{k}" for k in ks]
    b, V = fe_ols(d, y, names)
    dp = d[(d.year <= BASE) | d.year.isin(POOL)].copy()
    dp["post"] = dp.treated * dp.year.isin(POOL)
    bp, Vp = fe_ols(dp, y, ["post"])
    se = float(np.sqrt(Vp.loc["post", "post"]))
    coefs = pd.DataFrame({"year": ks, "b": b.to_numpy(), "se": np.sqrt(np.diag(V))})
    coefs = pd.concat([coefs, pd.DataFrame({"year": [BASE], "b": [0.0], "se": [0.0]})]).sort_values("year")
    return coefs, {"pooled_2009_2015": float(bp.post), "pooled_se": se,
                   "pooled_p": float(2 * stats.norm.sf(abs(bp.post) / se)),
                   "pre_trend_joint_p": wald_p(b, V, [f"t{k}" for k in YEARS if k < BASE]),
                   "n_municipalities": int(d.CD_MUN.nunique()), "n_treated": int(d[d.treated == 1].CD_MUN.nunique())}


def main():
    import matplotlib.pyplot as plt
    m = municipalities()
    cache = OUT / "priority_municipalities_panel.csv"
    p = pd.read_csv(cache) if cache.exists() else panel(m)
    p.to_csv(cache, index=False)
    p["log_events"] = np.log1p(p.events)
    pre = p[p.year <= BASE].groupby("CD_MUN").events.mean()
    floor = pre[p.drop_duplicates("CD_MUN").set_index("CD_MUN").group.eq("listed_2008")].quantile(0.1)
    similar = set(pre[pre >= floor].index)
    base = p[p.group != "listed_later"]
    samples = {"all controls": base,
               "similar controls": base[(base.group == "listed_2008") | base.CD_MUN.isin(similar)]}
    res, curves = {"similar_floor_mean_events_2003_2007": float(floor)}, {}
    share = p.groupby(["year", "group"]).events.sum().unstack()
    share = share.div(share.sum(1), axis=0)
    res["share_of_legal_amazon_events_listed_2008"] = share.listed_2008.round(3).to_dict()
    for sname, s in samples.items():
        for y in ("log_events", "median_doy"):
            coefs, r = event_study(s, y)
            r["n_controls"] = r["n_municipalities"] - r["n_treated"]
            res[f"{sname} | {y}"] = r
            curves[(sname, y)] = coefs
    (OUT / "priority_municipalities.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.9))
    raw = p[p.group != "listed_later"].groupby(["year", "group"]).events.sum().unstack()
    a = axs[0]
    for g, col, lab in (("listed_2008", SERIES[1], "36 municipalities listed in 2008"),
                        ("control", SERIES[0], "Never-listed Legal Amazon municipalities")):
        a.plot(raw.index, raw[g] / raw.loc[2003:BASE, g].mean(), color=col, lw=2, marker="o", ms=3.5, label=lab)
    a.axvline(2008, color=INK2, lw=0.8, ls="--")
    a.axhline(1, color=MUTED, lw=0.6)
    a.set_ylabel("Fire events, 2003–2007 mean = 1", color=INK2)
    a.set_title("Fire events over time (indexed)", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8.5)
    for a, (y, lab, scale) in zip(axs[1:], (("log_events", "Fire events: treated − control (%)", 100),
                                            ("median_doy", "Median fire start day: treated − control (days)", 1))):
        for (sname, yy), cc in curves.items():
            if yy != y:
                continue
            col = SERIES[0] if sname == "all controls" else SERIES[1]
            off = -0.15 if sname == "all controls" else 0.15
            b = cc.b * scale if y == "median_doy" else (np.exp(cc.b) - 1) * 100
            lo = (cc.b - 1.96 * cc.se) * scale if y == "median_doy" else (np.exp(cc.b - 1.96 * cc.se) - 1) * 100
            hi = (cc.b + 1.96 * cc.se) * scale if y == "median_doy" else (np.exp(cc.b + 1.96 * cc.se) - 1) * 100
            a.errorbar(cc.year + off, b, yerr=[b - lo, hi - b], fmt="o", ms=4, color=col, ecolor=col, lw=1,
                       capsize=2, label=f"{sname} (pooled 2009–15 p {res[f'{sname} | {y}']['pooled_p']:.3f}, "
                                        f"pre-trend p {res[f'{sname} | {y}']['pre_trend_joint_p']:.2f})")
        a.axhline(0, color=INK2, lw=0.8)
        a.axvline(BASE + 0.5, color=INK2, lw=0.8, ls="--")
        a.set_title(lab + f"\nrelative to {BASE}; state-by-year and municipality effects removed", loc="left",
                    fontsize=10, color=INK)
        a.legend(frameon=False, fontsize=8, loc="lower left")
    for a in axs:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Natural experiment: the 2008 priority-municipalities list (36 municipalities) vs never-listed "
                 "municipalities, MODIS Aqua fire events 2003–2024", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "priority_municipalities.png", dpi=140)


if __name__ == "__main__":
    main()
