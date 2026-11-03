"""State-by-year panel: is fire connectivity linked to deforestation within states, after removing anything that hit
all states in the same year (national weather, El Nino, the satellite) and fixed differences between states?

Units: the 9 Legal Amazon states (AC, AM, AP, MA west of 44 W, MT, PA, RO, RR, TO), 2003-2024.
Per state-year (fire events 2 km / 2 days, MODIS Aqua, started in that calendar year, inside the state):
  events     number of fire events
  ratio      d50 of fire / median d50 of RAND_REPS sets of random points in the state, same number
  delay      join-up day minus median of DELAY_REPS date-shuffled runs, distance 2 x sqrt(state area / events)
             (same rule as percolation_days.py, applied inside each state)
  climate    ERA5 rain, temperature, humidity, wind, soil moisture: fire-weighted mean over the state's cells
             in the state's own fire season (the 4 calendar months with the most fire events 2003-2024)
  deforestation  PRODES yearly rate for the state (Aug of the previous year to Jul), km2
State-years with fewer than MIN_EVENTS events are dropped (too few for a stable curve).

Model (two-way fixed effects):
  outcome_st = b log(deforestation_st) + c log(events_st) + climate_st + state_s + year_t + e_st
Standard errors clustered by state (only 9 clusters, so also):
Permutation test: whole deforestation series swapped between states (PERMS random swaps), so each series keeps its
own shape over time; p = share of swaps with |b| at least as large as observed.

    python audit/percolation_state_panel.py build     # ~15 min, writes percolation_state_panel.csv
    python audit/percolation_state_panel.py model     # fits and figure
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import ALBERS, GRID, INK, INK2, MUTED, OUT, SERIES, YEARS, project
from audit.percolation_climate import CLIMATE
from audit.percolation_days import SCALE, join_day, largest_by_day
from audit.percolation_lattice_threshold import RES
from audit.percolation_legal_amazon import d50, random_in
from audit.percolation_stability import events_with_dates
from firms import config

STATES = ["AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"]
PRODES_ID = {18278: "AC", 18279: "AM", 18282: "AP", 18288: "MA", 18285: "MT", 18281: "PA", 18277: "RO",
             18280: "RR", 18283: "TO"}
SHAPES = config.DATA_DIR / "cache" / "legal_amazon_states_albers.parquet"
PANEL = OUT / "percolation_state_panel.csv"
RAND_REPS = 3
DELAY_REPS = 10
MIN_EVENTS = 300
PERMS = 2000
STATE_COLOURS = dict(zip(STATES, ["#2a78d6", "#e8590c", "#7a7a7a", "#2f9e44", "#6741a0", "#e3388f", "#f08c00",
                                  "#1098ad", "#a0522d"]))


def state_shapes():
    import geopandas as gpd
    from shapely.geometry import box
    if SHAPES.exists():
        return gpd.read_parquet(SHAPES)
    m = gpd.read_file(config.RAW_DIR / "boundaries" / "BR_Municipios_2025.shp")[["SIGLA_UF", "geometry"]]
    m = m[m.SIGLA_UF.isin(STATES)]
    ma = m[m.SIGLA_UF == "MA"].to_crs("EPSG:4674").clip(box(-80, -40, -44, 10)).to_crs(m.crs)
    m = pd.concat([m[m.SIGLA_UF != "MA"], ma]).to_crs(ALBERS)
    s = m.dissolve("SIGLA_UF").reset_index()
    s["geometry"] = s.buffer(0)
    s.to_parquet(SHAPES)
    return s


def prodes_states():
    r = json.loads((config.RAW_DIR / "prodes_rates" / "rates2026.json").read_text())
    rows = [{"year": p["endDate"]["year"], "state": PRODES_ID[f["loiname"]],
             "deforestation_km2": sum(a["area"] for a in f["areas"])}
            for p in r["periods"] for f in p["features"] if f["loiname"] in PRODES_ID]
    return pd.DataFrame(rows)


def state_of(xy_km, shapes):
    import shapely
    out = np.full(len(xy_km), "", dtype=object)
    for st, g in zip(shapes.SIGLA_UF, shapes.geometry):
        shapely.prepare(g)
        out[shapely.contains_xy(g, xy_km[:, 0] * 1000, xy_km[:, 1] * 1000)] = st
    return out


def climate_by_state(shapes, ev_all):
    """Fire-season climate per state-year: 4 top-fire months per state, fire-weighted over the state's cells."""
    from pyproj import Transformer
    ev_all = ev_all[ev_all.state != ""]
    season = {st: sorted(g.start.dt.month.value_counts().index[:4]) for st, g in ev_all.groupby("state")}
    ev_all = ev_all.assign(ix=np.floor(ev_all.lon / RES + 0.5).astype(int), iy=np.floor(ev_all.lat / RES + 0.5).astype(int))
    w = ev_all.groupby(["state", "ix", "iy"]).size().rename("w").reset_index()
    c = pd.read_parquet(config.DATA_DIR / "cache" / "era5_brazil_monthly.parquet")
    cells = c[["ix", "iy"]].drop_duplicates()
    x, y = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True).transform(cells.ix * RES, cells.iy * RES)
    cells["state"] = state_of(np.column_stack([x, y]) / 1000, shapes)
    w = w.merge(cells[cells.state != ""], on=["state", "ix", "iy"])     # weight only cells whose centre is in the state
    c = c.merge(w, on=["ix", "iy"])
    c = c[[m in season[s] for m, s in zip(c.month.dt.month, c.state)]]
    c["year"] = c.month.dt.year
    out = c.groupby(["state", "year"]).apply(
        lambda g: pd.Series({v: np.average(g[v], weights=g.w) for v in CLIMATE}), include_groups=False)
    return out.reset_index(), season


def build():
    from scipy.spatial import cKDTree
    shapes = state_shapes()
    area = dict(zip(shapes.SIGLA_UF, shapes.area / 1e6))
    rng = np.random.default_rng(42)
    rows, every = [], []
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y].copy()
        xy = project(ev.lon.to_numpy(), ev.lat.to_numpy())
        ev["state"] = state_of(xy, shapes)
        every.append(ev[["lon", "lat", "start", "state"]])
        for st in STATES:
            m = (ev.state == st).to_numpy()
            n = int(m.sum())
            r = {"state": st, "year": y, "events": n}
            if n >= MIN_EVENTS:
                p_xy = xy[m]
                g = shapes.geometry[shapes.SIGLA_UF == st].iloc[0]
                fire = d50(p_xy)[0]
                rand = np.median([d50(random_in(g, n, rng))[0] for _ in range(RAND_REPS)])
                doy = (ev.start[m].dt.dayofyear - 1).to_numpy()
                d = SCALE * np.sqrt(area[st] / n)
                pr = cKDTree(p_xy).query_pairs(d, output_type="ndarray")
                real = join_day(largest_by_day(n, pr[:, 0], pr[:, 1], doy), n)
                null = [join_day(largest_by_day(n, pr[:, 0], pr[:, 1], rng.permutation(doy)), n)
                        for _ in range(DELAY_REPS)]
                r.update({"fire_d50_km": fire, "random_d50_km": rand, "ratio": fire / rand, "d_km": d,
                          "join_day": real, "null_day": float(np.nanmedian(null)), "delay": real - np.nanmedian(null)})
            rows.append(r)
        print(y, {r["state"]: (r["events"], round(r.get("delay", np.nan), 1), round(r.get("ratio", np.nan), 2))
                  for r in rows[-len(STATES):]}, flush=True)
    panel = pd.DataFrame(rows)
    clim, season = climate_by_state(shapes, pd.concat(every))
    panel = panel.merge(clim, on=["state", "year"], how="left").merge(prodes_states(), on=["state", "year"], how="left")
    panel.to_csv(PANEL, index=False)
    (OUT / "percolation_state_panel_seasons.json").write_text(json.dumps({k: [int(x) for x in v] for k, v in season.items()}))
    print(panel.groupby("state")[["events", "delay", "ratio", "deforestation_km2"]].mean().round(2))


def fe_fit(df, y, xs, weights=None):
    """Two-way FE OLS (state + year dummies), SE clustered by state."""
    import statsmodels.api as sm
    X = pd.concat([df[xs], pd.get_dummies(df.state, prefix="s", drop_first=True, dtype=float),
                   pd.get_dummies(df.year, prefix="y", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    mod = sm.WLS(df[y], X, weights=weights) if weights is not None else sm.OLS(df[y], X)
    return mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.state)[0]})


def perm_p(df, y, xs, key="log_defor", rng=None):
    """Swap whole deforestation series between states; two-sided p for the coefficient on key."""
    obs = fe_fit(df, y, xs).params[key]
    wide = df.pivot(index="year", columns="state", values=key)
    sts = list(wide.columns)
    hits = 0
    for _ in range(PERMS):
        perm = dict(zip(sts, rng.permutation(sts)))
        d = df.copy()
        d[key] = [wide.at[yr, perm[s]] for yr, s in zip(d.year, d.state)]
        d = d.dropna(subset=[key])
        hits += abs(fe_fit(d, y, xs).params[key]) >= abs(obs)
    return (hits + 1) / (PERMS + 1)


def model():
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    rng = np.random.default_rng(7)
    p = pd.read_csv(PANEL)
    p = p.dropna(subset=["delay", "ratio", "deforestation_km2"]).copy()
    p["log_defor"] = np.log(p.deforestation_km2)
    p["log_events"] = np.log(p.events)
    p["log_ratio"] = np.log(p.ratio)
    for v in CLIMATE:
        p[v + "_z"] = p.groupby("state")[v].transform(lambda s: (s - s.mean()) / s.std())
    clim = [v + "_z" for v in CLIMATE]
    specs = {"deforestation only": ["log_defor"],
             "+ fire count": ["log_defor", "log_events"],
             "+ fire count + climate": ["log_defor", "log_events", *clim]}
    res = {"n_state_years": int(len(p)), "states": sorted(p.state.unique()),
           "state_years_per_state": p.state.value_counts().to_dict()}
    for y in ("delay", "log_ratio"):
        res[y] = {}
        for name, xs in specs.items():
            f = fe_fit(p, y, xs)
            r = {"coef_log_defor": float(f.params.log_defor), "se": float(f.bse.log_defor),
                 "p_cluster": float(f.pvalues.log_defor), "r2": float(f.rsquared)}
            if name == "+ fire count + climate":
                r["climate_coefs"] = {v: [round(float(f.params[v]), 3), round(float(f.pvalues[v]), 3)] for v in clim}
                r["climate_joint_p"] = float(f.f_test(", ".join(f"{v} = 0" for v in clim)).pvalue)
                r["p_permutation"] = perm_p(p, y, xs, rng=rng)
                fw = fe_fit(p, y, xs, weights=p.events)
                r["weighted_by_events"] = {"coef": float(fw.params.log_defor), "p_cluster": float(fw.pvalues.log_defor)}
                big = p[~p.state.isin(["AP", "RR", "TO"])]
                fb = fe_fit(big, y, xs)
                r["without_AP_RR_TO"] = {"coef": float(fb.params.log_defor), "p_cluster": float(fb.pvalues.log_defor),
                                         "n": int(len(big))}
                lead = p.merge(p[["state", "year", "log_defor"]].assign(year=lambda d: d.year - 1)
                               .rename(columns={"log_defor": "log_defor_next"}), on=["state", "year"])
                fl = fe_fit(lead.assign(log_defor=lead.log_defor_next), y, xs)
                r["prodes_year_ending_next_july"] = {"coef": float(fl.params.log_defor),
                                                     "p_cluster": float(fl.pvalues.log_defor), "n": int(len(lead))}
            res[y][name] = r
    (OUT / "percolation_state_panel.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))

    # partial-regression plots: outcome and log deforestation, each with fire count, climate, state and year removed
    xs = ["log_events", *clim]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for a, (y, lab) in zip(axs[:2], (("delay", "Join-up delay (days)"), ("log_ratio", "log clustering ratio (fire / random)"))):
        ry = fe_fit(p, y, xs).resid
        rx = fe_fit(p, "log_defor", xs).resid
        for st in sorted(p.state.unique()):
            m = (p.state == st).to_numpy()
            a.scatter(rx[m], ry[m], s=24, color=STATE_COLOURS[st], label=st, zorder=3)
        b = np.polyfit(rx, ry, 1)
        xx = np.linspace(rx.min(), rx.max(), 50)
        a.plot(xx, np.polyval(b, xx), color=INK, lw=1.5)
        r = res[y]["+ fire count + climate"]
        a.set_title(f"{lab}\nslope {r['coef_log_defor']:+.2f}, p {r['p_cluster']:.3f} (clustered), "
                    f"{r['p_permutation']:.3f} (permutation)", loc="left", fontsize=10, color=INK)
        a.set_xlabel("log deforestation, after removing state, year, fire count, climate", color=INK2)
        a.set_ylabel(lab + ", same removed", color=INK2)
        a.axhline(0, color=MUTED, lw=0.6)
        a.axvline(0, color=MUTED, lw=0.6)
    axs[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    a = axs[2]
    labels, coefs, los, his = [], [], [], []
    for y, ylab in (("delay", "Delay"), ("log_ratio", "log ratio")):
        for name in specs:
            r = res[y][name]
            labels.append(f"{ylab}: {name}")
            coefs.append(r["coef_log_defor"] / (1 if y == "delay" else 0.1))
            los.append((r["coef_log_defor"] - 1.96 * r["se"]) / (1 if y == "delay" else 0.1))
            his.append((r["coef_log_defor"] + 1.96 * r["se"]) / (1 if y == "delay" else 0.1))
    yy = np.arange(len(labels))[::-1]
    a.errorbar(coefs, yy, xerr=[np.subtract(coefs, los), np.subtract(his, coefs)], fmt="o", color=SERIES[0],
               ecolor=SERIES[0], capsize=3)
    a.axvline(0, color=INK2, lw=0.8)
    a.set_yticks(yy, labels, fontsize=8.5)
    a.set_title("Deforestation coefficient across models (95% CI, clustered)\ndelay in days; log ratio × 10",
                loc="left", fontsize=10, color=INK)
    for a in axs:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"Legal Amazon states 2003–2024 ({len(p)} state-years): does more clearing in a state go with different "
                 "fire connectivity, compared with the same state in other years and other states in the same year?",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_state_panel.png", dpi=140)


if __name__ == "__main__":
    {"build": build, "model": model}[sys.argv[1]]()
