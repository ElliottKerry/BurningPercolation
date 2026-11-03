"""Days until fire joins up: percolation in time, one year at a time, 2003-2024.

Fire events (2 km / 2 days, with start dates; cached by percolation_stability.py) are added day by day in the
order they start. Two events are linked when they are within d of each other. Each day we record the share of
the YEAR's events in the largest cluster so far.

Two rules for d:
  fixed   20 km every year
  scaled  SCALE x the typical spacing of that year's events, sqrt(Brazil area / events). SCALE = 2 gives about
          20 km in an average year; it keeps the final joined share similar across years with more or fewer fires.

Join day: first day the largest cluster holds >= X of the year's events, averaged over X = 40%, 50%, 60% so one
lucky merge does not decide the result.
Count-only day: day on which half the year's events have happened (no geography).
Shuffled dates: same events, same places, start dates permuted among them (REPS runs, median). Same daily counts
and the same final map; only the timing of where fire happens is lost.
lag = join day - shuffled join day. > 0: fire joins up later than if the same fires came in random order.

    python audit/percolation_days.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES, YEARS, project
from audit.percolation_stability import events_with_dates
from firms import config

FIXED_KM = 20.0
SCALE = 2.0
LEVELS = (0.4, 0.5, 0.6)
REPS = 10
RULES = ("fixed", "scaled")
RULE_NAME = {"fixed": f"Fixed {FIXED_KM:.0f} km", "scaled": f"Scaled: {SCALE:g} × typical spacing that year"}


def largest_by_day(n, ei, ej, day_of_point, ndays=366):
    """Largest cluster size at the end of each day, adding points and links as they appear."""
    parent = np.arange(n)
    size = np.ones(n, int)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    de = np.maximum(day_of_point[ei], day_of_point[ej])
    order = np.argsort(de, kind="stable")
    ei, ej, de = ei[order], ej[order], de[order]
    any_point = np.bincount(day_of_point, minlength=ndays) > 0
    out = np.zeros(ndays, int)
    big, k = 0, 0
    for d in range(ndays):
        if any_point[d]:
            big = max(big, 1)
        while k < len(de) and de[k] == d:
            a, b = find(ei[k]), find(ej[k])
            if a != b:
                if size[a] < size[b]:
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]
                big = max(big, size[a])
            k += 1
        out[d] = big
    return out


def join_day(curve, n):
    """Mean of the first days the largest cluster reaches each level (NaN if a level is never reached)."""
    days = []
    for lv in LEVELS:
        hit = np.nonzero(curve >= lv * n)[0]
        days.append(hit[0] + 1 if len(hit) else np.nan)
    return float(np.mean(days))


def main():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from audit.fire_percolation_pilot import ALBERS
    area = gpd.read_parquet(config.DATA_DIR / "cache" / "brazil_outline_5880.parquet").to_crs(ALBERS).area.sum() / 1e6
    rng = np.random.default_rng(30)
    rows, curves = [], {}
    for y in YEARS:
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        xy = project(ev.lon.to_numpy(), ev.lat.to_numpy())
        doy = (ev.start.dt.dayofyear - 1).to_numpy()
        n = len(xy)
        tree = cKDTree(xy)
        cum = np.cumsum(np.bincount(doy, minlength=366))
        r = {"year": y, "events": n, "half_count_day": int(np.nonzero(cum >= n / 2)[0][0]) + 1}
        perms = [rng.permutation(doy) for _ in range(REPS)]
        for rule in RULES:
            d = FIXED_KM if rule == "fixed" else SCALE * np.sqrt(area / n)
            p = tree.query_pairs(d, output_type="ndarray")
            real = largest_by_day(n, p[:, 0], p[:, 1], doy)
            nulls = np.array([largest_by_day(n, p[:, 0], p[:, 1], sd) for sd in perms])
            nd = [join_day(c, n) for c in nulls]
            r.update({f"{rule}_d_km": d, f"{rule}_final_share": real[-1] / n, f"{rule}_join_day": join_day(real, n),
                      f"{rule}_null_day": float(np.nanmedian(nd)), f"{rule}_null_lo": float(np.nanmin(nd)),
                      f"{rule}_null_hi": float(np.nanmax(nd))})
            curves[(y, rule)] = (real / n, np.median(nulls, 0) / n, cum / n)
        rows.append(r)
        print(y, {k: round(v, 2) for k, v in r.items() if k != "year"}, flush=True)
    df = pd.DataFrame(rows).set_index("year")
    for rule in RULES:
        df[f"{rule}_lag"] = df[f"{rule}_join_day"] - df[f"{rule}_null_day"]
    df.to_csv(OUT / "percolation_days.csv")
    summ = {}
    for rule in RULES:
        j, nl, lag = df[f"{rule}_join_day"], df[f"{rule}_null_day"], df[f"{rule}_lag"]
        summ[rule] = {"d_km_range": [float(df[f"{rule}_d_km"].min()), float(df[f"{rule}_d_km"].max())],
                      "final_share_range": [float(df[f"{rule}_final_share"].min()), float(df[f"{rule}_final_share"].max())],
                      "join_day_mean": float(j.mean()), "join_day_sd": float(j.std()),
                      "shuffled_day_sd": float(nl.std()), "corr_join_vs_shuffled": float(j.corr(nl)),
                      "share_of_join_variance_matched_by_shuffled": float(j.corr(nl) ** 2),
                      "lag_mean": float(lag.mean()), "lag_sd": float(lag.std()),
                      "years_later_than_all_shuffled": int((j > df[f"{rule}_null_hi"]).sum()),
                      "years_sooner_than_all_shuffled": int((j < df[f"{rule}_null_lo"]).sum()),
                      "spearman_join_vs_events": float(j.corr(df.events, method="spearman")),
                      "spearman_lag_vs_events": float(lag.corr(df.events, method="spearman"))}
    summ["count_only_day_sd"] = float(df.half_count_day.std())
    (OUT / "percolation_days.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))

    starts = pd.date_range("2001-01-01", periods=12, freq="MS")
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 1.4]})
    a = axs[0, 0]
    for y, colour in ((2019, SERIES[0]), (2014, SERIES[1])):
        real, null, cum = curves[(y, "scaled")]
        days = np.arange(1, 367)
        a.plot(days, real, color=colour, lw=2, label=f"{y}: largest cluster")
        a.plot(days, null, color=colour, lw=1.2, ls="--", label=f"{y}: dates shuffled")
        a.plot(days, cum, color=colour, lw=1, ls=":", label=f"{y}: share of fires so far")
    a.axhspan(0.4, 0.6, color=MUTED, alpha=0.12, lw=0)
    a.set_ylabel("Share of the year's fire events", color=INK2)
    a.set_title("Building up through the year (scaled distance; shaded = 40–60%)", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")
    a.set_xticks(starts.dayofyear, [s.strftime("%b")[0] for s in starts])
    a = axs[1, 0]
    for rule, colour in zip(RULES, (SERIES[3], SERIES[0])):
        a.bar(df.index + (-0.2 if rule == "fixed" else 0.2), df[f"{rule}_lag"], width=0.4, color=colour,
              label=RULE_NAME[rule])
    a.axhline(0, color=INK2, lw=0.8)
    a.set_ylabel("Days (actual − shuffled)", color=INK2)
    a.set_title("Timing effect: how much later actual fire joins up than shuffled dates", loc="left", fontsize=10,
                color=INK)
    a.legend(frameon=False, fontsize=8.5)
    for a, rule in zip(axs[:, 1], RULES):
        a.fill_between(df.index, df[f"{rule}_null_lo"], df[f"{rule}_null_hi"], color=MUTED, alpha=0.25, lw=0)
        a.plot(df.index, df[f"{rule}_null_day"], color=MUTED, lw=1.5, ls="--",
               label=f"Dates shuffled ({REPS} runs, band = range)")
        a.plot(df.index, df[f"{rule}_join_day"], color=SERIES[0], lw=2, marker="o", ms=4,
               label="Actual fire: join-up day (mean of 40/50/60%)")
        a.plot(df.index, df.half_count_day, color=SERIES[3], lw=1.5, marker="o", ms=3, ls=":",
               label="Half the year's fires have happened (count only)")
        a.set_yticks(starts.dayofyear, [s.strftime("%b") for s in starts])
        a.set_ylim(225, 335)
        s = summ[rule]
        a.set_title(f"{RULE_NAME[rule]}: actual SD {s['join_day_sd']:.0f} days, shuffled SD {s['shuffled_day_sd']:.0f}, "
                    f"correlation {s['corr_join_vs_shuffled']:.2f}", loc="left", fontsize=10, color=INK)
        a.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    for a in axs.ravel():
        a.grid(color=GRID, lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Days until fire joins up, Brazil 2003–2024 (MODIS Aqua fire events, added on the day they start)",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_days.png", dpi=150)


if __name__ == "__main__":
    main()
