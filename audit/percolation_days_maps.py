"""Example maps for 'days until fire joins up': the largest cluster on chosen dates, actual vs shuffled dates.

Same rule as percolation_days.py (scaled distance, events added on the day they start). For each snapshot date
only events that have started by then are drawn; two events are joined if within d. Colours: purple largest
cluster, green 2nd, pink 3rd, orange 4th, grey all other events so far.
Years: 2010 (actual joins up ~14 days after shuffled) and 2008 (no delay, -3 days).
Shuffled row: of 10 runs with start dates permuted among the same events, the one whose join day is the median.

    python audit/percolation_days_maps.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from audit.fire_percolation_pilot import ALBERS, INK, INK2, OUT, project
from audit.percolation_days import SCALE, join_day, largest_by_day
from audit.percolation_lattice_threshold import RANK_COLOURS
from audit.percolation_stability import events_with_dates
from firms import config

YEARS = [2010, 2008]
DATES = ["09-01", "10-01", "10-15", "11-01"]


def snapshot(n, p, doy, day):
    """Cluster rank (0 = largest, ... 3, 4 = other) per started event, and largest share of the year's events."""
    on = doy <= day
    keep = on[p[:, 0]] & on[p[:, 1]]
    e = p[keep]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    size = np.bincount(lab[on], minlength=lab.max() + 1)
    top = np.argsort(size)[::-1][:4]
    rank = np.full(n, 4)
    for r, c in enumerate(top):
        rank[lab == c] = r
    return on, rank, size.max() / n


def main():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    area = gpd.read_parquet(config.DATA_DIR / "cache" / "brazil_outline_5880.parquet").to_crs(ALBERS).area.sum() / 1e6
    biomes = gpd.read_file(config.RAW_DIR / "biomes" / "lm_bioma_250.shp").set_crs("EPSG:4674", allow_override=True)
    days = pd.read_csv(OUT / "percolation_days.csv").set_index("year")
    rng = np.random.default_rng(30)
    fig, axs = plt.subplots(2 * len(YEARS), len(DATES), figsize=(16, 16.5))
    for yi, y in enumerate(YEARS):
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y]
        lon, lat = ev.lon.to_numpy(), ev.lat.to_numpy()
        xy = project(lon, lat)
        doy = (ev.start.dt.dayofyear - 1).to_numpy()
        n = len(xy)
        d = SCALE * np.sqrt(area / n)
        p = cKDTree(xy).query_pairs(d, output_type="ndarray")
        row = days.loc[y]
        perms = [rng.permutation(doy) for _ in range(10)]
        pj = [join_day(largest_by_day(n, p[:, 0], p[:, 1], q), n) for q in perms]
        typical = perms[int(np.argmin(np.abs(np.array(pj) - np.median(pj))))]
        for k, (name, dd) in enumerate((("actual dates", doy), ("dates shuffled", typical))):
            r = 2 * yi + k
            for c, md in enumerate(DATES):
                day = pd.Timestamp(f"{y}-{md}").dayofyear - 1
                on, rank, share = snapshot(n, p, dd, day)
                a = axs[r, c]
                biomes.boundary.plot(ax=a, color=INK2, lw=0.3)
                m = on & (rank == 4)
                a.scatter(lon[m], lat[m], s=0.05, color="#c9c7bf", rasterized=True)
                for q in range(3, -1, -1):
                    m = on & (rank == q)
                    a.scatter(lon[m], lat[m], s=0.05, color=RANK_COLOURS[q], rasterized=True)
                when = pd.Timestamp(f"{y}-{md}").strftime("%-d %b") if sys.platform != "win32" else \
                    pd.Timestamp(f"{y}-{md}").strftime("%#d %b")
                a.set_title(f"{when}: {on.mean():.0%} of fires so far\nlargest cluster {share:.0%} of the year's fires",
                            loc="left", fontsize=9.5, color=INK, weight="bold" if share >= 0.5 else "normal")
                a.set_xlim(-74.5, -34)
                a.set_ylim(-34, 5.8)
                a.set_axis_off()
                a.set_aspect("equal")
            jd = join_day(largest_by_day(n, p[:, 0], p[:, 1], dd), n)
            label = f"{y}\n{name}\njoins up ~{(pd.Timestamp(f'{y}-01-01') + pd.Timedelta(days=jd - 1)).strftime('%#d %b')}"
            axs[r, 0].text(-0.08, 0.5, label, transform=axs[r, 0].transAxes, fontsize=12, color=INK, rotation=90,
                           va="center", ha="right", weight="bold" if k == 0 else "normal")
    fig.suptitle("How the largest fire cluster builds through the season: actual start dates vs the same fires with "
                 "dates shuffled (shown: the shuffled run closest to the median of 10)\n2010: actual joins up ~14 days later than shuffled (median of "
                 "10 runs). 2008: no delay. Scaled distance (16 km in 2010, 18 km in 2008).\nPurple largest, green 2nd, pink 3rd, orange 4th, grey other fires so "
                 "far. Bold = largest cluster past 50%", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    fig.savefig(OUT / "percolation_days_maps.png", dpi=120)


if __name__ == "__main__":
    main()
