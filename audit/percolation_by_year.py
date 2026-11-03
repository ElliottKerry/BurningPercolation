"""Q1: do fire connectivity patterns change between years? Both critical points against year, with fire totals.

Reads results already computed for every year 2003-2024:
  distance method, fire events:     expected_vs_actual_years_events.csv  (fire_percolation_expected.py --events)
  distance method, raw detections:  percolation_pilot_2019.json "evolution" (fire_percolation_pilot.py)
  lattice method, events per cell:  lattice_threshold_years.csv           (percolation_lattice_threshold.py)
d50 = joining distance at which the largest cluster holds half the points. k* = events-per-cell threshold
where the second-largest lattice cluster peaks; p* = share of cells still on at k*.

    python audit/percolation_by_year.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES


def table():
    e = pd.read_csv(OUT / "expected_vs_actual_years_events.csv").set_index("year")
    lat = pd.read_csv(OUT / "lattice_threshold_years.csv").set_index("year")
    det = pd.DataFrame(json.loads((OUT / "percolation_pilot_2019.json").read_text())["evolution"]).set_index("year")
    return pd.DataFrame({
        "detections": det.detections, "events": e.points,
        "d50_detections_km": det.real_d50_km, "d50_detections_random_km": det.null_d50_km,
        "d50_events_km": e.observed, "d50_events_random_km": e.anywhere_mid,
        "d50_events_samecell_km": e.samecell_mid,
        "k_star": lat.k_star, "k_star_shuffled": lat.null_k_star_mid,
        "p_star": lat.p_star, "p_star_shuffled": lat.null_p_star_mid})


def main():
    import matplotlib.pyplot as plt
    df = table()
    df["ratio_events"] = df.d50_events_km / df.d50_events_random_km
    df.to_csv(OUT / "percolation_by_year.csv")
    tests = {}
    for col in ("d50_events_km", "d50_detections_km", "ratio_events", "k_star", "p_star"):
        r, p = stats.spearmanr(df.events, df[col])
        tests[col] = {"spearman_with_events": round(float(r), 3), "p": round(float(p), 4)}
    r, p = stats.spearmanr(df.d50_events_km, df.k_star)
    tests["d50_events_vs_k_star"] = {"spearman": round(float(r), 3), "p": round(float(p), 4)}
    (OUT / "percolation_by_year.json").write_text(json.dumps(tests, indent=1))
    print(df.round(3).to_string())
    print(json.dumps(tests, indent=1))

    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    a = ax[0]
    a.bar(df.index - 0.2, df.detections / 1000, width=0.4, color=SERIES[3], label="Detections")
    a.bar(df.index + 0.2, df.events / 1000, width=0.4, color=SERIES[0], label="Fire events (2 km / 2 days)")
    a.set_ylabel("Thousands", color=INK2)
    a.set_title("Fire totals per year, Brazil (MODIS Aqua, daytime)", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=9)
    a = ax[1]
    a.plot(df.index, df.d50_events_km, color=SERIES[0], lw=2, marker="o", ms=4, label="Fire events")
    a.plot(df.index, df.d50_events_random_km, color=SERIES[0], lw=1.2, ls="--", label="Random points, same number")
    a.plot(df.index, df.d50_detections_km, color=SERIES[3], lw=2, marker="o", ms=4, label="Detections")
    a.plot(df.index, df.d50_detections_random_km, color=SERIES[3], lw=1.2, ls="--", label="Random points, same number")
    a.set_ylabel("d50 (km)", color=INK2)
    a.set_title("Distance method: joining distance at which the largest cluster holds half the points (d50)",
                loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8.5, ncol=2)
    a = ax[2]
    a.plot(df.index, df.k_star, color=SERIES[2], lw=2, marker="o", ms=4, label="Fire events")
    a.plot(df.index, df.k_star_shuffled, color=MUTED, lw=1.2, ls="--", label="Counts shuffled among cells")
    a.set_ylabel("k* (events per cell)", color=INK2)
    a.set_title("Lattice method: threshold k* where the second-largest cluster peaks (0.25° cells)",
                loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=9)
    a = ax[3]
    a.plot(df.index, df.p_star, color=SERIES[2], lw=2, marker="o", ms=4, label="Fire events")
    a.plot(df.index, df.p_star_shuffled, color=MUTED, lw=1.2, ls="--", label="Counts shuffled among cells")
    a.axhline(0.593, color=INK2, lw=0.8, ls=":")
    a.set_ylabel("p* (share of cells on)", color=INK2)
    a.set_title("Lattice method: share of cells still on at k* (random-lattice theory 0.593, dotted)",
                loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=9)
    for a in ax:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[-1].set_xticks(df.index[::2])
    fig.suptitle("Fire connectivity by year, 2003–2024: both critical points with the fire totals", x=0.01,
                 ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_by_year.png", dpi=150)


if __name__ == "__main__":
    main()
