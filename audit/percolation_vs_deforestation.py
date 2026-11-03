"""Critical point over time against yearly deforestation (PRODES Legal Amazon), 2003-2024.

Deforestation: INPE PRODES official annual rate, km² cleared in the Legal Amazon, summed over states from the
file the TerraBrasilis dashboard uses (data/raw/prodes_rates/rates2026.json, downloaded 2026-10-29 from
https://terrabrasilis.dpi.inpe.br/app/prodes/dashboard/deforestation/files/rates2026.json).
A PRODES year runs August to July, so "2019" = Aug 2018-Jul 2019: land cleared just before the 2019 burning
season (Aug-Oct). That lines up with the fire year. Only the Legal Amazon has an official yearly series back to
2003; the Cerrado is not included.

Fire: the half-joined distance (d50) of fire events and its ratio to random points, from
percolation_pilot_2019_events.json (run fire_percolation_pilot.py run 2019 --events first).

    python audit/percolation_vs_deforestation.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from audit.fire_percolation_pilot import GRID, INK, INK2, MUTED, OUT, SERIES
from firms import config


def prodes():
    r = json.loads((config.RAW_DIR / "prodes_rates" / "rates2026.json").read_text())
    return pd.Series({p["endDate"]["year"]: sum(a["area"] for f in p["features"] for a in f["areas"])
                      for p in r["periods"]}, name="deforestation_km2")


def main():
    import matplotlib.pyplot as plt
    evo = pd.DataFrame(json.loads((OUT / "percolation_pilot_2019_events.json").read_text())["evolution"])
    df = evo.set_index("year")[["real_d50_km", "null_d50_km"]].join(prodes(), how="inner")
    df["ratio"] = df.real_d50_km / df.null_d50_km
    df.to_csv(OUT / "percolation_vs_deforestation.csv")

    res = {}
    for col in ("ratio", "real_d50_km"):
        x, y = df.deforestation_km2, df[col]
        res[col] = {"pearson": stats.pearsonr(x, y), "spearman": stats.spearmanr(x, y),
                    "pearson_year_to_year_change": stats.pearsonr(x.diff().dropna(), y.diff().dropna())}
    for col, r in res.items():
        print(col, {k: (round(float(v[0]), 2), round(float(v[1]), 3)) for k, v in r.items()})

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    a = ax[0]
    a.bar(df.index, df.deforestation_km2 / 1000, color=SERIES[3], width=0.75)
    a.set_title("Deforestation, Legal Amazon (PRODES, thousand km² per year)", loc="left", fontsize=10, color=INK)
    a = ax[1]
    a.plot(df.index, df.ratio, color=SERIES[0], lw=2, marker="o", ms=4)
    a.axhline(1, color=MUTED, lw=0.8, ls=":")
    a.set_title("Critical point: fire / random half-joined distance", loc="left", fontsize=10, color=INK)
    a = ax[2]
    early = df.index <= 2008
    a.scatter(df.deforestation_km2[~early] / 1000, df.ratio[~early], color=SERIES[0], s=40, zorder=3, label="2009–2024")
    a.scatter(df.deforestation_km2[early] / 1000, df.ratio[early], color=SERIES[1], s=40, zorder=3, label="2003–2008")
    for y, row in df.iterrows():
        a.annotate(str(y), (row.deforestation_km2 / 1000, row.ratio), xytext=(4, 3), textcoords="offset points",
                   fontsize=7.5, color=INK2)
    rp, pp = res["ratio"]["pearson"]
    a.set_title(f"Each dot is a year (r = {rp:.2f}, p = {pp:.2f})", loc="left", fontsize=10, color=INK)
    a.set_xlabel("Deforestation (thousand km²)", color=INK2)
    a.set_ylabel("Fire / random", color=INK2)
    a.legend(frameon=False, fontsize=9)
    for a in ax:
        a.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle("Is the critical point linked to how much forest was cleared? MODIS Aqua fire events 2003–2024 "
                 "(PRODES year = Aug–Jul, ending just before that year's burning season)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "percolation_vs_deforestation.png", dpi=150)
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
