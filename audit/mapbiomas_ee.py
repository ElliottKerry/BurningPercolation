"""MapBiomas land cover (Collection 10, 30 m, yearly) through Google Earth Engine, for the percolation work.

Needs a logged-in Earth Engine account (earthengine authenticate; project set below).

  fires   land-cover class under every MODIS Aqua fire event (2 km / 2 days, events_with_dates), 2003-2024:
          lc_prev = class in the map of the year before the fire, lc_same = class in the fire's own year.
          Point sample of the 30 m map at the event position (mean of its detections).
          -> data/cache/mapbiomas/fires_{year}.parquet (lon, lat, start, lc_prev, lc_same)
  cells   share of each land-cover group in every 0.25 deg cell (same cells as the lattice / ERA5:
          index floor(lon / 0.25 + 0.5)), per year 2003-2024. Exported to Google Drive as one table per
          year (Earth Engine batch task, CSV), then downloaded.
          -> data/cache/mapbiomas/cells_{year}.csv, then data/cache/mapbiomas_cells.parquet
  status  list Earth Engine tasks

Groups (MapBiomas Collection 10 legend codes):
  forest     3 forest formation, 5 mangrove, 6 floodable forest, 49 wooded sandbank vegetation
  savanna    4 savanna formation
  grassland  11 wetland, 12 grassland, 13 other non-forest natural, 29 rocky outcrop, 32 salt flat, 50 herbaceous sandbank
  pasture    15
  agriculture 9 forest plantation, 18-20, 35, 36, 39-41, 46-48, 62 (crops)
  mosaic     21 mosaic of uses
  other      everything else (urban, mining, water, not observed, ...)

    python audit/mapbiomas_ee.py fires
    python audit/mapbiomas_ee.py cells-export
    python audit/mapbiomas_ee.py cells-download
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from firms import config

PROJECT = "primordial-arc-513513-c5"
ASSET = "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2"
YEARS = range(2003, 2025)
DIR = config.DATA_DIR / "cache" / "mapbiomas"
CHUNK = 4000
GROUPS = {"forest": [3, 5, 6, 49], "savanna": [4], "grassland": [11, 12, 13, 29, 32, 50], "pasture": [15],
          "agriculture": [9, 18, 19, 20, 35, 36, 39, 40, 41, 46, 47, 48, 62], "mosaic": [21]}
DRIVE_FOLDER = "mapbiomas_lattice"


def group_of(code):
    code = np.asarray(code)
    out = np.full(code.shape, "other", dtype=object)
    for g, cs in GROUPS.items():
        out[np.isin(code, cs)] = g
    return out


def init():
    import ee
    ee.Initialize(project=PROJECT)
    return ee


def fires():
    from audit.percolation_stability import events_with_dates
    ee = init()
    mb = ee.Image(ASSET)
    DIR.mkdir(parents=True, exist_ok=True)

    def chunk(args):
        img, lon, lat, i0 = args
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {"i": int(i0 + k)})
                                   for k, (x, y) in enumerate(zip(lon, lat))])
        for attempt in range(6):
            try:
                r = img.reduceRegions(fc, ee.Reducer.first(), scale=30).select(["i", "lc_prev", "lc_same"], None, False)
                return [f["properties"] for f in r.getInfo()["features"]]
            except Exception as e:                             # transient EE errors / rate limits: wait and retry
                err = e
                time.sleep(30 * (attempt + 1))
        raise err

    for y in YEARS:
        out = DIR / f"fires_{y}.parquet"
        if out.exists():
            continue
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y].reset_index(drop=True)
        img = mb.select([f"classification_{y - 1}", f"classification_{y}"], ["lc_prev", "lc_same"])
        jobs = [(img, ev.lon.to_numpy()[s:s + CHUNK], ev.lat.to_numpy()[s:s + CHUNK], s) for s in range(0, len(ev), CHUNK)]
        with ThreadPoolExecutor(4) as ex:
            rows = [p for part in ex.map(chunk, jobs) for p in part]
        r = pd.DataFrame(rows).set_index("i").reindex(range(len(ev)))
        ev["lc_prev"] = r.lc_prev.to_numpy()
        ev["lc_same"] = r.lc_same.to_numpy()
        ev.to_parquet(out, index=False)
        print(y, len(ev), "events;", "share by group (year before):",
              pd.Series(group_of(ev.lc_prev.fillna(0))).value_counts(normalize=True).round(3).to_dict(), flush=True)


def cells_export():
    """One Earth Engine table export per year: mean of each group's 0/1 layer over every 0.25 deg cell.

    Stage 1 averages the 30 m map to ~300 m (reduceResolution, exact shares); stage 2 averages those over the
    cell polygon (reduceRegions). Doing it per cell keeps each request small."""
    from audit.percolation_lattice_threshold import RES, brazil_cells
    ee = init()
    mb = ee.Image(ASSET)
    fine = ee.Projection("EPSG:4326").atScale(300)
    mask, ix0, iy0 = brazil_cells()
    yy, xx = np.nonzero(mask)
    grid = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([ix * RES - RES / 2, iy * RES - RES / 2,
                                                                   ix * RES + RES / 2, iy * RES + RES / 2]),
                                            {"ix": int(ix), "iy": int(iy)})
                                 for ix, iy in zip(xx + ix0, yy + iy0)])
    print(len(xx), "cells", flush=True)
    for y in YEARS:
        c = mb.select(f"classification_{y}")
        mapped = c.mask().gt(0).unmask(0)
        cu = c.unmask(0)
        bands = [mapped.rename("mapped")] + [cu.remap(cs, [1] * len(cs), 0).rename(g) for g, cs in GROUPS.items()]
        img = ee.Image.cat(bands).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(fine)
        tab = img.reduceRegions(grid, ee.Reducer.mean(), scale=300, crs="EPSG:4326", tileScale=4)
        t = ee.batch.Export.table.toDrive(collection=tab, description=f"mapbiomas_cells_{y}", folder=DRIVE_FOLDER,
                                          fileNamePrefix=f"cells_{y}", fileFormat="CSV",
                                          selectors=["ix", "iy", "mapped", *GROUPS])
        t.start()
        print("started", y, t.id, flush=True)


NATURAL = GROUPS["forest"] + GROUPS["savanna"] + GROUPS["grassland"]
FARM = GROUPS["pasture"] + GROUPS["agriculture"] + GROUPS["mosaic"]
NEAR_MAX_M = 2000
MIN_PIX = 70                                   # 70 x 30 m x 30 m = 6.3 ha, the PRODES minimum patch


def near():
    """Distance (m, up to NEAR_MAX_M) from each fire event to the nearest 30 m pixel converted from natural land to
    farmland, in patches of at least MIN_PIX connected pixels: conv_a = between the maps of y-1 and y, conv_b = between y and y+1 (as PRODES years y and y+1 in
    percolation_clearing.py). Missing = no converted pixel within NEAR_MAX_M. 2024 has no 2025 map (conv_b missing).
    -> data/cache/mapbiomas/near_{year}.parquet (row order as fires_{year}.parquet)"""
    from audit.percolation_stability import events_with_dates
    ee = init()
    mb = ee.Image(ASSET)
    kernel = ee.Kernel.euclidean(NEAR_MAX_M, "meters")

    def conv(a, b):
        ca, cb = mb.select(f"classification_{a}"), mb.select(f"classification_{b}")
        m = ca.remap(NATURAL, [1] * len(NATURAL), 0).And(cb.remap(FARM, [1] * len(FARM), 0))
        m = m.selfMask()
        m = m.updateMask(m.connectedPixelCount(MIN_PIX + 1, True).gte(MIN_PIX))   # patches >= ~6.25 ha, as PRODES
        return m.unmask(0).distance(kernel)

    def chunk(args):
        img, lon, lat, i0 = args
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {"i": int(i0 + k)})
                                   for k, (x, y) in enumerate(zip(lon, lat))])
        for attempt in range(8):
            try:
                r = img.reduceRegions(fc, ee.Reducer.first(), scale=30).select(["i", "conv_a", "conv_b"], None, False)
                return [f["properties"] for f in r.getInfo()["features"]]
            except Exception as e:
                err = e
                time.sleep(30 * (attempt + 1))
        raise err

    for y in YEARS:
        out = DIR / f"near_{y}.parquet"
        if out.exists():
            continue
        ev = events_with_dates(y)
        ev = ev[ev.start.dt.year == y].reset_index(drop=True)
        bands = [conv(y - 1, y).rename("conv_a")]
        bands.append(conv(y, y + 1).rename("conv_b") if y < 2024 else ee.Image.constant(0).selfMask().rename("conv_b"))
        img = ee.Image.cat(bands)
        jobs = [(img, ev.lon.to_numpy()[s:s + CHUNK], ev.lat.to_numpy()[s:s + CHUNK], s) for s in range(0, len(ev), CHUNK)]
        with ThreadPoolExecutor(5) as ex:
            rows = [p for part in ex.map(chunk, jobs) for p in part]
        r = pd.DataFrame(rows).set_index("i").reindex(range(len(ev)))
        out_df = pd.DataFrame({"conv_a": r.get("conv_a"), "conv_b": r.get("conv_b")}).astype(float)
        out_df.to_parquet(out, index=False)
        d = out_df.min(axis=1)
        print(y, len(ev), "events; within 1 km of conversion:", round(float((d <= 1000).mean()), 3),
              "2 km:", round(float((d <= 2000).mean()), 3), flush=True)


def status():
    ee = init()
    for t in ee.batch.Task.list()[:40]:
        s = t.status()
        print(s["description"], s["state"], s.get("error_message", ""))


def drive():
    import ee
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=Credentials(**ee.oauth.get_credentials_arguments()))


def cells_download():
    import io
    from googleapiclient.http import MediaIoBaseDownload
    d = drive()
    DIR.mkdir(parents=True, exist_ok=True)
    files = d.files().list(q="name contains 'cells_' and name contains '.csv' and trashed = false",
                           fields="files(id, name)", pageSize=200).execute()["files"]
    for f in files:
        out = DIR / f["name"]
        if out.exists():
            continue
        dl = MediaIoBaseDownload(buf := io.BytesIO(), d.files().get_media(fileId=f["id"]))
        done = False
        while not done:
            _, done = dl.next_chunk()
        out.write_bytes(buf.getvalue())
        print("downloaded", f["name"], flush=True)
    rows = []
    for y in YEARS:
        p = DIR / f"cells_{y}.csv"
        if not p.exists():
            print("missing", p.name)
            continue
        df = pd.read_csv(p).assign(year=y)
        df = df[df.mapped > 0]
        for g in GROUPS:                                         # share of the mapped (land) area of the cell
            df[g] = df[g] / df.mapped
        df["other"] = 1 - df[list(GROUPS)].sum(axis=1)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(config.DATA_DIR / "cache" / "mapbiomas_cells.parquet", index=False)
    print(out.groupby("year")[[*GROUPS, "other"]].mean().round(3).to_string())


if __name__ == "__main__":
    {"fires": fires, "cells-export": cells_export, "cells-download": cells_download,
     "status": status, "near": near}[sys.argv[1]]()
