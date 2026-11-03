"""IBGE municipal boundaries and point-in-municipality assignment for fire detections.

Assignment uses a 0.05 degree grid: each (cell, municipality) pair is precomputed as
either `inside` (cell wholly within the municipality, so no geometry test is needed
for points there) or edge (exact ST_Contains test only against the few candidates).
Results are keyed by distinct (latitude, longitude) in `point_municipality`, so newly
loaded detections can be assigned incrementally.

IBGE geometries are SIRGAS 2000 (EPSG:4674); FIRMS is WGS84. The two agree to well
under a metre, far below FIRMS pixel size (375 m-1 km), so no reprojection is done.
"""
import logging
import time
import zipfile

from . import config, db
from .client import Client, sha256_of
from .log import log_event

YEAR = 2025
ZIP_URL = ("https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/"
           f"municipio_{YEAR}/Brasil/BR_Municipios_{YEAR}.zip")
BOUNDARY_DIR = config.RAW_DIR / "boundaries"
ZIP_PATH = BOUNDARY_DIR / f"BR_Municipios_{YEAR}.zip"
SHP_PATH = BOUNDARY_DIR / f"BR_Municipios_{YEAR}.shp"
CELL = 0.05  # degrees

BIOME_URL = "https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/biomas/vetores/Biomas_250mil.zip"
BIOME_DIR = config.RAW_DIR / "biomes"
BIOME_ZIP = BIOME_DIR / "Biomas_250mil.zip"
BIOME_SHP = BIOME_DIR / "lm_bioma_250.shp"

SCHEMA = """
CREATE TABLE IF NOT EXISTS municipalities (
    cd_mun      VARCHAR PRIMARY KEY,  -- IBGE 7-digit municipality code
    nm_mun      VARCHAR,
    cd_rgi      VARCHAR, nm_rgi   VARCHAR,   -- immediate geographic region
    cd_rgint    VARCHAR, nm_rgint VARCHAR,   -- intermediate geographic region
    cd_uf       VARCHAR, nm_uf    VARCHAR, sigla_uf VARCHAR,  -- state
    cd_regiao   VARCHAR, nm_regiao VARCHAR, sigla_rg VARCHAR, -- macro-region
    area_km2    DOUBLE,
    geom        GEOMETRY,             -- SIRGAS 2000 lon/lat
    source      VARCHAR,
    source_sha256 VARCHAR
);
CREATE TABLE IF NOT EXISTS municipality_grid (
    cx INTEGER, cy INTEGER, cd_mun VARCHAR, inside BOOLEAN
);
CREATE TABLE IF NOT EXISTS point_municipality (
    latitude  DOUBLE,
    longitude DOUBLE,
    cd_mun    VARCHAR,   -- NULL: not inside any municipality (offshore / just over a border)
    method    VARCHAR    -- cell | contains | none
);
CREATE TABLE IF NOT EXISTS biomes (
    cd_bioma INTEGER PRIMARY KEY, nm_bioma VARCHAR, geom GEOMETRY, source VARCHAR, source_sha256 VARCHAR
);
CREATE TABLE IF NOT EXISTS biome_pieces (piece_id BIGINT, cd_bioma INTEGER, geom GEOMETRY);  -- 1-degree tiles
CREATE TABLE IF NOT EXISTS biome_grid (cx INTEGER, cy INTEGER, piece_id BIGINT, inside BOOLEAN);
CREATE TABLE IF NOT EXISTS point_biome (latitude DOUBLE, longitude DOUBLE, cd_bioma INTEGER, method VARCHAR);
CREATE OR REPLACE VIEW fire_detections_mun AS
SELECT f.*, p.cd_mun, m.nm_mun, m.sigla_uf, m.nm_uf, m.nm_regiao, m.nm_rgint, m.nm_rgi, pb.cd_bioma, b.nm_bioma
FROM fire_detections f
LEFT JOIN point_municipality p USING (latitude, longitude)
LEFT JOIN municipalities m USING (cd_mun)
LEFT JOIN point_biome pb USING (latitude, longitude)
LEFT JOIN biomes b USING (cd_bioma);
"""


def ensure_schema(con):
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute(SCHEMA)


def ensure_files(client: Client):
    if not SHP_PATH.exists():
        if not ZIP_PATH.exists():
            client.download(ZIP_URL, ZIP_PATH, "boundaries:ibge_municipios")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(BOUNDARY_DIR)
        log_event("unzip", f"extracted {ZIP_PATH.name}", unit="boundaries:ibge_municipios")
    return sha256_of(ZIP_PATH)


def load_municipalities(con, sha: str, force=False):
    have = con.execute("SELECT count(*), any_value(source_sha256) FROM municipalities").fetchone()
    if have[0] and have[1] == sha and not force:
        log_event("skip", f"municipalities already loaded ({have[0]:,} polygons)", rows=have[0])
        return False
    t0 = time.perf_counter()
    con.execute("BEGIN")
    con.execute("DELETE FROM municipalities")
    con.execute("DELETE FROM municipality_grid")
    con.execute("DELETE FROM point_municipality")  # boundaries changed: all assignments are stale
    n = con.execute(f"""
        INSERT INTO municipalities
        SELECT CD_MUN, NM_MUN, CD_RGI, NM_RGI, CD_RGINT, NM_RGINT, CD_UF, NM_UF, SIGLA_UF,
               CD_REGIAO, NM_REGIAO, SIGLA_RG, AREA_KM2, ST_MakeValid(geom), ?, ?
        FROM ST_Read('{SHP_PATH.as_posix()}')""", [ZIP_URL, sha]).fetchone()[0]
    con.execute("COMMIT")
    invalid = con.execute("SELECT count(*) FROM municipalities WHERE NOT ST_IsValid(geom)").fetchone()[0]
    log_event("municipalities_loaded", f"{n:,} polygons from {SHP_PATH.name} ({invalid} still invalid)",
              rows=n, duration_ms=round((time.perf_counter() - t0) * 1000), invalid=invalid)
    return True


from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    name: str     # label for logs
    polys: str    # polygon table (one row per pid)
    pid: str      # polygon id column (the unit the grid is built on)
    key: str      # area id column reported per point (== pid, or coarser for tiled layers)
    grid: str     # grid table: (cx, cy, <pid>, inside)
    points: str   # result table: (latitude, longitude, <key>, method)


MUNICIPALITIES = Layer("municipality", "municipalities", "cd_mun", "cd_mun", "municipality_grid", "point_municipality")
BIOMES = Layer("biome", "biome_pieces", "piece_id", "cd_bioma", "biome_grid", "point_biome")


def build_grid(con, layer: Layer = MUNICIPALITIES):
    if con.execute(f"SELECT count(*) FROM {layer.grid}").fetchone()[0]:
        log_event("skip", f"{layer.name} grid already built")
        return
    t0 = time.perf_counter()
    env = f"ST_MakeEnvelope(cx*{CELL}, cy*{CELL}, (cx+1)*{CELL}, (cy+1)*{CELL})"
    con.execute(f"""
        INSERT INTO {layer.grid}
        WITH c AS (
            SELECT {layer.pid}, geom, cx, cy FROM {layer.polys},
              range(floor(ST_XMin(geom)/{CELL})::INT, floor(ST_XMax(geom)/{CELL})::INT + 1) a(cx),
              range(floor(ST_YMin(geom)/{CELL})::INT, floor(ST_YMax(geom)/{CELL})::INT + 1) b(cy))
        SELECT cx, cy, {layer.pid}, ST_Within({env}, geom) FROM c WHERE ST_Intersects({env}, geom)""")
    n, inside = con.execute(f"SELECT count(*), sum(inside::INT) FROM {layer.grid}").fetchone()
    log_event("grid_built", f"{layer.name}: {n:,} cell/polygon pairs, {inside:,} wholly inside ({CELL} deg cells)",
              rows=n, inside=inside, layer=layer.name, duration_ms=round((time.perf_counter() - t0) * 1000))


def assign(con, layer: Layer = MUNICIPALITIES):
    """Assign every not-yet-assigned distinct fire location to an area of `layer`."""
    t0 = time.perf_counter()
    L = layer
    cell = f"floor(longitude/{CELL})::INT AS cx, floor(latitude/{CELL})::INT AS cy"
    con.execute("BEGIN")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE new_pts AS
        SELECT latitude, longitude, {cell} FROM (
            SELECT DISTINCT latitude, longitude FROM fire_detections
            ANTI JOIN {L.points} USING (latitude, longitude))""")
    todo = con.execute("SELECT count(*) FROM new_pts").fetchone()[0]
    log_event("assign_start", f"{L.name}: {todo:,} new distinct fire locations to assign", rows=todo, layer=L.name)

    by_cell = con.execute(f"""
        INSERT INTO {L.points}
        SELECT p.latitude, p.longitude, min(a.{L.key}), 'cell'
        FROM new_pts p
        JOIN {L.grid} g ON g.cx = p.cx AND g.cy = p.cy AND g.inside
        JOIN {L.polys} a ON a.{L.pid} = g.{L.pid}
        GROUP BY ALL""").fetchone()[0]
    by_contains = con.execute(f"""
        INSERT INTO {L.points}
        SELECT p.latitude, p.longitude, min(a.{L.key}), 'contains'
        FROM new_pts p
        ANTI JOIN {L.points} pm USING (latitude, longitude)
        JOIN {L.grid} g ON g.cx = p.cx AND g.cy = p.cy AND NOT g.inside
        JOIN {L.polys} a ON a.{L.pid} = g.{L.pid}
        WHERE ST_Contains(a.geom, ST_Point(p.longitude, p.latitude))
        GROUP BY ALL""").fetchone()[0]
    unmatched = con.execute(f"""
        INSERT INTO {L.points}
        SELECT latitude, longitude, NULL, 'none' FROM new_pts
        ANTI JOIN {L.points} USING (latitude, longitude)""").fetchone()[0]
    con.execute("DROP TABLE new_pts")
    con.execute("COMMIT")
    log_event("assign_done", f"{L.name}: {by_cell:,} by whole cell, {by_contains:,} by exact test, "
              f"{unmatched:,} outside every {L.name}", rows=by_cell + by_contains, layer=L.name,
              by_cell=by_cell, by_contains=by_contains, unmatched=unmatched,
              duration_ms=round((time.perf_counter() - t0) * 1000))
    return dict(assigned=by_cell + by_contains, unmatched=unmatched)


def ensure_biome_files(client: Client):
    if not BIOME_SHP.exists():
        if not BIOME_ZIP.exists():
            client.download(BIOME_URL, BIOME_ZIP, "boundaries:ibge_biomas")
        with zipfile.ZipFile(BIOME_ZIP) as z:
            z.extractall(BIOME_DIR)
        log_event("unzip", f"extracted {BIOME_ZIP.name}", unit="boundaries:ibge_biomas")
    return sha256_of(BIOME_ZIP)


def load_biomes(con, sha: str, force=False):
    """Load IBGE biomes and cut them into 1-degree tiles: the raw polygons have up to 350k
    vertices each, far too many to test grid cells against directly."""
    have = con.execute("SELECT count(*), any_value(source_sha256) FROM biomes").fetchone()
    if have[0] and have[1] == sha and not force:
        log_event("skip", f"biomes already loaded ({have[0]} polygons)", rows=have[0])
        return False
    t0 = time.perf_counter()
    con.execute("BEGIN")
    for t in ("biomes", "biome_pieces", "biome_grid", "point_biome"):
        con.execute(f"DELETE FROM {t}")
    n = con.execute(f"""
        INSERT INTO biomes SELECT CD_Bioma, Bioma, ST_MakeValid(geom), ?, ?
        FROM ST_Read('{BIOME_SHP.as_posix()}')""", [BIOME_URL, sha]).fetchone()[0]
    pieces = con.execute("""
        INSERT INTO biome_pieces
        SELECT row_number() OVER (), cd_bioma, geom FROM (
            SELECT b.cd_bioma,
                   ST_CollectionExtract(ST_Intersection(b.geom, ST_MakeEnvelope(tx, ty, tx+1, ty+1)), 3) AS geom
            FROM biomes b,
              range(floor(ST_XMin(b.geom))::INT, floor(ST_XMax(b.geom))::INT + 1) a(tx),
              range(floor(ST_YMin(b.geom))::INT, floor(ST_YMax(b.geom))::INT + 1) c(ty)
            WHERE ST_Intersects(b.geom, ST_MakeEnvelope(tx, ty, tx+1, ty+1)))
        WHERE NOT ST_IsEmpty(geom)""").fetchone()[0]
    con.execute("COMMIT")
    log_event("biomes_loaded", f"{n} biomes from {BIOME_SHP.name}, cut into {pieces:,} 1-degree tiles",
              rows=n, pieces=pieces, duration_ms=round((time.perf_counter() - t0) * 1000))
    return True


def validate(con):
    total, unmatched = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE cd_mun IS NULL) FROM fire_detections_mun""").fetchone()
    log_event("validate_areas", f"{total - unmatched:,} of {total:,} detections mapped to a municipality "
              f"({unmatched:,} = {unmatched / max(total, 1):.3%} unmatched)",
              level=logging.WARNING if unmatched / max(total, 1) > 0.01 else logging.INFO,
              rows=total, unmatched=unmatched)
    top = con.execute("""
        SELECT sigla_uf, count(*) n FROM fire_detections_mun WHERE sigla_uf IS NOT NULL
        GROUP BY ALL ORDER BY n DESC LIMIT 10""").fetchall()
    log_event("validate_areas_top_states", ", ".join(f"{uf} {n:,}" for uf, n in top))
    biomes = con.execute("""
        SELECT coalesce(nm_bioma, '(none)'), count(*) n FROM fire_detections_mun
        GROUP BY ALL ORDER BY n DESC""").fetchall()
    log_event("validate_areas_biomes", ", ".join(f"{b} {n:,}" for b, n in biomes))
    return total, unmatched, top
