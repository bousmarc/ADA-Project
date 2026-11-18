"""
code/01_geo_features.py
-----------------------
Stage 1 of the pipeline: geographic features of every commune, from swisstopo's
swissBOUNDARIES3D: position (centroid) and distance to the national border.
This is the only stage that needs geopandas (reading the maps takes 1-2 minutes).

Input  : data/raw/swissBOUNDARIES3D_1_5_LV95_LN02.gdb   (swisstopo, free open data;
         layers TLM_HOHEITSGEBIET = communes, TLM_LANDESGEBIET = countries)
         data/raw/population_2022.csv                   (the 2025 commune list)
Output : data/processed/commune_geo.csv
         code, x_lv95, y_lv95, border_km, border_commune, geo_approx

Coordinates are Swiss LV95 (EPSG:2056), in metres: x = east, y = north.
Source: Federal Office of Topography swisstopo, swissBOUNDARIES3D.

Run from the repository folder:   python code/01_geo_features.py

Written with the help of Claude (Anthropic); see the paper's appendix.
"""

import geopandas as gpd
import pandas as pd
import shapely

import helpers
from helpers import PROCESSED, RAW

PROCESSED.mkdir(parents=True, exist_ok=True)
GDB = RAW / "swissBOUNDARIES3D_1_5_LV95_LN02.gdb"
communes_2025 = helpers.communes_2025()


# ---------------------------------------------------------------------------
# 1. Commune polygons
#    This version of swissBOUNDARIES3D uses the commune list of 1 January 2026,
#    while our data use the list of 6 April 2025. Between the two:
#      - Fétigny + Ménières (FR) merged into Fétigny-Ménières (2056)
#      - Ulmiz (FR) merged into Murten (2275)
#      - Halten + Oekingen (SO) merged into Kriegstetten (2525)
#      - Villnachern (AG) merged into Brugg (4095)
#      - Moutier moved from canton Bern (700) to canton Jura (6831), same territory
#    For a merged commune only the merged polygon exists, so its parts get the
#    position of the whole (flagged geo_approx = 1). For distances of several
#    kilometres this error is small.
# ---------------------------------------------------------------------------
LATER_CHANGES = {           # 2025 commune number -> 2026 polygon number
    2016: 2056, 2027: 2056,     # Fétigny, Ménières
    2278: 2275,                 # Ulmiz -> Murten
    2520: 2525, 2529: 2525,     # Halten, Oekingen -> Kriegstetten
    4122: 4095,                 # Villnachern -> Brugg
    700: 6831,                  # Moutier BE -> JU (not a merger)
}

com = gpd.read_file(GDB, layer="TLM_HOHEITSGEBIET")
com = com[com["ICC"] == "CH"]                   # drop Liechtenstein, Büsingen (DE), Campione (IT)
com = com.set_crs(2056, allow_override=True)    # LV95 (the file also stores heights; we use 2D)

# Keep only real communes: lakes (numbers 9xxx) and jointly owned land
# (e.g. "Comunanza", "Staatswald Galm") have no residents and are not communes.
wanted = {LATER_CHANGES.get(c, c) for c in communes_2025}
not_used = com.loc[~com["BFS_NUMMER"].isin(wanted), ["BFS_NUMMER", "NAME", "EINWOHNERZAHL"]]
assert (not_used["EINWOHNERZAHL"] == 0).all(), "a populated area is not matched to a commune"
com = com[com["BFS_NUMMER"].isin(wanted)]
assert com["BFS_NUMMER"].is_unique, "a commune has several polygons"


# ---------------------------------------------------------------------------
# 2. National border = the outline of Switzerland
#    It includes borders inside lakes (Lake Geneva, Lake Lugano, Lake Constance)
#    and around the foreign enclaves Büsingen and Campione. The border with
#    Liechtenstein counts as a national border.
# ---------------------------------------------------------------------------
land = gpd.read_file(GDB, layer="TLM_LANDESGEBIET").set_crs(2056, allow_override=True)
switzerland = shapely.force_2d(land.loc[land["ICC"] == "CH", "geometry"].union_all())
# The outlines have a point every few metres; simplifying to 5 m changes distances
# by at most a few metres but makes the computation hundreds of times faster.
border = switzerland.boundary.simplify(5)
shapely.prepare(border)                         # index the line once for fast distances


# ---------------------------------------------------------------------------
# 3. Features per polygon
# ---------------------------------------------------------------------------
shapes = shapely.force_2d(com.geometry.to_numpy())
centroid = shapely.centroid(shapes)             # centroid of the full-detail polygon
outline = shapely.simplify(shapely.boundary(shapes), 5)
geo = pd.DataFrame({
    "polygon": com["BFS_NUMMER"].astype(int).to_numpy(),
    "x_lv95": shapely.get_x(centroid).round(0),
    "y_lv95": shapely.get_y(centroid).round(0),
    # distance from the commune's centre to the nearest point of the border
    "border_km": (shapely.distance(centroid, border) / 1000).round(2),
    # 1 if the commune's territory touches the national border (within 20 m,
    # to allow for the simplification)
    "border_commune": shapely.dwithin(outline, border, 20).astype(int),
})


# ---------------------------------------------------------------------------
# 4. Back to the 2025 commune list
# ---------------------------------------------------------------------------
out = pd.DataFrame({"code": sorted(communes_2025)})
out["polygon"] = out["code"].map(lambda c: LATER_CHANGES.get(c, c))
out = out.merge(geo, on="polygon", how="left")
# approximate position when several 2025 communes share one 2026 polygon
# (the merged communes and the commune they joined; Moutier is exact)
out["geo_approx"] = out.duplicated("polygon", keep=False).astype(int)
out = out.drop(columns="polygon")

# Checks
assert len(out) == len(communes_2025) and out["code"].is_unique
assert out.notna().all().all(), "a commune without geographic data"
assert out["x_lv95"].between(2_480_000, 2_840_000).all(), "x outside Switzerland"
assert out["y_lv95"].between(1_070_000, 1_300_000).all(), "y outside Switzerland"
assert (out["border_km"] >= 0).all()

print(f"Communes: {len(out)} | approximate position: {out['geo_approx'].sum()} | "
      f"border communes: {out['border_commune'].sum()}")
out.to_csv(PROCESSED / "commune_geo.csv", index=False, encoding="utf-8-sig")
print(f"Saved -> {PROCESSED / 'commune_geo.csv'}")
