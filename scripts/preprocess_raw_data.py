#!/usr/bin/env python3
"""Raw geospatial extraction helpers for ``Preprocess_new.ipynb``.

Every table is keyed and sorted by an explicit city ID. Outputs follow the raw
layout consumed by ``build_hf_dataset_from_raw.py``.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
FIRST_YEAR, LAST_YEAR = 2003, 2020
EXPECTED_MONTHS = 216
GHS_EPOCHS = (2005, 2010, 2015, 2020)

SOURCE_URLS = {
    "UHI intensity": "https://figshare.com/articles/dataset/Global_Urban_Heat_Island_Intensity_Dataset/24821538",
    "City boundaries": "https://data-starcloud.pcl.ac.cn/iearthdata/14",
    "AgERA5": "https://gee-community-catalog.org/projects/agera5_datasets/",
    "MOD13A3": "https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A3",
    "PM2.5": "https://gee-community-catalog.org/projects/global_pm25/",
    "GHS-POP": "https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_POP",
    "GHS-BUILT-S": "https://human-settlement.emergency.copernicus.eu/ghs_buS2023.php",
    "GHS-BUILT-H": "https://human-settlement.emergency.copernicus.eu/ghs_buH2023.php",
    "GHS-BUILT-C": "https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_BUILT_C",
    "GHS-SMOD": "https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2023A_GHS_SMOD_V2-0",
}

# Frozen manuscript assets. A newer AgERA5 release has different band names.
AGERA5_ASSET = "projects/climate-engine-pro/assets/ce-ag-era5/daily"
MOD13A3_ASSET = "MODIS/061/MOD13A3"
PM25_ASSET = "projects/sat-io/open-datasets/GLOBAL-SATELLITE-PM25/MONTHLY"
CLIMATE_BANDS = (
    "Wind_Speed_10m_Mean", "Dew_Point_Temperature_2m_Mean",
    "Relative_Humidity_2m_06h", "Relative_Humidity_2m_15h",
    "Temperature_Air_2m_Max_24h", "Temperature_Air_2m_Mean_24h",
    "Temperature_Air_2m_Min_24h", "Temperature_Air_2m_Max_Day_Time",
    "Temperature_Air_2m_Mean_Day_Time", "Temperature_Air_2m_Mean_Night_Time",
    "Temperature_Air_2m_Min_Night_Time", "Cloud_Cover_Mean",
    "Precipitation_Rain_Duration_Fraction", "Precipitation_Flux",
    "Snow_Thickness_Mean", "Snow_Thickness_LWE_Mean",
    "Solar_Radiation_Flux", "Precipitation_Solid_Duration_Fraction",
    "Vapour_Pressure_Mean",
)
LAND_USE_GROUPS = {
    "Open Spaces": ("1", "2", "3"), "Water Surfaces": ("4",),
    "Road Surfaces": ("5",),
    "Residential": ("11", "12", "13", "14", "15"),
    "Non-Residential": ("21", "22", "23", "24", "25"),
}
LAND_USE_CLASSES = ("0", "1", "2", "3", "4", "5", "11", "12", "13", "14", "15", "21", "22", "23", "24", "25")


def month_pairs() -> list[tuple[int, int]]:
    return [(year, month) for year in range(FIRST_YEAR, LAST_YEAR + 1) for month in range(1, 13)]


def _complete(path: Path, city_ids: Sequence[int]) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    if "id" not in frame or frame["id"].astype(int).tolist() != list(city_ids):
        return False
    value_columns = [
        column
        for column in frame.columns
        if column != "id" and not str(column).startswith("Unnamed")
    ]
    if not value_columns:
        return False
    values = frame[value_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    return bool(np.isfinite(values).all())


def _column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    lookup = {str(value).strip().casefold(): str(value) for value in columns}
    return next((lookup[name.casefold()] for name in candidates if name.casefold() in lookup), None)


def prepare_city_boundaries(
    source_path: Path, filtered_path: Path, metrics_path: Path, *,
    area_column: str = "urbanArea", expected_cities: int = 1020,
    overwrite: bool = False,
) -> tuple[Any, pd.DataFrame]:
    """Select cities with area at least 100 km2, assign IDs, and compute metrics."""
    import geopandas as gpd
    from pyproj import Geod
    from shapely import make_valid

    source_path, filtered_path, metrics_path = map(
        lambda path: Path(path).expanduser().resolve(),
        (source_path, filtered_path, metrics_path),
    )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not overwrite and (filtered_path.exists() or metrics_path.exists()):
        raise FileExistsError("outputs exist; set OVERWRITE=True to rebuild")
    cities = gpd.read_file(source_path)
    if cities.crs is None or area_column not in cities.columns:
        raise ValueError(f"boundary file needs a CRS and {area_column!r} column")
    cities = cities.to_crs("EPSG:4326").copy()
    cities["source_row"] = np.arange(len(cities))
    cities[area_column] = pd.to_numeric(cities[area_column], errors="coerce")
    cities = cities[cities[area_column].ge(100)].copy()
    cities["geometry"] = cities.geometry.map(make_valid)
    cities = cities.reset_index(drop=True)
    cities.insert(0, "id", np.arange(len(cities)))
    if len(cities) != expected_cities:
        raise ValueError(f"expected {expected_cities} eligible cities, found {len(cities)}")
    if cities.geometry.is_empty.any() or cities.geometry.isna().any():
        raise ValueError("eligible boundaries contain empty geometries")
    invalid_types = ~cities.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    if invalid_types.any():
        bad = cities.loc[invalid_types, ["id", "source_row"]].to_dict("records")
        raise ValueError(f"eligible boundaries must be polygons: {bad[:5]}")

    geod = Geod(ellps="WGS84")
    name_col = _column(cities.columns, ("city_name", "city", "name", "name_en"))
    country_col = _column(cities.columns, ("country", "country_name", "country_en", "admin0"))
    records = []
    for _, row in cities.iterrows():
        geometry, area = row.geometry, float(row[area_column])
        centroid = geometry.centroid
        perimeter = float(geod.geometry_length(geometry)) / 1000
        geodesic_area, _ = geod.geometry_area_perimeter(geometry)
        if not math.isfinite(perimeter) or perimeter <= 0:
            raise ValueError(f"city {row['id']}: invalid perimeter")
        records.append({
            "id": int(row["id"]),
            "city_name": str(row[name_col]) if name_col else "",
            "country": str(row[country_col]) if country_col else "",
            "centroid_x": float(centroid.x), "centroid_y": float(centroid.y),
            "Area": area, "Geodesic Area": abs(float(geodesic_area)) / 1e6,
            "Log Area": math.log(area), "Shape Index (MSI)": perimeter / math.sqrt(area),
            "Compactness (CI)": 4 * math.pi * area / perimeter**2,
            "Edge Density (ED)": perimeter / area, "source_row": int(row["source_row"]),
        })
    metrics = pd.DataFrame(records)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cities.to_file(filtered_path, driver="GeoJSON")
    metrics.to_csv(metrics_path, index=False)
    return cities, metrics


def initialize_earth_engine(project: str, *, authenticate: bool = False) -> Any:
    import ee
    if authenticate:
        ee.Authenticate()
    try:
        ee.Initialize(project=project)
    except Exception as error:
        raise RuntimeError("GEE initialization failed; authenticate once or set AUTHENTICATE=True") from error
    return ee


def make_ee_features(cities: Any) -> list[dict[str, Any]]:
    import ee
    return [
        {"id": int(row["id"]), "feature": ee.Feature(ee.Geometry(row.geometry.__geo_interface__), {"city_id": int(row["id"])})}
        for _, row in cities.sort_values("id").iterrows()
    ]


def _retry(call: Any, attempts: int = 5) -> Any:
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _mean_block(image: Any, block: Sequence[dict[str, Any]], bands: Sequence[str], scale: int, tile_scale: int) -> list[dict[str, Any]]:
    import ee
    collection = ee.FeatureCollection([item["feature"] for item in block])
    reduced = image.reduceRegions(collection=collection, reducer=ee.Reducer.mean(), scale=scale, tileScale=tile_scale)
    response = _retry(lambda: reduced.map(lambda feature: ee.Feature(None, feature.toDictionary())).getInfo())
    rows = {}
    for feature in response["features"]:
        props, values = feature["properties"], {}
        city_id = int(feature["properties"]["city_id"])
        for band in bands:
            value = props.get(band, props.get("mean") if len(bands) == 1 else None)
            values[band] = np.nan if value is None else float(value)
        rows[city_id] = {"id": city_id, **values}
    expected = [int(item["id"]) for item in block]
    if sorted(rows) != sorted(expected):
        raise ValueError("GEE response omitted or duplicated city IDs")
    return [rows[city_id] for city_id in expected]


def _mean_frame(image: Any, features: Sequence[dict[str, Any]], bands: Sequence[str], *, scale: int, block_size: int, max_workers: int, tile_scale: int) -> pd.DataFrame:
    blocks = [features[i:i + block_size] for i in range(0, len(features), block_size)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        jobs = {pool.submit(_mean_block, image, block, bands, scale, tile_scale): i for i, block in enumerate(blocks)}
        completed = {jobs[job]: job.result() for job in as_completed(jobs)}
    frame = pd.DataFrame([row for i in range(len(blocks)) for row in completed[i]]).sort_values("id")
    if frame["id"].astype(int).tolist() != [int(item["id"]) for item in features]:
        raise ValueError("GEE output is not aligned to city IDs")
    return frame


def extract_gee_monthly_collection(
    collection_id: str, bands: Sequence[str], features: Sequence[dict[str, Any]],
    output_dir: Path, prefix: str, *, factor: float = 1, scale: int = 1000,
    block_size: int = 10, max_workers: int = 4, tile_scale: int = 4,
    overwrite: bool = False,
) -> None:
    import ee
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ids, collection = [int(item["id"]) for item in features], ee.ImageCollection(collection_id)
    for year, month in month_pairs():
        path = output_dir / f"{prefix}_{year}_{month:02d}.csv"
        if not overwrite and _complete(path, ids):
            continue
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        image = collection.filterDate(ee.Date(f"{year}-{month:02d}-01"), ee.Date(f"{next_year}-{next_month:02d}-01")).select(list(bands)).mean()
        frame = _mean_frame(image, features, bands, scale=scale, block_size=block_size, max_workers=max_workers, tile_scale=tile_scale)
        frame[list(bands)] = frame[list(bands)].apply(pd.to_numeric, errors="coerce") * factor
        frame.to_csv(path, index=False); LOGGER.info("wrote %s", path)


def extract_gee_epoch_images(
    asset_template: str, band: str, years: Sequence[int], features: Sequence[dict[str, Any]],
    output_dir: Path, prefix: str, *, scale: int = 1000, block_size: int = 10,
    max_workers: int = 4, tile_scale: int = 4, overwrite: bool = False,
) -> None:
    import ee
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(item["id"]) for item in features]
    for year in years:
        path = output_dir / f"{prefix}_{year}.csv"
        if not overwrite and _complete(path, ids):
            continue
        image = ee.Image(asset_template.format(year=year)).select([band])
        _mean_frame(image, features, [band], scale=scale, block_size=block_size, max_workers=max_workers, tile_scale=tile_scale).to_csv(path, index=False)
        LOGGER.info("wrote %s", path)


def aggregate_land_use(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for group, classes in LAND_USE_GROUPS.items():
        if missing := [value for value in classes if value not in frame.columns]:
            raise ValueError(f"missing land-use classes: {missing}")
        frame[group] = frame[list(classes)].sum(axis=1)
    return frame


def _histogram_block(image: Any, block: Sequence[dict[str, Any]], scale: int, tile_scale: int) -> list[dict[str, Any]]:
    import ee
    collection = ee.FeatureCollection([item["feature"] for item in block])
    reduced = image.reduceRegions(collection=collection, reducer=ee.Reducer.frequencyHistogram(), scale=scale, tileScale=tile_scale)
    response = _retry(lambda: reduced.map(lambda feature: ee.Feature(None, feature.toDictionary())).getInfo())
    rows = {}
    for feature in response["features"]:
        props, city_id = feature["properties"], int(feature["properties"]["city_id"])
        histogram = props.get("histogram", props.get("built_characteristics"))
        if histogram is None:
            raise ValueError(f"city {city_id}: missing land-use histogram")
        total = float(sum(histogram.values()))
        rows[city_id] = {"id": city_id, **{code: 100 * float(histogram.get(code, 0)) / total if total else 0 for code in LAND_USE_CLASSES}}
    return [rows[int(item["id"])] for item in block]


def extract_gee_land_use(
    features: Sequence[dict[str, Any]], output_path: Path, *, scale: int = 10,
    block_size: int = 4, max_workers: int = 4, tile_scale: int = 4,
    overwrite: bool = False,
) -> None:
    import ee
    output_path = Path(output_path); ids = [int(item["id"]) for item in features]
    if not overwrite and _complete(output_path, ids):
        return
    blocks = [features[i:i + block_size] for i in range(0, len(features), block_size)]
    image = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018").select(["built_characteristics"])
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        jobs = {pool.submit(_histogram_block, image, block, scale, tile_scale): i for i, block in enumerate(blocks)}
        done = {jobs[job]: job.result() for job in as_completed(jobs)}
    frame = aggregate_land_use(pd.DataFrame([row for i in range(len(blocks)) for row in done[i]]).sort_values("id"))
    output_path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(output_path, index=False)


def extract_uhi_rasters(
    raster_root: Path, cities: Any, output_dir: Path, *, dataset: str = "AMod2",
    period: str = "Day", factor: float = 0.01, overwrite: bool = False,
) -> None:
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import transform_geom
    raster_root, output_dir = Path(raster_root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cities = cities.sort_values("id"); ids = cities["id"].astype(int).tolist()
    for year, month in month_pairs():
        output = output_dir / f"UHI_{year}_{month:02d}.csv"
        if not overwrite and _complete(output, ids):
            continue
        raster = raster_root / dataset / f"{dataset}_{period}_{year}_{month}.tif"
        if not raster.is_file():
            raise FileNotFoundError(raster)
        rows = []
        with rasterio.open(raster) as source:
            if source.crs is None:
                raise ValueError(f"{raster}: missing CRS")
            for _, row in cities.iterrows():
                geometry = transform_geom("EPSG:4326", source.crs, row.geometry.__geo_interface__)
                pixels, _ = mask(source, [geometry], crop=True, filled=False)
                values = np.ma.masked_invalid(pixels[0])
                if values.count() == 0:
                    raise ValueError(f"{raster}: city {row['id']} has no valid pixels")
                rows.append({"id": int(row["id"]), dataset: float(values.mean()) * factor})
        pd.DataFrame(rows).to_csv(output, index=False); LOGGER.info("wrote %s", output)


def reshape_uhi_by_city(monthly_dir: Path, output_dir: Path, *, expected_cities: int = 1020, dataset: str = "AMod2", overwrite: bool = False) -> None:
    monthly_dir, output_dir = Path(monthly_dir), Path(output_dir)
    frames, ids = [], list(range(expected_cities))
    for year, month in month_pairs():
        path = monthly_dir / f"UHI_{year}_{month:02d}.csv"
        frame = pd.read_csv(path)
        if frame["id"].astype(int).tolist() != ids:
            raise ValueError(f"{path}: IDs are missing or out of order")
        frame = frame[["id", dataset]].copy(); frame.insert(0, "month", month); frame.insert(0, "year", year); frames.append(frame)
    all_months = pd.concat(frames, ignore_index=True); output_dir.mkdir(parents=True, exist_ok=True)
    for city_id in ids:
        output = output_dir / f"UHI_city_{city_id}.csv"
        if output.exists() and not overwrite and len(pd.read_csv(output)) == EXPECTED_MONTHS:
            continue
        city = all_months[all_months["id"] == city_id].reset_index(drop=True)
        if len(city) != EXPECTED_MONTHS or city[dataset].isna().any():
            raise ValueError(f"city {city_id}: incomplete UHI series")
        city.to_csv(output, index=False)


def validate_raw_layout(raw_root: Path, metrics_path: Path, *, expected_cities: int = 1020) -> None:
    raw_root, ids = Path(raw_root), list(range(expected_cities))
    if pd.read_csv(metrics_path)["id"].astype(int).tolist() != ids:
        raise ValueError("city metrics IDs must be 0..1019")
    for directory, prefix in (("ClimateData", "climate"), ("MOD13A3_EVI_NDVI", "evi_ndvi"), ("PM25", "pm25")):
        for year, month in month_pairs():
            path = raw_root / directory / f"{prefix}_{year}_{month:02d}.csv"
            if not _complete(path, ids):
                raise ValueError(f"missing or misaligned: {path}")
    for city_id in ids:
        path = raw_root / "Extracted_City_Data" / f"UHI_city_{city_id}.csv"
        if not path.is_file() or len(pd.read_csv(path)) != EXPECTED_MONTHS:
            raise ValueError(f"missing or incomplete: {path}")
    for directory, prefix, years in (
        ("GHS_POP", "GHS_POP", GHS_EPOCHS), ("GHS_BUILT_S", "GHS_BUILT_S", GHS_EPOCHS),
        ("GHS_SMOD", "ghs_smod", GHS_EPOCHS), ("GHS_BUILT_H", "ghs_built_h", (2018,)),
        ("GHS_BUILT_C", "ghs_built_c", (2018,)),
    ):
        for year in years:
            path = raw_root / directory / f"{prefix}_{year}.csv"
            if not _complete(path, ids):
                raise ValueError(f"missing or misaligned: {path}")


def describe_sources() -> pd.DataFrame:
    return pd.DataFrame([{"source": name, "url": url} for name, url in SOURCE_URLS.items()])
