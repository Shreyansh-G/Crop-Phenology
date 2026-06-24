"""
GEE – Sentinel-2 Index Fetcher + Image Downloader
==================================================
Pulls vegetation indices AND downloads satellite images directly from
Google Earth Engine for any lat/lon location over a date range.

Image types available:
  rgb         -> True colour (B4, B3, B2)
  falsecolor  -> False colour NIR composite (B8, B4, B3)
  ndvi        -> NDVI heatmap (colourised, red to green)
  geotiff     -> Raw multi-band GeoTIFF (all S2 bands)

Export destinations:
  local       -> PNG/TIF saved to --image-dir on your machine
  drive       -> Exported to your Google Drive (async GEE task)
  both        -> Local + Drive

Prerequisites:
    pip install earthengine-api pandas numpy pillow requests

Authenticate once:
    earthengine authenticate

Usage examples:
    # Indices only (no images)
    python gee_index_fetcher.py --lat 25.5620 --lon 84.8720 --start 2023-01-01 --end 2023-06-30

    # Indices + RGB and NDVI images saved locally
    python gee_index_fetcher.py --lat 25.5620 --lon 84.8720 --start 2023-01-01 --end 2023-06-30 \\
        --image-types rgb ndvi --export local

    # All image types, 16-day composites, export to Google Drive
    python gee_index_fetcher.py --lat 25.5620 --lon 84.8720 --start 2023-01-01 --end 2023-06-30 \\
        --gap 16 --image-types rgb falsecolor ndvi geotiff --export drive

    # Everything at once
    python gee_index_fetcher.py --lat 25.5620 --lon 84.8720 --start 2023-01-01 --end 2023-06-30 \\
        --gap 16 --indices NDVI EVI NDWI \\
        --image-types rgb ndvi falsecolor geotiff \\
        --export both --image-dir ./images --output bihta.csv
"""

import argparse
import os
import sys
import requests
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# INDEX REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
INDEX_REGISTRY = {
    "NDVI":  ("(NIR - RED) / (NIR + RED)",                           ["B8", "B4"]),
    "EVI":   ("2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)",   ["B8", "B4", "B2"]),
    "EVI2":  ("2.5 * (NIR - RED) / (NIR + 2.4*RED + 1)",            ["B8", "B4"]),
    "SAVI":  ("1.5 * (NIR - RED) / (NIR + RED + 0.5)",              ["B8", "B4"]),
    "MSAVI": ("(2*NIR + 1 - sqrt((2*NIR+1)**2 - 8*(NIR-RED))) / 2", ["B8", "B4"]),
    "GNDVI": ("(NIR - GREEN) / (NIR + GREEN)",                       ["B8", "B3"]),
    "NDRE":  ("(NIR - REDEDGE) / (NIR + REDEDGE)",                  ["B8", "B5"]),
    "NDWI":  ("(GREEN - NIR) / (GREEN + NIR)",                       ["B3", "B8"]),
    "NDMI":  ("(NIR - SWIR1) / (NIR + SWIR1)",                      ["B8", "B11"]),
    "NBR":   ("(NIR - SWIR2) / (NIR + SWIR2)",                      ["B8", "B12"]),
    "ARVI":  ("(NIR - (2*RED - BLUE)) / (NIR + (2*RED - BLUE))",    ["B8", "B4", "B2"]),
}
ALL_INDEX_NAMES = list(INDEX_REGISTRY.keys())

IMAGE_TYPES    = ["rgb", "falsecolor", "ndvi", "geotiff"]
EXPORT_TARGETS = ["local", "drive", "both"]

# NDVI colour ramp: red (bare/stressed) -> yellow -> green (healthy)
NDVI_PALETTE = [
    "#d73027", "#f46d43", "#fdae61", "#fee08b",
    "#ffffbf", "#d9ef8b", "#a6d96a", "#66bd63",
    "#1a9850", "#006837",
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. ARGUMENT PARSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch Sentinel-2 indices + download images from GEE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lat",   type=float, required=True, help="Latitude  (e.g. 25.5620).")
    parser.add_argument("--lon",   type=float, required=True, help="Longitude (e.g. 84.8720).")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end",   required=True, help="End date   YYYY-MM-DD.")
    parser.add_argument("--gap",   type=int, default=None,
                        help="Gap in days for composite windows. Omit = every scene.")
    parser.add_argument("--indices", nargs="+", default=ALL_INDEX_NAMES,
                        choices=ALL_INDEX_NAMES, metavar="INDEX",
                        help=f"Indices to compute. Default: all.")
    parser.add_argument("--buffer", type=int,   default=500, help="AOI buffer in metres (default 500).")
    parser.add_argument("--cloud",  type=float, default=20,  help="Max cloud %% (default 20).")
    parser.add_argument("--scale",  type=int,   default=10,  help="Pixel resolution metres (default 10).")
    parser.add_argument("--image-types", nargs="+", default=[], choices=IMAGE_TYPES, metavar="TYPE",
                        help=f"Image types to download: {IMAGE_TYPES}. Default: none (indices only).")
    parser.add_argument("--export", default="local", choices=EXPORT_TARGETS,
                        help="Export destination: local | drive | both (default: local).")
    parser.add_argument("--image-dir", default="./gee_images",
                        help="Local folder to save images (default: ./gee_images).")
    parser.add_argument("--drive-folder", default="GEE_Exports",
                        help="Google Drive folder name for exports (default: GEE_Exports).")
    parser.add_argument("--output", default=None, help="CSV output path.")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 2. GEE INIT
# ──────────────────────────────────────────────────────────────────────────────

def init_gee():
    try:
        import ee
    except ImportError:
        print("ERROR: earthengine-api not installed.\n    Run: pip install earthengine-api")
        sys.exit(1)
    try:
        ee.Initialize()
        print("OK   Google Earth Engine initialised.")
    except Exception:
        print("WARN Not authenticated. Running: earthengine authenticate ...")
        ee.Authenticate()
        ee.Initialize()
    return ee


# ──────────────────────────────────────────────────────────────────────────────
# 3. INDEX COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def add_indices(image, ee, index_names):
    s2 = image.divide(10000)
    aliases = {
        "BLUE":    s2.select("B2"),
        "GREEN":   s2.select("B3"),
        "RED":     s2.select("B4"),
        "REDEDGE": s2.select("B5"),
        "NIR":     s2.select("B8"),
        "SWIR1":   s2.select("B11"),
        "SWIR2":   s2.select("B12"),
    }
    for name in index_names:
        expr, _ = INDEX_REGISTRY[name]
        try:
            image = image.addBands(image.expression(expr, aliases).rename(name))
        except Exception as ex:
            print(f"  WARN Could not compute {name}: {ex}")
    return image


# ──────────────────────────────────────────────────────────────────────────────
# 4. VALUE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_values(image, aoi, index_names, scale, ee):
    result = image.select(index_names).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=scale, maxPixels=1e9
    )
    return result.getInfo()


# ──────────────────────────────────────────────────────────────────────────────
# 5. LOCAL IMAGE SAVE
# ──────────────────────────────────────────────────────────────────────────────

def _download_url(url, dest_path):
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)


def save_image_locally(image, aoi, label, image_types, image_dir, scale, ee):
    os.makedirs(image_dir, exist_ok=True)
    safe = label.replace(" ", "_")

    for itype in image_types:

        if itype == "rgb":
            vis = {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.3, "gamma": 1.4}
            src = image.divide(10000)
            fname = os.path.join(image_dir, f"{safe}_rgb.png")

        elif itype == "falsecolor":
            vis = {"bands": ["B8", "B4", "B3"], "min": 0.0, "max": 0.5, "gamma": 1.4}
            src = image.divide(10000)
            fname = os.path.join(image_dir, f"{safe}_falsecolor.png")

        elif itype == "ndvi":
            src = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            vis = {"bands": ["NDVI"], "min": -0.2, "max": 0.8, "palette": NDVI_PALETTE}
            fname = os.path.join(image_dir, f"{safe}_ndvi.png")

        elif itype == "geotiff":
            url = image.getDownloadURL({
                "bands": ["B2", "B3", "B4", "B5", "B8", "B11", "B12"],
                "region": aoi,
                "scale": scale,
                "format": "GEO_TIFF",
            })
            fname = os.path.join(image_dir, f"{safe}_raw.tif")
            try:
                _download_url(url, fname)
                print(f"    SAVE GeoTIFF -> {fname}")
            except Exception as ex:
                print(f"    WARN GeoTIFF download failed: {ex}")
            continue

        else:
            continue

        try:
            url = src.getThumbURL({"region": aoi, "format": "png", **vis})
            _download_url(url, fname)
            print(f"    SAVE {itype.upper()} PNG -> {fname}")
        except Exception as ex:
            print(f"    WARN {itype} save failed: {ex}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. GOOGLE DRIVE EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_to_drive(image, aoi, label, image_types, drive_folder, scale, ee):
    safe = label.replace(" ", "_").replace("-", "")

    for itype in image_types:
        if itype == "rgb":
            export_img = image.divide(10000).select(["B4", "B3", "B2"])
            desc = f"{safe}_rgb"
        elif itype == "falsecolor":
            export_img = image.divide(10000).select(["B8", "B4", "B3"])
            desc = f"{safe}_falsecolor"
        elif itype == "ndvi":
            export_img = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            desc = f"{safe}_ndvi"
        elif itype == "geotiff":
            export_img = image.select(["B2", "B3", "B4", "B5", "B8", "B11", "B12"])
            desc = f"{safe}_raw"
        else:
            continue

        task = ee.batch.Export.image.toDrive(
            image=export_img,
            description=desc,
            folder=drive_folder,
            region=aoi,
            scale=scale,
            maxPixels=1e9,
            fileFormat="GeoTIFF",
        )
        task.start()
        print(f"    DRIVE Export submitted: {desc}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. SCENE FETCH LOOPS
# ──────────────────────────────────────────────────────────────────────────────

def fetch_all_scenes(collection, aoi, args, ee):
    import pandas as pd
    records  = []
    img_list = collection.toList(collection.size())
    n        = img_list.size().getInfo()
    print(f"  Found {n} scenes. Extracting ...\n")

    for i in range(n):
        img       = ee.Image(img_list.get(i))
        date_str  = img.date().format("YYYY-MM-dd").getInfo()
        cloud_pct = img.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        img       = add_indices(img, ee, args.indices)
        vals      = extract_values(img, aoi, args.indices, args.scale, ee)

        row = {"date": date_str, "cloud_pct": round(cloud_pct, 2)}
        row.update(vals)
        records.append(row)
        print(f"  [{i+1}/{n}] {date_str}  cloud={cloud_pct:.1f}%  NDVI={vals.get('NDVI', 'N/A')}")

        if args.image_types:
            if args.export in ("local", "both"):
                save_image_locally(img, aoi, date_str, args.image_types, args.image_dir, args.scale, ee)
            if args.export in ("drive", "both"):
                export_to_drive(img, aoi, date_str, args.image_types, args.drive_folder, args.scale, ee)

    return pd.DataFrame(records)


def fetch_gap_composites(collection, aoi, args, ee):
    import pandas as pd
    records  = []
    current  = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d")
    win_n    = 0

    while current <= end_dt:
        window_end = min(current + timedelta(days=args.gap - 1), end_dt)
        s     = current.strftime("%Y-%m-%d")
        e     = window_end.strftime("%Y-%m-%d")
        label = f"{s}_to_{e}"
        win_n += 1

        sub   = collection.filterDate(s, e)
        count = sub.size().getInfo()

        if count == 0:
            print(f"  [{win_n}] {s} -> {e}  (no scenes)")
        else:
            composite = sub.median()
            composite = add_indices(composite, ee, args.indices)
            vals      = extract_values(composite, aoi, args.indices, args.scale, ee)

            row = {"period_start": s, "period_end": e, "scene_count": count}
            row.update(vals)
            records.append(row)
            print(f"  [{win_n}] {s} -> {e}  scenes={count}  NDVI={vals.get('NDVI', 'N/A')}")

            if args.image_types:
                if args.export in ("local", "both"):
                    save_image_locally(composite, aoi, label, args.image_types, args.image_dir, args.scale, ee)
                if args.export in ("drive", "both"):
                    export_to_drive(composite, aoi, label, args.image_types, args.drive_folder, args.scale, ee)

        current = window_end + timedelta(days=1)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\nGEE - Sentinel-2 Index Fetcher + Image Downloader")
    print("-" * 58)
    print(f"  Location    : lat={args.lat}, lon={args.lon}")
    print(f"  Date range  : {args.start}  ->  {args.end}")
    print(f"  Gap         : {args.gap if args.gap else 'None (every scene)'} days")
    print(f"  Indices     : {args.indices}")
    print(f"  Image types : {args.image_types if args.image_types else 'None (indices only)'}")
    if args.image_types:
        print(f"  Export to   : {args.export}")
        if args.export in ("local", "both"):
            print(f"  Image dir   : {args.image_dir}")
        if args.export in ("drive", "both"):
            print(f"  Drive folder: {args.drive_folder}")
    print(f"  Buffer      : {args.buffer} m  |  Cloud: {args.cloud}%  |  Scale: {args.scale} m")
    print()

    ee = init_gee()

    # AOI
    point = ee.Geometry.Point([args.lon, args.lat])
    aoi   = point.buffer(args.buffer)

    # Sentinel-2 SR collection
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(args.start, args.end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", args.cloud))
        .sort("system:time_start")
    )

    total = collection.size().getInfo()
    if total == 0:
        print("ERROR: No scenes found. Try relaxing --cloud or widening the date range.")
        sys.exit(1)

    print(f"  Matching scenes in GEE: {total}\n")

    # Fetch
    if args.gap:
        df = fetch_gap_composites(collection, aoi, args, ee)
    else:
        df = fetch_all_scenes(collection, aoi, args, ee)

    # Display
    import pandas as pd
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 130)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\n{'='*58}")
    print(f"  RESULTS  -  {len(df)} rows x {len(df.columns)} columns")
    print(f"{'='*58}")
    print(df.to_string(index=False))

    # Save CSV
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nCSV saved -> {args.output}")

    # Summary
    if args.image_types and args.export in ("local", "both"):
        print(f"\nImages saved -> {os.path.abspath(args.image_dir)}/")
    if args.image_types and args.export in ("drive", "both"):
        print(f"\nDrive exports submitted -> folder: '{args.drive_folder}'")
        print("Monitor at: https://code.earthengine.google.com/tasks")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
