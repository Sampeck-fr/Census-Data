"""
Census Google Sheets → Shapefile Join Pipeline (Simplified)
------------------------------------------------------------
Expects a clean CSV with:
  - Column headers: "State House District 1 (2024); Hawaii!!Total!!Estimate" etc.
  - Row headers: characteristic names (no nesting, one level only)
  - One sheet containing both house and senate districts as columns

Workflow:
  1. Load CSV
  2. Parse column headers to extract chamber, district ID, and measure type
  3. Transpose so districts become rows
  4. Split into house and senate by chamber
  5. Join each to its shapefile
  6. Export joined shapefiles

Setup:
  pip install pandas geopandas
"""

import pandas as pd
import geopandas as gpd
import re
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────

CSV_FILE = r"C:\GIS Files\Census by Legislative District\Census Tables\Citizen Population in target area - Pared down AF.csv"

SHAPEFILES = {
    "house":  r"C:\GIS Files\Census by Legislative District\geoJSON files\House.geojson",
    "senate": r"C:\GIS Files\Census by Legislative District\geoJSON files\Senate.geojson",
}

SHAPEFILE_JOIN_KEY = {
    "house":  "SLDLST",
    "senate": "SLDUST",
}

# Shapefile IDs are zero-padded to 3 digits: "001", "009", "041"
DISTRICT_ID_PAD = 3

# Only keep Estimates, drop Margins of Error — set to False to keep both
ESTIMATES_ONLY = True

OUTPUT_DIR = "output"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def parse_census_header(col: str) -> dict | None:
    """
    Parses a Census column header into chamber, district ID, and measure.

    Input:  "State Senate District 1 (2024); Hawaii!!Total!!Estimate"
    Output: {"chamber": "senate", "district_id": "001", "measure": "Estimate"}
    """
    if "!!" not in col:
        return None

    parts = col.split("!!")
    if len(parts) < 2:
        return None

    measure = parts[-1].strip()
    geo_part = col.split(";")[0]

    m = re.search(r"State (House|Senate) District (\d+)", geo_part, re.IGNORECASE)
    if not m:
        return None

    return {
        "chamber":     m.group(1).lower(),
        "district_id": str(int(m.group(2))).zfill(DISTRICT_ID_PAD),
        "measure":     measure,
    }


def sanitize_field_name(name: str, existing: list) -> str:
    name = str(name).strip()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name[:10]
    original = name
    counter = 1
    while name in existing:
        suffix = str(counter)
        name = original[:10 - len(suffix)] + suffix
        counter += 1
    existing.append(name)
    return name


def sanitize_all_columns(df: pd.DataFrame, keep_as_is: list) -> pd.DataFrame:
    used_names = list(keep_as_is)
    rename_map = {}
    for col in df.columns:
        if col in keep_as_is:
            continue
        safe = sanitize_field_name(col, used_names)
        if safe != col:
            rename_map[col] = safe
    if rename_map:
        print("\n  Field name changes (original -> shapefile-safe):")
        for orig, safe in rename_map.items():
            print(f"    '{orig}' -> '{safe}'")
    return df.rename(columns=rename_map)


# ─── CORE ─────────────────────────────────────────────────────────────────────

def load_and_transpose(filepath: str) -> dict:
    """
    Reads the CSV, parses headers, transposes to one row per district,
    and returns {"house": df, "senate": df}.
    """
    df = pd.read_csv(filepath, dtype=str)

    print(f"\n  Raw shape: {df.shape}")
    print(f"  Label column: '{df.columns[0]}'")
    print(f"  Sample district column: '{df.columns[1]}'")

    label_col = df.columns[0]
    characteristics = df[label_col].tolist()

    records = {"house": {}, "senate": {}}  # {chamber: {district_id: {char: value}}}

    for col in df.columns[1:]:
        parsed = parse_census_header(col)
        if parsed is None:
            print(f"  Skipping: '{col[:80]}'")
            continue
        if ESTIMATES_ONLY and parsed["measure"].lower() != "estimate":
            continue

        chamber = parsed["chamber"]
        dist_id = parsed["district_id"]

        if dist_id not in records[chamber]:
            records[chamber][dist_id] = {"District_ID": dist_id}

        for row_idx, char_name in enumerate(characteristics):
            value = df.iloc[row_idx][col]
            records[chamber][dist_id][char_name] = value

    result = {}
    for chamber in ["house", "senate"]:
        if not records[chamber]:
            print(f"  WARNING: No {chamber} districts found — check header format")
            result[chamber] = pd.DataFrame()
        else:
            result[chamber] = pd.DataFrame(list(records[chamber].values()))
            print(f"  {chamber.capitalize()}: {len(result[chamber])} districts, "
                  f"{len(result[chamber].columns) - 1} characteristics")

    return result


def join_to_shapefile(shp_path, census_df, shp_key, output_path):
    gdf = gpd.read_file(shp_path)

    gdf[shp_key]             = gdf[shp_key].astype(str).str.strip()
    census_df["District_ID"] = census_df["District_ID"].astype(str).str.strip()

    shp_ids    = set(gdf[shp_key])
    census_ids = set(census_df["District_ID"])

    unmatched_shp    = shp_ids - census_ids
    unmatched_census = census_ids - shp_ids
    if unmatched_shp:
        print(f"  WARNING in shapefile, NOT in census: {sorted(unmatched_shp)}")
    if unmatched_census:
        print(f"  WARNING in census, NOT in shapefile: {sorted(unmatched_census)}")
    print(f"  Matched: {len(shp_ids & census_ids)} of {len(shp_ids)} districts")

    joined = gdf.merge(census_df, left_on=shp_key, right_on="District_ID", how="left")
    if "District_ID" in joined.columns and "District_ID" != shp_key:
        joined = joined.drop(columns=["District_ID"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joined.to_file(output_path, driver="GeoJSON")
    print(f"  Exported: {output_path}")
    return joined


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  STEP 1: Loading and transposing CSV")
    print(f"{'='*60}")
    splits = load_and_transpose(CSV_FILE)

    for chamber in ["house", "senate"]:
        print(f"\n{'='*60}")
        print(f"  STEP 2: Joining — {chamber.upper()}")
        print(f"{'='*60}")

        census_df = splits[chamber]
        if census_df.empty:
            print(f"  Skipping {chamber} — no data")
            continue

        preview_path = os.path.join(OUTPUT_DIR, f"{chamber}_census.csv")
        census_df.to_csv(preview_path, index=False)
        print(f"\n  Preview saved (check before trusting join): {preview_path}")

        census_df = sanitize_all_columns(census_df, keep_as_is=["District_ID"])

        join_to_shapefile(
            shp_path    = SHAPEFILES[chamber],
            census_df   = census_df,
            shp_key     = SHAPEFILE_JOIN_KEY[chamber],
            output_path = os.path.join(OUTPUT_DIR, f"{chamber}_joined.geojson"),
        )

    print(f"\n{'='*60}")
    print(f"  Done. Outputs in '{OUTPUT_DIR}/'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()