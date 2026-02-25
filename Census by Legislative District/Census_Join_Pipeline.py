"""
Census Google Sheets → Shapefile Join Pipeline
-----------------------------------------------
Handles tables where District IDs contain "House"/"Senate" prefixes,
splitting rows into the correct chamber automatically after transpose.

Workflow:
  1. Load CSVs (all 4 tables, mixed house + senate districts as columns)
  2. Transpose so District IDs become rows
  3. Split rows into house vs senate by District_ID prefix
  4. Clean District_IDs (strip chamber label) and field names
  5. Join each chamber's data to its shapefile
  6. Export joined shapefiles ready for Leaflet

Setup:
  pip install pandas geopandas
"""

import pandas as pd
import geopandas as gpd
import re
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# All four census CSVs — mixed house + senate, will be split automatically
CSV_FILES = [
    "data/table1.csv",
    "data/table2.csv",
    "data/table3.csv",
    "data/table4.csv",
]

# Shapefiles for each chamber
SHAPEFILES = {
    "house":  "shapefiles/house_districts.shp",
    "senate": "shapefiles/senate_districts.shp",
}

# The field name in each shapefile's attribute table that holds the bare district number
SHAPEFILE_JOIN_KEY = {
    "house":  "DISTRICT",   # check in QGIS — adjust if different
    "senate": "DISTRICT",
}

# The string patterns that identify each chamber in the District_ID column.
# After transpose, your district IDs might look like:
#   "House District 1", "HD 1", "H-01", "Senate 5", "SD-05", etc.
# Adjust these regex patterns to match whatever's actually in your headers.
CHAMBER_PATTERNS = {
    "house":  r"(?i)house|(?i)\bHD\b|(?i)\bH-",
    "senate": r"(?i)senate|(?i)\bSD\b|(?i)\bS-",
}

# After identifying the chamber, strip everything non-numeric to get a bare district number.
# "House District 1" -> "1", "HD-01" -> "01"
# Set STRIP_LEADING_ZEROS = True if your shapefile IDs are "1", "2" not "01", "02"
STRIP_LEADING_ZEROS = True

# Output directory
OUTPUT_DIR = "output"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

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


def extract_district_number(district_id: str) -> str:
    """Strip everything non-numeric from a district ID string."""
    num = re.sub(r"[^0-9]", "", str(district_id))
    if STRIP_LEADING_ZEROS and num:
        num = str(int(num))
    return num


def transpose_csv(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV where row 0 = characteristics, columns = districts.
    Returns transposed DataFrame with District_ID as first column.
    """
    df = pd.read_csv(filepath, dtype=str)

    print(f"\n  Raw shape: {df.shape} — {os.path.basename(filepath)}")
    print(f"  First column name: '{df.columns[0]}'")
    print(f"  District columns (first 5): {list(df.columns[1:6])}")

    # Set characteristic column as index, then transpose
    df = df.set_index(df.columns[0])
    df = df.T
    df.index.name = "District_ID"
    df = df.reset_index()

    print(f"  Transposed shape: {df.shape}")
    print(f"  District_IDs (first 5): {list(df['District_ID'][:5])}")

    return df


def split_by_chamber(df: pd.DataFrame) -> dict:
    """
    Splits a transposed DataFrame into house and senate subsets
    based on District_ID string matching CHAMBER_PATTERNS.
    Strips the chamber label and returns bare district numbers.
    """
    result = {}

    for chamber, pattern in CHAMBER_PATTERNS.items():
        mask = df["District_ID"].str.contains(pattern, regex=True, na=False)
        subset = df[mask].copy()

        if subset.empty:
            print(f"  WARNING: No rows matched '{chamber}' pattern — check CHAMBER_PATTERNS in config")
        else:
            subset["District_ID"] = subset["District_ID"].apply(extract_district_number)
            print(f"  {chamber.capitalize()}: {len(subset)} districts "
                  f"(IDs: {list(subset['District_ID'][:5])} ...)")

        result[chamber] = subset

    # Warn about any rows that matched neither chamber
    matched_mask = df["District_ID"].str.contains(
        "|".join(CHAMBER_PATTERNS.values()), regex=True, na=False
    )
    unmatched_rows = df[~matched_mask]
    if not unmatched_rows.empty:
        print(f"  WARNING: Rows that didn't match any chamber pattern: {list(unmatched_rows['District_ID'])}")

    return result


def merge_tables_for_chamber(all_splits: list, chamber: str) -> pd.DataFrame:
    """
    Merges the chamber-specific slices from each CSV table into one DataFrame.
    all_splits is a list of dicts: [{chamber: df}, {chamber: df}, ...]
    """
    merged = None
    for splits in all_splits:
        df = splits[chamber]
        if df.empty:
            continue
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="District_ID", how="outer", suffixes=("", "_dup"))
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            if dup_cols:
                print(f"  Dropping duplicate columns after merge: {dup_cols}")
            merged = merged.drop(columns=dup_cols)

    return merged


def join_to_shapefile(shp_path, census_df, shp_key, output_path):
    gdf = gpd.read_file(shp_path)

    print(f"\n  Shapefile fields: {list(gdf.columns)}")

    # Normalize both join keys to stripped strings
    gdf[shp_key]             = gdf[shp_key].astype(str).str.strip()
    census_df["District_ID"] = census_df["District_ID"].astype(str).str.strip()

    if STRIP_LEADING_ZEROS:
        gdf[shp_key] = gdf[shp_key].apply(
            lambda x: str(int(x)) if x.isdigit() else x
        )

    # Mismatch report before joining
    shp_ids    = set(gdf[shp_key])
    census_ids = set(census_df["District_ID"])
    unmatched_shp    = shp_ids - census_ids
    unmatched_census = census_ids - shp_ids

    if unmatched_shp:
        print(f"  WARNING: In shapefile, NOT in census: {sorted(unmatched_shp)}")
    if unmatched_census:
        print(f"  WARNING: In census, NOT in shapefile: {sorted(unmatched_census)}")

    matched = shp_ids & census_ids
    print(f"  Matched: {len(matched)} of {len(shp_ids)} districts")

    joined = gdf.merge(census_df, left_on=shp_key, right_on="District_ID", how="left")

    if "District_ID" in joined.columns and "District_ID" != shp_key:
        joined = joined.drop(columns=["District_ID"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joined.to_file(output_path)
    print(f"  Exported: {output_path}")

    return joined


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Transpose all CSVs and split each into house/senate
    print(f"\n{'='*60}")
    print(f"  STEP 1: Transposing CSVs and splitting by chamber")
    print(f"{'='*60}")

    all_splits = []
    for fp in CSV_FILES:
        print(f"\n--- {os.path.basename(fp)} ---")
        df = transpose_csv(fp)
        splits = split_by_chamber(df)
        all_splits.append(splits)

    # Step 2: Merge tables per chamber and join to shapefiles
    for chamber in ["house", "senate"]:
        print(f"\n{'='*60}")
        print(f"  STEP 2: Merging + joining — {chamber.upper()}")
        print(f"{'='*60}")

        census_df = merge_tables_for_chamber(all_splits, chamber)

        if census_df is None or census_df.empty:
            print(f"  WARNING: No data for {chamber} — skipping shapefile join")
            continue

        # Save merged census table for inspection before trusting the join
        preview_path = os.path.join(OUTPUT_DIR, f"{chamber}_census_merged.csv")
        census_df.to_csv(preview_path, index=False)
        print(f"\n  Merged census table saved (check this first): {preview_path}")

        # Sanitize field names for shapefile compatibility
        census_df = sanitize_all_columns(census_df, keep_as_is=["District_ID"])

        # Join to shapefile
        output_shp = os.path.join(OUTPUT_DIR, f"{chamber}_districts_joined.shp")
        join_to_shapefile(
            shp_path    = SHAPEFILES[chamber],
            census_df   = census_df,
            shp_key     = SHAPEFILE_JOIN_KEY[chamber],
            output_path = output_shp,
        )

    print(f"\n{'='*60}")
    print(f"  Done. Output files in '{OUTPUT_DIR}/'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()