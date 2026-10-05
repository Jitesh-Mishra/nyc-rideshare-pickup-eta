#!/usr/bin/env bash
# Rebuild data/ from the raw TLC month (download both files from
# https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page first):
#   2025 > March > High Volume For-Hire Vehicle Trip Records, and Taxi Zone Lookup Table.
# Keeps Lyft (HV0005) and the request-time-relevant columns; weekly parts stay under GitHub's 100MB limit.
set -euo pipefail
SRC=${1:-fhvhv_tripdata_2025-03.parquet}
mkdir -p data
python3 - "$SRC" <<'PY'
import sys, duckdb
src = sys.argv[1]
cols = ("request_datetime, on_scene_datetime, pickup_datetime, dropoff_datetime, PULocationID, DOLocationID, "
        "trip_miles, trip_time, base_passenger_fare, shared_request_flag, wav_request_flag")
bounds = [("2025-01-01", "2025-03-08"), ("2025-03-08", "2025-03-15"), ("2025-03-15", "2025-03-22"), ("2025-03-22", "2099-01-01")]
for i, (a, b) in enumerate(bounds, 1):
    duckdb.sql(f"COPY (SELECT {cols} FROM '{src}' WHERE hvfhs_license_num='HV0005' AND request_datetime>='{a}' "
               f"AND request_datetime<'{b}' ORDER BY request_datetime) TO 'data/lyft_2025_03_part{i}.parquet' "
               "(FORMAT parquet, COMPRESSION zstd)")
PY
cp taxi_zone_lookup.csv data/ 2>/dev/null || echo "copy taxi_zone_lookup.csv into data/"
