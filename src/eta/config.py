"""Central config: paths, split dates, feature windows."""
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
MODEL_DIR = ROOT / "models"

TRIPS_GLOB = str(DATA_DIR / "lyft_2025_03_part*.parquet")
ZONES_CSV = DATA_DIR / "taxi_zone_lookup.csv"

# on_scene_datetime is ~99.8% null for HV0005 before 2025-03-06 and 0% null after,
# so labelled rows start on Mar 6. Earlier rows still feed the feature windows.
LABEL_START = datetime(2025, 3, 6)
VAL_START = datetime(2025, 3, 22)
TEST_START = datetime(2025, 3, 25)
TEST_END = datetime(2025, 4, 1)

WAIT_CLIP_MIN = 30.0
REQ_WINDOWS_MIN = (5, 15, 60)   # rolling request counts (demand)
DROP_WINDOWS_MIN = (15, 60)     # rolling drop-offs into zone (supply proxy)
WAIT_WINDOW_MIN = 30            # rolling mean of observed waits (driver arrivals)
MAX_WINDOW_MIN = 60
STALE_AFTER_MIN = 5             # serving falls back to baseline past this

AIRPORT_ZONES = {1, 132, 138}   # EWR, JFK, LGA
SEED = 42
