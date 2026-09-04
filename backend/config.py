import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = Path(os.getenv("SMARTOUR_CSV_PATH", DATA_DIR / "raw" / "tourist_spots.csv"))
