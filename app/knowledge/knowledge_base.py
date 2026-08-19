"""
Knowledge base — data layer facade.
All data (tourist spots, distances, car types, fuel prices, category
recommendations) is loaded exclusively from PostgreSQL.
"""

import logging
import sys
from typing import Dict, List, Tuple

logger = logging.getLogger("smartour.data")

# All structures start empty — populated from PostgreSQL at import time
TOURIST_SPOTS: Dict[str, dict] = {}
CAR_TYPES: Dict[str, dict] = {}
FUEL_PRICES: Dict[str, float] = {}
DISTANCES_KM: Dict[Tuple[str, str], float] = {}
CATEGORY_RECOMMENDATIONS: Dict[str, List[str]] = {}
MUNICIPAL_DISTANCES: Dict[Tuple[str, str], float] = {}

_source = "embedded"


def data_source() -> str:
    return _source


def _apply_db_data(data: dict) -> None:
    global TOURIST_SPOTS, CAR_TYPES, FUEL_PRICES, DISTANCES_KM, CATEGORY_RECOMMENDATIONS, MUNICIPAL_DISTANCES, _source
    TOURIST_SPOTS = data["TOURIST_SPOTS"]
    CAR_TYPES = data["CAR_TYPES"]
    FUEL_PRICES = data["FUEL_PRICES"]
    DISTANCES_KM = data["DISTANCES_KM"]
    CATEGORY_RECOMMENDATIONS = data["CATEGORY_RECOMMENDATIONS"]
    MUNICIPAL_DISTANCES = data["MUNICIPAL_DISTANCES"]
    _source = "postgresql"


def _init() -> None:
    from app.db import postgres as db

    if not db.is_configured():
        logger.error(
            "SMART_DB_HOST is not set. All data must come from PostgreSQL. "
            "Set SMART_DB_HOST (and optionally SMART_DB_PORT, SMART_DB_USER, "
            "SMART_DB_PASSWORD, SMART_DB_NAME) then run "
            "`python scripts/import_seed_to_db.py` to create and seed the tables."
        )
        sys.exit(1)

    logger.info(
        "Connecting to PostgreSQL at %s:%s db=%s ...",
        db.DB_CONFIG["host"], db.DB_CONFIG["port"], db.DB_CONFIG["dbname"],
    )

    data = db.load_all_data()
    if data is None:
        logger.error(
            "Could not load data from PostgreSQL. "
            "Make sure the database is running and seeded:\n"
            "  python scripts/import_seed_to_db.py"
        )
        sys.exit(1)

    _apply_db_data(data)
    logger.info(
        "Loaded from PostgreSQL: %d spots, %d spot-distances, %d municipal-distances, "
        "%d car types, %d fuel prices, %d recommendations.",
        len(TOURIST_SPOTS),
        len(DISTANCES_KM),
        len(MUNICIPAL_DISTANCES),
        len(CAR_TYPES),
        len(FUEL_PRICES),
        sum(len(v) for v in CATEGORY_RECOMMENDATIONS.values()),
    )


_init()
