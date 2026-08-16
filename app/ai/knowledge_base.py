"""
Knowledge base — data layer facade.
Exposes the same dicts (TOURIST_SPOTS, CAR_TYPES, FUEL_PRICES, DISTANCES_KM,
CATEGORY_RECOMMENDATIONS) loaded from MySQL when SMART_DB_* is configured,
otherwise from the embedded dataset.
"""

import logging

from app.ai import embedded_data as _embedded

logger = logging.getLogger("smartour.data")

# Module-level structures — mutated in place at import time if the DB is used,
# so every `from app.ai.knowledge_base import ...` keeps working unchanged.
TOURIST_SPOTS = dict(_embedded.TOURIST_SPOTS)
CAR_TYPES = dict(_embedded.CAR_TYPES)
FUEL_PRICES = dict(_embedded.FUEL_PRICES)
DISTANCES_KM = dict(_embedded.DISTANCES_KM)
CATEGORY_RECOMMENDATIONS = dict(_embedded.CATEGORY_RECOMMENDATIONS)


_source = "embedded"


def data_source() -> str:
    return _source


def _apply_db_data(data: dict) -> None:
    global _source
    TOURIST_SPOTS.clear()
    TOURIST_SPOTS.update(data["TOURIST_SPOTS"])
    CAR_TYPES.clear()
    CAR_TYPES.update(data["CAR_TYPES"])
    FUEL_PRICES.clear()
    FUEL_PRICES.update(data["FUEL_PRICES"])
    DISTANCES_KM.clear()
    DISTANCES_KM.update(data["DISTANCES_KM"])
    CATEGORY_RECOMMENDATIONS.clear()
    CATEGORY_RECOMMENDATIONS.update(data["CATEGORY_RECOMMENDATIONS"])
    _source = "mysql"


def _init() -> None:
    from app.ai import db

    if not db.is_configured():
        logger.info("No SMART_DB_HOST configured — using embedded knowledge base.")
        return

    data = db.load_all_data()
    if data is None:
        logger.warning("MySQL not available — using embedded knowledge base.")
        return

    _apply_db_data(data)
    logger.info(
        "Knowledge base loaded from MySQL (%d spots, %d distances, %d car types).",
        len(data["TOURIST_SPOTS"]),
        len(data["DISTANCES_KM"]),
        len(data["CAR_TYPES"]),
    )


_init()
