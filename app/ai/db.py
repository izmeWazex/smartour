"""
MySQL data layer for the knowledge base.
Loads the same structures ``app/ai/embedded_data.py`` defines. Enabled only
when the SMART_DB_* env vars are set (see README). Falls back to embedded.
"""

import json
import logging
import os
from typing import Dict, List, Optional

try:
    import pymysql
    import pymysql.cursors
except ImportError:  # pragma: no cover - dependency missing
    pymysql = None

logger = logging.getLogger("smartour.data")

DB_CONFIG = {
    "host": os.getenv("SMART_DB_HOST", "localhost"),
    "port": int(os.getenv("SMART_DB_PORT", "3306")),
    "user": os.getenv("SMART_DB_USER", "root"),
    "password": os.getenv("SMART_DB_PASSWORD", ""),
    "database": os.getenv("SMART_DB_NAME", "smartour"),
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tourist_spots (
    id          VARCHAR(50)  PRIMARY KEY,
    name        VARCHAR(200) NOT NULL,
    location    VARCHAR(200) NOT NULL,
    description TEXT         NOT NULL,
    categories  JSON         NOT NULL,
    lat         DECIMAL(10,7) NOT NULL,
    lng         DECIMAL(10,7) NOT NULL,
    highlights  JSON         NOT NULL,
    best_time   VARCHAR(200)
) CHARACTER SET utf8mb4;

CREATE TABLE IF NOT EXISTS spot_distances (
    from_spot   VARCHAR(50)  NOT NULL,
    to_spot     VARCHAR(50)  NOT NULL,
    distance_km DECIMAL(10,2) NOT NULL,
    PRIMARY KEY (from_spot, to_spot)
) CHARACTER SET utf8mb4;

CREATE TABLE IF NOT EXISTS car_types (
    id                    VARCHAR(50)  PRIMARY KEY,
    label                 VARCHAR(200) NOT NULL,
    consumption_per_100km DECIMAL(6,2) NOT NULL,
    aliases               JSON         NOT NULL
) CHARACTER SET utf8mb4;

CREATE TABLE IF NOT EXISTS fuel_prices (
    fuel_type       VARCHAR(20)  PRIMARY KEY,
    price_per_liter DECIMAL(8,2) NOT NULL
) CHARACTER SET utf8mb4;

CREATE TABLE IF NOT EXISTS category_recommendations (
    category VARCHAR(50) NOT NULL,
    spot_id  VARCHAR(50) NOT NULL,
    position INT         NOT NULL,
    PRIMARY KEY (category, spot_id)
) CHARACTER SET utf8mb4;
"""


def is_configured() -> bool:
    """True when the user explicitly opted into MySQL via SMART_DB_HOST."""
    return bool(os.getenv("SMART_DB_HOST")) and pymysql is not None


def _connect():
    return pymysql.connect(
        **DB_CONFIG,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
    )


# Loading

def load_all_data() -> Optional[dict]:
    """Load the knowledge base from MySQL; None if the DB is unavailable."""
    if not is_configured():
        return None
    try:
        conn = _connect()
    except Exception as exc:
        logger.warning("MySQL connection failed (%s) — using embedded knowledge base.", exc)
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, location, description, categories, lat, lng, highlights, best_time FROM tourist_spots")
            spots_rows = cur.fetchall()

            cur.execute("SELECT from_spot, to_spot, distance_km FROM spot_distances")
            distance_rows = cur.fetchall()

            cur.execute("SELECT id, label, consumption_per_100km, aliases FROM car_types")
            car_rows = cur.fetchall()

            cur.execute("SELECT fuel_type, price_per_liter FROM fuel_prices")
            fuel_rows = cur.fetchall()

            cur.execute("SELECT category, spot_id, position FROM category_recommendations ORDER BY category, position")
            rec_rows = cur.fetchall()
    except Exception as exc:
        logger.warning("MySQL query failed (%s) — using embedded knowledge base. "
                       "Run `python scripts/import_seed_to_db.py` to create/seed the tables.", exc)
        return None
    finally:
        conn.close()

    spots: Dict[str, dict] = {}
    for r in spots_rows:
        spots[r["id"]] = {
            "name": r["name"],
            "location": r["location"],
            "description": r["description"],
            "category": json.loads(r["categories"]),
            "lat": float(r["lat"]),
            "lng": float(r["lng"]),
            "highlights": json.loads(r["highlights"]),
            "best_time": r["best_time"],
        }

    distances = {
        (r["from_spot"], r["to_spot"]): float(r["distance_km"])
        for r in distance_rows
    }

    car_types = {
        r["id"]: {
            "label": r["label"],
            "consumption_per_100km": float(r["consumption_per_100km"]),
            "aliases": json.loads(r["aliases"]),
        }
        for r in car_rows
    }

    fuel_prices = {r["fuel_type"]: float(r["price_per_liter"]) for r in fuel_rows}

    recommendations: Dict[str, List[str]] = {}
    for r in rec_rows:
        recommendations.setdefault(r["category"], []).append(r["spot_id"])

    return {
        "TOURIST_SPOTS": spots,
        "CAR_TYPES": car_types,
        "FUEL_PRICES": fuel_prices,
        "DISTANCES_KM": distances,
        "CATEGORY_RECOMMENDATIONS": recommendations,
    }


# Schema & seeding (used by scripts/import_seed_to_db.py)

def init_schema(conn) -> None:
    with conn.cursor() as cur:
        for statement in SCHEMA_SQL.split(";"):
            statement = statement.strip()
            if statement:
                cur.execute(statement)
    conn.commit()


def import_embedded_data(conn) -> dict:
    """Truncate and reload all tables from the embedded dataset."""
    from app.ai import embedded_data as data

    counts: Dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE tourist_spots")
        for sid, s in data.TOURIST_SPOTS.items():
            cur.execute(
                "INSERT INTO tourist_spots (id, name, location, description, categories, lat, lng, highlights, best_time) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (sid, s["name"], s["location"], s["description"],
                 json.dumps(s["category"]), s["lat"], s["lng"],
                 json.dumps(s["highlights"]), s.get("best_time")),
            )
        counts["tourist_spots"] = len(data.TOURIST_SPOTS)

        cur.execute("TRUNCATE TABLE spot_distances")
        for (frm, to), km in data.DISTANCES_KM.items():
            cur.execute(
                "INSERT INTO spot_distances (from_spot, to_spot, distance_km) VALUES (%s, %s, %s)",
                (frm, to, km),
            )
        counts["spot_distances"] = len(data.DISTANCES_KM)

        cur.execute("TRUNCATE TABLE car_types")
        for cid, c in data.CAR_TYPES.items():
            cur.execute(
                "INSERT INTO car_types (id, label, consumption_per_100km, aliases) VALUES (%s, %s, %s, %s)",
                (cid, c["label"], c["consumption_per_100km"], json.dumps(c["aliases"])),
            )
        counts["car_types"] = len(data.CAR_TYPES)

        cur.execute("TRUNCATE TABLE fuel_prices")
        for fuel, price in data.FUEL_PRICES.items():
            cur.execute(
                "INSERT INTO fuel_prices (fuel_type, price_per_liter) VALUES (%s, %s)",
                (fuel, price),
            )
        counts["fuel_prices"] = len(data.FUEL_PRICES)

        cur.execute("TRUNCATE TABLE category_recommendations")
        for cat, spot_ids in data.CATEGORY_RECOMMENDATIONS.items():
            for pos, sid in enumerate(spot_ids):
                cur.execute(
                    "INSERT INTO category_recommendations (category, spot_id, position) VALUES (%s, %s, %s)",
                    (cat, sid, pos),
                )
        counts["category_recommendations"] = sum(len(v) for v in data.CATEGORY_RECOMMENDATIONS.values())

    conn.commit()
    return counts
