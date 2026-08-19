"""
PostgreSQL data layer for the knowledge base.
Simplified schema: tourist_spots (with embedded municipality + categories),
spot_distances, category_recommendations, municipal_distances, car_types.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover - dependency missing
    psycopg2 = None

logger = logging.getLogger("smartour.data")

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

DB_CONFIG = {
    "host": os.getenv("SMART_DB_HOST", "localhost"),
    "port": int(os.getenv("SMART_DB_PORT", "5432")),
    "user": os.getenv("SMART_DB_USER", "postgres"),
    "password": os.getenv("SMART_DB_PASSWORD", ""),
    "dbname": os.getenv("SMART_DB_NAME", "smartour"),
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS municipalities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    latitude DECIMAL(10,7) NOT NULL,
    longitude DECIMAL(10,7) NOT NULL
);

CREATE TABLE IF NOT EXISTS tourist_spots (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    municipality_id INT NOT NULL REFERENCES municipalities(id),
    latitude DECIMAL(10,7) NOT NULL,
    longitude DECIMAL(10,7) NOT NULL,
    entrance_fee DECIMAL(10,2) DEFAULT 0.00,
    opening_hours VARCHAR(200),
    best_time VARCHAR(200),
    is_hidden_gem BOOLEAN DEFAULT FALSE,
    categories JSONB NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS spot_distances (
    from_spot VARCHAR(50) NOT NULL REFERENCES tourist_spots(id) ON DELETE CASCADE,
    to_spot VARCHAR(50) NOT NULL REFERENCES tourist_spots(id) ON DELETE CASCADE,
    distance_km DECIMAL(8,2) NOT NULL,
    PRIMARY KEY (from_spot, to_spot)
);

CREATE TABLE IF NOT EXISTS category_recommendations (
    category VARCHAR(50) NOT NULL,
    spot_id VARCHAR(50) NOT NULL REFERENCES tourist_spots(id) ON DELETE CASCADE,
    position INT NOT NULL DEFAULT 0,
    PRIMARY KEY (category, spot_id)
);

CREATE TABLE IF NOT EXISTS car_types (
    id VARCHAR(50) PRIMARY KEY,
    label VARCHAR(200) NOT NULL,
    consumption_per_100km DECIMAL(5,2) NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]',
    fuel_prices JSONB NOT NULL DEFAULT '{}'
);
"""


def is_configured() -> bool:
    """True when the user explicitly opted into PostgreSQL via SMART_DB_HOST."""
    return bool(os.getenv("SMART_DB_HOST")) and psycopg2 is not None


def _connect():
    return psycopg2.connect(**DB_CONFIG, connect_timeout=5)


# Loading

def load_all_data() -> Optional[dict]:
    """Load all knowledge-base data from PostgreSQL; None if DB unavailable."""
    if not is_configured():
        return None
    try:
        conn = _connect()
    except Exception as exc:
        logger.warning("PostgreSQL connection failed (%s).", exc)
        return None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # --- municipalities ---
            cur.execute("SELECT id, name, latitude, longitude FROM municipalities")
            muni_rows = cur.fetchall()
            municipalities = {
                r["name"]: {"lat": float(r["latitude"]), "lng": float(r["longitude"])}
                for r in muni_rows
            }

            # --- tourist spots ---
            cur.execute("""
                SELECT ts.id, ts.name, ts.description, m.name AS municipality,
                       ts.latitude, ts.longitude, ts.best_time, ts.categories
                FROM tourist_spots ts
                JOIN municipalities m ON m.id = ts.municipality_id
            """)
            spots_rows = cur.fetchall()

            # --- distances ---
            cur.execute("SELECT from_spot, to_spot, distance_km FROM spot_distances")
            distance_rows = cur.fetchall()

            # --- car types ---
            cur.execute(
                "SELECT id, label, consumption_per_100km, aliases, fuel_prices FROM car_types"
            )
            car_rows = cur.fetchall()

            # --- category recommendations ---
            cur.execute(
                "SELECT category, spot_id FROM category_recommendations "
                "ORDER BY category, position"
            )
            rec_rows = cur.fetchall()

    except Exception as exc:
        logger.warning(
            "PostgreSQL query failed (%s). Run `python scripts/import_seed_to_db.py`.",
            exc,
        )
        return None
    finally:
        conn.close()

    # Build structures matching the shapes engine.py expects

    spots: Dict[str, dict] = {}
    for r in spots_rows:
        cats = json.loads(r["categories"]) if isinstance(r["categories"], str) else r["categories"]
        spots[r["id"]] = {
            "name": r["name"],
            "location": r["municipality"],
            "description": r["description"],
            "category": cats,
            "lat": float(r["latitude"]),
            "lng": float(r["longitude"]),
            "highlights": [],
            "best_time": r["best_time"] or "",
        }

    distances: Dict[Tuple[str, str], float] = {
        (r["from_spot"], r["to_spot"]): float(r["distance_km"])
        for r in distance_rows
    }

    fuel_prices: Dict[str, float] = {}
    car_types: Dict[str, dict] = {}
    for r in car_rows:
        aliases = json.loads(r["aliases"]) if isinstance(r["aliases"], str) else r["aliases"]
        fp = json.loads(r["fuel_prices"]) if isinstance(r["fuel_prices"], str) else r["fuel_prices"]
        car_types[r["id"]] = {
            "label": r["label"],
            "consumption_per_100km": float(r["consumption_per_100km"]),
            "aliases": aliases,
        }
        if not fuel_prices and fp:
            fuel_prices = {k: float(v) for k, v in fp.items()}

    if not fuel_prices:
        fuel_prices = {"gasoline": 62.0, "diesel": 58.0, "premium": 68.0}

    recommendations: Dict[str, List[str]] = {}
    for r in rec_rows:
        recommendations.setdefault(r["category"], []).append(r["spot_id"])

    # Build municipal distances from municipality coordinates (Haversine)
    import math
    def _haversine(lat1, lng1, lat2, lng2):
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lng2 - lng1)
        a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lam/2)**2
        return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) * 1.3, 1)

    muni_names = list(municipalities.keys())
    municipal_distances: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(muni_names):
        for b in muni_names[i+1:]:
            dist = _haversine(
                municipalities[a]["lat"], municipalities[a]["lng"],
                municipalities[b]["lat"], municipalities[b]["lng"],
            )
            municipal_distances[(a, b)] = dist

    return {
        "TOURIST_SPOTS": spots,
        "CAR_TYPES": car_types,
        "FUEL_PRICES": fuel_prices,
        "DISTANCES_KM": distances,
        "CATEGORY_RECOMMENDATIONS": recommendations,
        "MUNICIPAL_DISTANCES": municipal_distances,
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
    from app.knowledge import embedded_data as data

    counts: Dict[str, int] = {}

    with conn.cursor() as cur:
        # --- municipalities ---
        municipalities: Dict[str, int] = {}  # name -> id
        all_locations = sorted({s["location"] for s in data.TOURIST_SPOTS.values()})
        for loc in all_locations:
            # Calculate center point from spots in this municipality
            spots_in_muni = [s for s in data.TOURIST_SPOTS.values() if s["location"] == loc]
            avg_lat = sum(s["lat"] for s in spots_in_muni) / len(spots_in_muni)
            avg_lng = sum(s["lng"] for s in spots_in_muni) / len(spots_in_muni)
            cur.execute(
                "INSERT INTO municipalities (name, latitude, longitude) VALUES (%s, %s, %s) RETURNING id",
                (loc, round(avg_lat, 7), round(avg_lng, 7)),
            )
            municipalities[loc] = cur.fetchone()[0]
        counts["municipalities"] = len(municipalities)

        # --- tourist_spots ---
        cur.execute("DELETE FROM tourist_spots")
        for sid, s in data.TOURIST_SPOTS.items():
            cur.execute(
                """
                INSERT INTO tourist_spots
                    (id, name, description, municipality_id, latitude, longitude,
                     best_time, categories)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sid,
                    s["name"],
                    s["description"],
                    municipalities[s["location"]],
                    s["lat"],
                    s["lng"],
                    s.get("best_time", ""),
                    json.dumps(s["category"]),
                ),
            )
        counts["tourist_spots"] = len(data.TOURIST_SPOTS)

        # --- spot_distances ---
        cur.execute("DELETE FROM spot_distances")
        for (frm, to), km in data.DISTANCES_KM.items():
            cur.execute(
                "INSERT INTO spot_distances (from_spot, to_spot, distance_km) "
                "VALUES (%s, %s, %s)",
                (frm, to, km),
            )
        counts["spot_distances"] = len(data.DISTANCES_KM)

        # --- category_recommendations ---
        cur.execute("DELETE FROM category_recommendations")
        for cat, spot_ids in data.CATEGORY_RECOMMENDATIONS.items():
            for pos, sid in enumerate(spot_ids):
                cur.execute(
                    "INSERT INTO category_recommendations (category, spot_id, position) "
                    "VALUES (%s, %s, %s)",
                    (cat, sid, pos),
                )
        counts["category_recommendations"] = sum(
            len(v) for v in data.CATEGORY_RECOMMENDATIONS.values()
        )

        # --- car_types (with embedded fuel_prices) ---
        cur.execute("DELETE FROM car_types")
        for cid, c in data.CAR_TYPES.items():
            cur.execute(
                "INSERT INTO car_types (id, label, consumption_per_100km, aliases, fuel_prices) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    cid,
                    c["label"],
                    c["consumption_per_100km"],
                    json.dumps(c["aliases"]),
                    json.dumps(data.FUEL_PRICES),
                ),
            )
        counts["car_types"] = len(data.CAR_TYPES)

    conn.commit()
    return counts


# NOTE: municipal_distances are now calculated at runtime from municipality coordinates.
# No need to seed them — they're derived from the municipalities table.
