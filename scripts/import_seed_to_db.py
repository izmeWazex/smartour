"""
Create the PostgreSQL knowledge-base schema and seed it from the embedded dataset.

Usage: python scripts/import_seed_to_db.py
Requires the SMART_DB_* env vars (see backend/README.md).
Runs idempotently: creates tables if missing, then truncates and reseeds.
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import postgres as db  # noqa: E402


def main() -> int:
    if not db.is_configured():
        print("Error: PostgreSQL is not configured. Set SMART_DB_HOST (and optionally")
        print("   SMART_DB_PORT / SMART_DB_USER / SMART_DB_PASSWORD / SMART_DB_NAME)")
        print("   then run again. Also install psycopg2: pip install -r requirements.txt")
        return 1

    print(f"Connecting to PostgreSQL at {db.DB_CONFIG['host']}:{db.DB_CONFIG['port']} "
          f"db={db.DB_CONFIG['dbname']} user={db.DB_CONFIG['user']} ...")
    try:
        conn = db._connect()
    except Exception as exc:
        print(f"Error: Could not connect to PostgreSQL: {exc}")
        return 1

    try:
        print("Creating tables (if missing) ...")
        db.init_schema(conn)
        print("Seeding from embedded dataset ...")
        counts = db.import_embedded_data(conn)
        print("\nDone. Loaded:")
        for table, n in counts.items():
            print(f"   {table:28s} {n} rows")
        print("\nRestart the API server and it will read from PostgreSQL "
              "(SMART_DB_HOST is set).")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
