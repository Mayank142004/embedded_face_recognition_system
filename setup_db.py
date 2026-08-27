#!/usr/bin/env python3
"""
setup_db.py — One-time MongoDB schema and index setup.

Run:  python setup_db.py
"""
from db import ensure_indexes, get_db


def main():
    db = get_db()
    print(f"Connected to MongoDB database: {db.name}")

    # Ensure collections exist
    existing = db.list_collection_names()
    for col_name in ("employees", "attendance"):
        if col_name not in existing:
            db.create_collection(col_name)
            print(f"  ✅ Created collection: {col_name}")
        else:
            print(f"  ⏩ Collection already exists: {col_name}")

    # Create indexes
    ensure_indexes()
    print("\n  Indexes:")
    for col_name in ("employees", "attendance"):
        indexes = db[col_name].index_information()
        for idx_name, idx_info in indexes.items():
            print(f"    [{col_name}] {idx_name}: {idx_info['key']}")

    print("\n✅ Database setup complete.")


if __name__ == "__main__":
    main()
