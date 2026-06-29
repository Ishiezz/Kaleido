"""Seed the Kaleido facet registry from facets_enriched.csv.

Usage:
    python scripts/seed_registry.py [--csv PATH] [--db URL]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, "src")

from kaleido.config import Settings
from kaleido.embedding import make_encoder
from kaleido.registry import FacetRegistry


async def main() -> None:
    ap = argparse.ArgumentParser(description="Seed the Kaleido facet registry.")
    ap.add_argument("--csv", default=None, help="Path to facets_enriched.csv")
    ap.add_argument("--db", default=None, help="Database URL (overrides KALEIDO_DATABASE_URL)")
    args = ap.parse_args()

    cfg = Settings()
    db_url = args.db or cfg.database_url
    csv_path = args.csv or cfg.facets_csv_path
    encoder = make_encoder(cfg.backend, cfg.embedding_model)

    registry = FacetRegistry(db_url, encoder, cfg.registry_version)
    n = await registry.load_from_csv(csv_path)
    print(f"[seed] Loaded {n} facets into registry.")


if __name__ == "__main__":
    asyncio.run(main())
