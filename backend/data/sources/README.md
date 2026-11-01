# Reviewed first-aid source snapshots

Run `python scripts/ingest_sources.py` from `backend/` to create `snapshots.jsonl`. Inspect the extracted text and source URLs manually before approving. Only then run `python scripts/approve_sources.py --approve-reviewed` and `python scripts/index_sources.py`. The indexer includes only records marked reviewed.
