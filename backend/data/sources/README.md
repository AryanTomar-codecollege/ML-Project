# Approved first-aid sources

`approved_sources.json` deliberately contains only pages about cuts, burns, and
bruises from the four organizations already cited by WoundGuide.

Run `python scripts/ingest_sources.py` from `backend/` to download a fresh,
human-readable snapshot into `data/sources/snapshots.jsonl`. Then review the
snapshots before indexing them:

```powershell
python scripts/ingest_sources.py
python scripts/approve_sources.py --approve-reviewed
python scripts/index_sources.py
python scripts/test_retrieval.py
```

The ingestion script does **not** silently index new web content. It marks every
snapshot `needs_review: true`. Set it to `false` only after checking that the
extracted text is relevant, current, and not navigation or unrelated material.
The indexer accepts only reviewed snapshots. This keeps the medical corpus
curated and auditable.
