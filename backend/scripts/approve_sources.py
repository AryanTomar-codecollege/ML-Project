"""Mark inspected snapshots as approved for indexing.

Run only after a human has checked every record in snapshots.jsonl.
"""
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
SNAPSHOTS = BACKEND / 'data' / 'sources' / 'snapshots.jsonl'


def main():
    if '--approve-reviewed' not in sys.argv:
        raise SystemExit('Refusing to approve automatically. Run with --approve-reviewed after inspection.')
    records = [json.loads(line) for line in SNAPSHOTS.read_text(encoding='utf-8').splitlines() if line]
    if not records:
        raise SystemExit('No snapshots found. Run ingest_sources.py first.')
    for record in records:
        record['needs_review'] = False
    SNAPSHOTS.write_text(
        ''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records), encoding='utf-8'
    )
    print(f'Approved {len(records)} inspected snapshots for indexing.')


if __name__ == '__main__':
    main()
