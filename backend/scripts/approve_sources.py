"""Approve snapshots only after a human has inspected each source passage."""
import json, sys
from pathlib import Path
BACKEND=Path(__file__).resolve().parents[1]; SNAPSHOTS=BACKEND/'data'/'sources'/'snapshots.jsonl'
def main():
    if '--approve-reviewed' not in sys.argv: raise SystemExit('Refusing to approve automatically. Inspect snapshots, then pass --approve-reviewed.')
    records=[json.loads(x) for x in SNAPSHOTS.read_text(encoding='utf-8').splitlines() if x]
    if not records: raise SystemExit('No snapshots found. Run scripts/ingest_sources.py first.')
    # This command is an explicit human assertion that all records were reviewed.
    for r in records: r['needs_review']=False
    SNAPSHOTS.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(f'Marked {len(records)} reviewed records approved.')
if __name__=='__main__': main()
