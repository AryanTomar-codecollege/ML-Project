"""Download approved first-aid pages into snapshots for manual review."""
import json, re
from datetime import datetime, timezone
from pathlib import Path
import requests
from bs4 import BeautifulSoup
BACKEND = Path(__file__).resolve().parents[1]
SOURCES = BACKEND / 'data' / 'sources' / 'approved_sources.json'
OUTPUT = BACKEND / 'data' / 'sources' / 'snapshots.jsonl'
HEADERS = {'User-Agent': 'WoundGuide-CollegeProject/1.0 (educational RAG corpus)'}
def extract_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    for node in soup(['script','style','nav','footer','header','form','aside']): node.decompose()
    title = soup.title.get_text(' ', strip=True) if soup.title else ''
    root = soup.find('article') or soup.find('main') or soup.body or soup
    text = re.sub(r'\s+', ' ', root.get_text(' ', strip=True))
    marker = text.find('Emergency Steps')
    if marker > 0: text = text[marker:]
    for marker in (' Alternative Names ', ' References '):
        if marker in text: text = text.split(marker, 1)[0]
    return title, text
def main():
    approved = json.loads(SOURCES.read_text(encoding='utf-8')); records=[]
    for source in approved:
        print(f"Downloading {source['id']}...")
        try:
            response=requests.get(source['url'], headers=HEADERS, timeout=30); response.raise_for_status()
            title, body=extract_text(response.text)
            if len(body)<400: raise RuntimeError('Too little readable text extracted')
            records.append({**source,'title':title,'text':body,'retrieved_at':datetime.now(timezone.utc).isoformat(),'needs_review':True})
        except Exception as exc: print(f"  Skipped: {exc}")
    OUTPUT.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(f'Saved {len(records)} source snapshots to {OUTPUT}. Review every snapshot before approval.')
if __name__ == '__main__': main()
