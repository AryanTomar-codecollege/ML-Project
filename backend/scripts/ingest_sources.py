"""Download only the approved first-aid pages into reviewable JSONL snapshots."""
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BACKEND = Path(__file__).resolve().parents[1]
SOURCES = BACKEND / 'data' / 'sources' / 'approved_sources.json'
OUTPUT = BACKEND / 'data' / 'sources' / 'snapshots.jsonl'
HEADERS = {'User-Agent': 'WoundGuide-CollegeProject/1.0 (educational RAG corpus)'}


def extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, 'html.parser')
    for node in soup(['script', 'style', 'nav', 'footer', 'header', 'form', 'aside']):
        node.decompose()
    title = soup.title.get_text(' ', strip=True) if soup.title else ''
    root = soup.find('article') or soup.find('main') or soup.body or soup
    text = root.get_text(' ', strip=True)
    text = re.sub(r'\s+', ' ', text)
    # Red Cross pages place promotional banners before the first-aid article.
    # Keep only the clinical content when that stable article heading exists.
    marker = text.find('Emergency Steps')
    if marker > 0:
        text = text[marker:]
    # Medical Encyclopedia pages append navigation, image labels and references
    # after the clinical article. They are not useful retrieval evidence.
    for marker in (' Alternative Names ', ' References '):
        if marker in text:
            text = text.split(marker, 1)[0]
    return title, text


def main():
    approved = json.loads(SOURCES.read_text(encoding='utf-8'))
    records = []
    failures = []
    for source in approved:
        print(f"Downloading {source['id']}...")
        try:
            response = requests.get(source['url'], headers=HEADERS, timeout=30)
            response.raise_for_status()
            title, text = extract_text(response.text)
            if len(text) < 400:
                raise RuntimeError('Too little readable text was extracted')
            records.append({
                **source,
                'title': title,
                'text': text,
                'retrieved_at': datetime.now(UTC).isoformat(),
                'needs_review': True,
            })
        except (requests.RequestException, RuntimeError) as exc:
            failures.append({'id': source['id'], 'url': source['url'], 'error': str(exc)})
            print(f"  Skipped: {exc}")
    with OUTPUT.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f'Saved {len(records)} snapshots to {OUTPUT}')
    if failures:
        print(f'Skipped {len(failures)} source(s); see the messages above and use a permitted export/manual source file if needed.')
    print('Review snapshots.jsonl, change needs_review to false, then run index_sources.py.')


if __name__ == '__main__':
    main()
