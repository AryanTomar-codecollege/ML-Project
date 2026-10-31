"""Chunk reviewed source snapshots and store their embeddings in local ChromaDB."""
import json
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import RAG_COLLECTION, RAG_DB_PATH, RAG_EMBEDDING_MODEL
SNAPSHOTS = BACKEND / 'data' / 'sources' / 'snapshots.jsonl'
CHUNK_SIZE = 1100
OVERLAP = 180


def chunks(text: str):
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        if end < len(text):
            boundary = max(text.rfind('. ', start, end), text.rfind('; ', start, end))
            if boundary > start + 400:
                end = boundary + 1
        yield text[start:end].strip()
        if end == len(text):
            break
        start = end - OVERLAP


def reviewed_records():
    if not SNAPSHOTS.exists():
        raise FileNotFoundError('Run scripts/ingest_sources.py before indexing.')
    records = [json.loads(line) for line in SNAPSHOTS.read_text(encoding='utf-8').splitlines() if line]
    return [record for record in records if not record.get('needs_review', True)]


def main():
    records = reviewed_records()
    if not records:
        raise RuntimeError('No reviewed snapshots. Set needs_review to false only after review.')
    if RAG_DB_PATH.exists():
        shutil.rmtree(RAG_DB_PATH)
    RAG_DB_PATH.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(RAG_EMBEDDING_MODEL, local_files_only=True)
    client = chromadb.PersistentClient(path=str(RAG_DB_PATH))
    collection = client.create_collection(name=RAG_COLLECTION, metadata={'hnsw:space': 'cosine'})
    documents, ids, metadata = [], [], []
    for record in records:
        for number, text in enumerate(chunks(record['text'])):
            documents.append(text)
            ids.append(f"{record['id']}-{number}")
            metadata.append({
                'injury': record['injury'], 'publisher': record['publisher'],
                'title': record['title'], 'url': record['url'],
                'source_id': record['id'], 'retrieved_at': record['retrieved_at'],
            })
    embeddings = model.encode_document(documents, normalize_embeddings=True, show_progress_bar=True).tolist()
    collection.add(ids=ids, documents=documents, metadatas=metadata, embeddings=embeddings)
    print(f'Indexed {len(documents)} chunks from {len(records)} reviewed sources at {RAG_DB_PATH}')


if __name__ == '__main__':
    main()
