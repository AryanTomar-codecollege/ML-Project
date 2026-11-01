"""Chunk reviewed snapshots and create the local Chroma index."""
import json, shutil, sys
from pathlib import Path
BACKEND=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BACKEND))
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import RAG_COLLECTION,RAG_DB_PATH,RAG_EMBEDDING_MODEL
SNAPSHOTS=BACKEND/'data'/'sources'/'snapshots.jsonl'; CHUNK_SIZE=1100; OVERLAP=180
def chunks(text):
    start=0
    while start<len(text):
        end=min(len(text),start+CHUNK_SIZE)
        if end<len(text):
            boundary=max(text.rfind('. ',start,end),text.rfind('; ',start,end))
            if boundary>start+400: end=boundary+1
        yield text[start:end].strip()
        if end==len(text): break
        start=end-OVERLAP
def main():
    if not SNAPSHOTS.exists(): raise FileNotFoundError('Run scripts/ingest_sources.py first.')
    records=[json.loads(x) for x in SNAPSHOTS.read_text(encoding='utf-8').splitlines() if x]
    records=[r for r in records if not r.get('needs_review',True)]
    if not records: raise RuntimeError('No approved snapshots; inspect and approve them first.')
    if RAG_DB_PATH.exists(): shutil.rmtree(RAG_DB_PATH)
    RAG_DB_PATH.mkdir(parents=True,exist_ok=True)
    # Downloads the embedding model if it is not already cached.
    model=SentenceTransformer(RAG_EMBEDDING_MODEL)
    client=chromadb.PersistentClient(path=str(RAG_DB_PATH))
    collection=client.create_collection(name=RAG_COLLECTION,metadata={'hnsw:space':'cosine'})
    docs=[]; ids=[]; metas=[]
    for record in records:
        for i,chunk in enumerate(chunks(record['text'])):
            docs.append(chunk); ids.append(f"{record['id']}-{i}")
            metas.append({'injury':record['injury'],'publisher':record['publisher'],'title':record['title'],'url':record['url'],'source_id':record['id'],'retrieved_at':record['retrieved_at']})
    vectors=model.encode_document(docs,normalize_embeddings=True,show_progress_bar=True).tolist()
    collection.add(ids=ids,documents=docs,metadatas=metas,embeddings=vectors)
    print(f'Indexed {len(docs)} passages from {len(records)} reviewed sources at {RAG_DB_PATH}')
if __name__=='__main__': main()
