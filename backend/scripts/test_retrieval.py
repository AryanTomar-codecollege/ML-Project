"""Smoke-test local retrieval without calling the LLM provider."""
import sys
from pathlib import Path
BACKEND=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(BACKEND))
from app.rag.retriever import KnowledgeRetriever
def main():
    retriever=KnowledgeRetriever(); result=retriever.retrieve('cut',['clean cloth','bandage'])
    print('Mode:',result['mode']); print('RAG error:',result.get('rag_error'))
    for i,p in enumerate(result.get('retrieved_chunks',[]),1): print(f"\n{i}. {p['source']} | score={p['score']}\n{p['text'][:400]}")
    if result['mode']!='rag': raise SystemExit('RAG not ready; check sources, embedding cache, collection and DB path.')
if __name__=='__main__': main()
