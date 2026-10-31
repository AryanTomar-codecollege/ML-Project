"""Smoke-test local semantic retrieval without calling an LLM."""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.rag.retriever import KnowledgeRetriever  # noqa: E402


def main():
    retriever = KnowledgeRetriever()
    result = retriever.retrieve('cut', ['clean cloth', 'bandage'])
    print(f"Mode: {result['mode']}")
    print(f"Query: {result.get('query', 'legacy fallback')}")
    for index, chunk in enumerate(result.get('retrieved_chunks', []), start=1):
        print(f"\n{index}. {chunk['source']} (score {chunk['score']:.2f})\n{chunk['text'][:500]}")
    if result['mode'] != 'rag':
        raise SystemExit('RAG is unavailable; item retrieval checks cannot run.')

    cases = [
        ('bruise', 'ice pack', True),
        ('bruise', 'antibacterial cream', False),
        ('cut', 'clean water', True),
    ]
    for injury, item, expected in cases:
        item_result = retriever.retrieve(injury, [item])
        evidence = [entry for entry in item_result['item_evidence'] if entry['item'] == item]
        actual = bool(evidence)
        print(f'\n{injury} + {item}: item evidence={actual} (expected {expected})')
        if actual is not expected:
            raise AssertionError(f'Unexpected item evidence for {injury} + {item}')
        if evidence:
            print('Sources:', ', '.join(sorted({passage['source'] for passage in evidence[0]['passages']})))


if __name__ == '__main__':
    main()
