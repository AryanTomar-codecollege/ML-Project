import json
import re
from pathlib import Path

from app.config import RAG_COLLECTION, RAG_DB_PATH, RAG_EMBEDDING_MODEL, RAG_ENABLED, RAG_MIN_SCORE

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


class KnowledgeRetriever:
    """Retrieve approved first-aid passages semantically, with JSON as a safe fallback."""

    def __init__(self):
        self.data = {'injuries': {}, 'materials': {}, 'sources': []}
        files = [DATA_DIR / 'knowledge_base.json'] + sorted((DATA_DIR / 'extra').glob('*.json'))
        for path in files:
            with path.open('r', encoding='utf-8') as f:
                part = json.load(f)
            self.data['injuries'].update(part.get('injuries', {}))
            self.data['materials'].update(part.get('materials', {}))
            self.data['sources'] += [s for s in part.get('sources', []) if s not in self.data['sources']]
        self.collection = None
        self.embedding_model = None
        self.rag_error = None
        self._open_vector_store()

    def _open_vector_store(self):
        """Failure here must never stop the established fallback guidance from working."""
        if not RAG_ENABLED or not RAG_DB_PATH.exists():
            self.rag_error = 'The local vector database has not been indexed yet.'
            return
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            client = chromadb.PersistentClient(path=str(RAG_DB_PATH))
            self.collection = client.get_collection(RAG_COLLECTION)
            # The model is installed during indexing. Runtime retrieval must not
            # make an internet request or fail if the network is unavailable.
            self.embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL, local_files_only=True)
        except Exception as exc:
            self.rag_error = str(exc)
            self.collection = None

    @staticmethod
    def _has_phrase(text: str, phrase: str) -> bool:
        return re.search(rf'\b{re.escape(phrase)}\b', text) is not None

    def _materials(self, items: list[str], include_guidance: bool = False):
        materials, unmatched = [], []
        for raw in items:
            name = raw.strip().lower()
            if not name:
                continue
            found = False
            for key, value in self.data['materials'].items():
                names = [key] + value.get('aliases', [])
                if any(self._has_phrase(name, alias) for alias in names):
                    if not any(m['item'] == key for m in materials):
                        material = {'item': key, 'aliases': value.get('aliases', [])}
                        if include_guidance:
                            material.update(value)
                        materials.append(material)
                    found = True
            if not found:
                unmatched.append(raw.strip())
        return materials, unmatched

    def _legacy_context(self, injury: str, items: list[str]) -> dict:
        materials, unmatched = self._materials(items, include_guidance=True)
        return {
            'injury': injury,
            'injury_guidance': self.data['injuries'].get(injury, {}),
            'matched_materials': materials,
            'unmatched_items': unmatched,
            'sources': self.data['sources'],
        }

    @staticmethod
    def _query(injury: str, items: list[str]) -> str:
        named_items = ', '.join(items) if items else 'no supplies listed'
        return (
            f'Minor {injury} first aid. Available supplies: {named_items}. '
            'Find safe immediate actions, how to use the listed supplies, what to avoid, '
            'infection-risk reduction, and warning signs requiring professional care.'
        )

    def _rag_context(self, injury: str, items: list[str], legacy: dict) -> dict | None:
        if not self.collection or not self.embedding_model:
            return None
        query = self._query(injury, items)
        try:
            embedding = self.embedding_model.encode_query(query, normalize_embeddings=True).tolist()
            result = self.collection.query(
                query_embeddings=[embedding], n_results=8, where={'injury': injury},
                include=['documents', 'metadatas', 'distances'],
            )
        except Exception as exc:
            self.rag_error = str(exc)
            return None
        documents = result.get('documents') or [[]]
        metadatas = result.get('metadatas') or [[]]
        distances = result.get('distances') or [[]]
        chunks, seen_sources = [], set()
        for text, metadata, distance in zip(
            documents[0], metadatas[0], distances[0]
        ):
            score = round(1 - float(distance), 3)
            if score < RAG_MIN_SCORE:
                continue
            chunks.append({'text': text, 'score': score, 'source': metadata['publisher'],
                           'title': metadata['title'], 'url': metadata['url']})
            seen_sources.add((metadata['publisher'], metadata['title'], metadata['url']))
        if not chunks:
            self.rag_error = 'No sufficiently relevant reviewed source passages were found.'
            return None
        materials, unmatched = self._materials(items)
        item_evidence = []
        for item in items:
            evidence = self._retrieve_item_evidence(injury, item)
            if evidence:
                item_evidence.append({'item': item.strip(), 'passages': evidence})

        all_chunks = chunks + [
            passage for entry in item_evidence for passage in entry['passages']
        ]
        sources = {}
        for passage in all_chunks:
            key = (passage['source'], passage['title'], passage['url'])
            sources[key] = {'name': f'{key[0]}: {key[1]}', 'url': key[2]}
        return {
            'mode': 'rag', 'query': query, 'retrieved_chunks': chunks,
            'item_evidence': item_evidence,
            'matched_materials': materials, 'unmatched_items': unmatched,
            'sources': list(sources.values()),
            'injury_guidance': {}, 'fallback_context': legacy,
        }

    def _retrieve_item_evidence(self, injury: str, item: str) -> list[dict]:
        name = item.strip().lower()
        if not name or not self.collection or not self.embedding_model:
            return []

        terms = {name}
        for key, value in self.data['materials'].items():
            aliases = [key] + value.get('aliases', [])
            if any(self._has_phrase(name, alias) or self._has_phrase(alias, name) for alias in aliases):
                terms.update(alias.lower() for alias in aliases)

        query = (
            f'First aid for a {injury}: trusted source guidance on whether and how to use '
            f'{item}, including cautions.'
        )
        try:
            embedding = self.embedding_model.encode_query(query, normalize_embeddings=True).tolist()
            result = self.collection.query(
                query_embeddings=[embedding], n_results=8,
                where={'$or': [{'injury': injury}, {'injury': 'general'}]},
                include=['documents', 'metadatas', 'distances'],
            )
        except Exception as exc:
            self.rag_error = str(exc)
            return []

        documents = result.get('documents') or [[]]
        metadatas = result.get('metadatas') or [[]]
        distances = result.get('distances') or [[]]
        evidence = []
        for text, metadata, distance in zip(
            documents[0], metadatas[0], distances[0],
        ):
            score = round(1 - float(distance), 3)
            if score < RAG_MIN_SCORE or not any(self._has_phrase(text.lower(), term) for term in terms):
                continue
            evidence.append({
                'text': text, 'score': score, 'source': metadata['publisher'],
                'title': metadata['title'], 'url': metadata['url'],
            })
        return evidence

    def retrieve(self, injury: str, items: list[str]) -> dict:
        legacy = self._legacy_context(injury, items)
        rag = self._rag_context(injury, items, legacy)
        if rag:
            return rag
        legacy.update({'mode': 'knowledge_base', 'retrieved_chunks': [], 'rag_error': self.rag_error})
        return legacy
