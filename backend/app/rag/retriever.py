"""Reviewed-source Chroma retriever adapted for the five-class WoundGuide model."""
import json
import re
from pathlib import Path

from app.config import RAG_COLLECTION, RAG_DB_PATH, RAG_EMBEDDING_MODEL, RAG_ENABLED, RAG_MIN_SCORE

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class KnowledgeRetriever:
    def __init__(self):
        self.data = {"injuries": {}, "materials": {}, "sources": []}
        files = [DATA_DIR / "knowledge_base.json"] + sorted((DATA_DIR / "extra").glob("*.json"))
        for path in files:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                part = json.load(f)
            self.data["injuries"].update(part.get("injuries", {}))
            materials = part.get("materials", {})
            if isinstance(materials, list):
                materials = {str(m.get("name", "")).lower(): m for m in materials if isinstance(m, dict)}
            self.data["materials"].update(materials)
            self.data["sources"] += [s for s in part.get("sources", []) if s not in self.data["sources"]]

        self.collection = None
        self.embedding_model = None
        self.rag_error = None
        self._open_vector_store()

    def _open_vector_store(self):
        if not RAG_ENABLED:
            self.rag_error = "RAG_ENABLED is false."
            return
        if not RAG_DB_PATH.exists():
            self.rag_error = f"Vector database not found at {RAG_DB_PATH}; index approved sources first."
            return
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            client = chromadb.PersistentClient(path=str(RAG_DB_PATH))
            self.collection = client.get_collection(RAG_COLLECTION)
            # Runtime must use cached embeddings; no network fetch during requests.
            self.embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL, local_files_only=True)
        except Exception as exc:
            self.rag_error = str(exc)
            self.collection = None
            self.embedding_model = None

    @staticmethod
    def _has_phrase(text: str, phrase: str) -> bool:
        return bool(phrase) and re.search(rf"\b{re.escape(phrase)}\b", text, re.I) is not None

    def _materials(self, items):
        matched, unmatched = [], []
        for raw in items:
            name = raw.strip().lower()
            if not name:
                continue
            found = None
            for key, value in self.data["materials"].items():
                if not isinstance(value, dict):
                    continue
                aliases = [str(key)] + [str(a) for a in value.get("aliases", [])]
                if any(self._has_phrase(name, alias.lower()) or self._has_phrase(alias.lower(), name) for alias in aliases):
                    found = {"item": str(key), "aliases": value.get("aliases", [])}
                    break
            if found:
                if not any(m["item"] == found["item"] for m in matched):
                    matched.append(found)
            else:
                unmatched.append(raw.strip())
        return matched, unmatched

    @staticmethod
    def _query(injury, items):
        supplies = ", ".join(items) if items else "no supplies listed"
        return (f"First aid for a minor {injury}. Available supplies: {supplies}. "
                "Find safe immediate actions, what to avoid, infection prevention, and warning signs.")

    def _query_collection(self, query, where, n_results=8):
        embedding = self.embedding_model.encode_query(query, normalize_embeddings=True).tolist()
        return self.collection.query(query_embeddings=[embedding], n_results=n_results, where=where,
                                     include=["documents", "metadatas", "distances"])

    def _passages(self, result):
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        output = []
        for text, meta, distance in zip(documents[0], metadatas[0], distances[0]):
            score = round(1 - float(distance), 3)
            if score < RAG_MIN_SCORE:
                continue
            output.append({"text": text, "score": score,
                           "source": meta.get("publisher", "Unknown publisher"),
                           "title": meta.get("title", "Untitled source"),
                           "url": meta.get("url", "")})
        return output

    def retrieve(self, injury: str, items: list[str]) -> dict:
        """Returns reviewed RAG passages; never silently falls back to legacy medical guidance."""
        if injury not in {"bruise", "burn", "cut"}:
            raise ValueError(f"RAG guidance is not available for class: {injury}")
        matched, unmatched = self._materials(items)
        if not self.collection or not self.embedding_model:
            return {"mode": "unavailable", "rag_error": self.rag_error,
                    "retrieved_chunks": [], "item_evidence": [],
                    "matched_materials": matched, "unmatched_items": unmatched, "sources": []}
        try:
            query = self._query(injury, items)
            chunks = self._passages(self._query_collection(query, {"injury": injury}))
            if not chunks:
                return {"mode": "unavailable", "rag_error": "No sufficiently relevant reviewed passages found.",
                        "retrieved_chunks": [], "item_evidence": [],
                        "matched_materials": matched, "unmatched_items": unmatched, "sources": []}
            item_evidence = []
            for raw_item in items:
                q = f"First aid for {injury}: reviewed guidance about whether and how to use {raw_item}, including cautions."
                try:
                    result = self._query_collection(q, {"$or": [{"injury": injury}, {"injury": "general"}]})
                    passages = self._passages(result)
                    terms = [raw_item.lower()]
                    for key, value in self.data["materials"].items():
                        aliases = [str(key).lower()] + [str(a).lower() for a in value.get("aliases", [])] if isinstance(value, dict) else [str(key).lower()]
                        if any(self._has_phrase(raw_item.lower(), alias) or self._has_phrase(alias, raw_item.lower()) for alias in aliases):
                            terms.extend(aliases)
                    passages = [p for p in passages if any(self._has_phrase(p["text"].lower(), t) for t in terms)]
                    if passages:
                        item_evidence.append({"item": raw_item.strip(), "passages": passages})
                except Exception:
                    continue
            all_passages = chunks + [p for entry in item_evidence for p in entry["passages"]]
            source_map = {}
            for p in all_passages:
                key = (p["source"], p["title"], p["url"])
                source_map[key] = {"name": f"{key[0]}: {key[1]}", "url": key[2]}
            return {"mode": "rag", "query": query, "retrieved_chunks": chunks,
                    "item_evidence": item_evidence, "matched_materials": matched,
                    "unmatched_items": unmatched, "sources": list(source_map.values())}
        except Exception as exc:
            self.rag_error = str(exc)
            return {"mode": "unavailable", "rag_error": self.rag_error,
                    "retrieved_chunks": [], "item_evidence": [],
                    "matched_materials": matched, "unmatched_items": unmatched, "sources": []}
