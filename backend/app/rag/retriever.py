import json
from pathlib import Path


class KnowledgeRetriever:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(__file__).resolve().parents[2] / "data" / "knowledge_base.json"
        with self.path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)

    def retrieve(self, injury: str, items: list[str]) -> dict:
        injuries = self.data.get("injuries", {})
        guidance = injuries.get(injury)
        if guidance is None:
            raise KeyError(f"No knowledge-base entry for injury '{injury}'")

        materials = self.data.get("materials", [])
        matched, unmatched = [], []

        for requested in items:
            needle = requested.lower().strip()
            found = None
            for material in materials:
                names = [material.get("name", "")] + material.get("aliases", [])
                if any(needle == n.lower() for n in names if n):
                    found = material
                    break
            if found:
                matched.append({
                    "item": requested,
                    "safe_uses": found.get("safe_uses", []),
                    "avoid": found.get("avoid", []),
                })
            else:
                unmatched.append(requested)

        return {
            "injury_guidance": guidance,
            "matched_materials": matched,
            "unmatched_items": unmatched,
            "sources": self.data.get("sources", []),
        }
