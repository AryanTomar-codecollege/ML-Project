import json
import re
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.config import MAX_IMAGE_MB, MIN_CONFIDENCE, MIN_MARGIN, LLM_PROVIDER, CLASS_NAMES
from app.ml.predictor import WoundPredictor
from app.rag.retriever import KnowledgeRetriever
from app.llm.provider import get_provider

router = APIRouter(prefix="/api", tags=["analysis"])
predictor = WoundPredictor()
retriever = KnowledgeRetriever()

SYSTEM_RULES = (
    "You are the wording layer of an educational first-aid prototype, not a clinician. "
    "Use ONLY the retrieved reviewed-source passages supplied below. Do not use prior knowledge "
    "or invent treatments, doses, materials, or procedures. If the sources do not support a claim, "
    "omit it. Use clear language. Encourage professional care for severe, worsening, or uncertain injuries. "
    "Every summary or guidance claim must cite one or more evidence IDs that directly support it. "
    "Never invent an evidence ID. If no supplied passage supports a claim, omit that claim."
)
JSON_FORMAT = """Return ONLY one valid JSON object with these keys:
{
  "summary": "two short sentences, supported by supplied evidence",
  "do": ["action supported by retrieved passages"],
  "dont": ["avoidance supported by retrieved passages"],
  "items": [{"item": "exact item label from item evidence", "how_to_use": "source-supported use or caution"}],
  "good_signs": ["source-supported signs"],
  "warning_signs": ["source-supported warning signs"],
  "see_doctor_now": false,
  "citations": {
    "summary": ["S1"],
    "do": [["S1"], ["S2"]],
    "dont": [["S1"]],
    "good_signs": [["S2"]],
    "warning_signs": [["S1"]],
    "items": [["S3"]]
  }
}
Citation rules:
- `summary` citations is a flat list of evidence IDs.
- For `do`, `dont`, `good_signs`, and `warning_signs`, citations are arrays parallel to the corresponding text arrays. Each inner array lists IDs supporting that exact bullet.
- `items` citations are arrays parallel to `items`; use only item-specific evidence IDs that support that item's instructions.
- Use only IDs present in the supplied passages. Do not cite a source merely because it is generally relevant.
- Use injury passages for general guidance. Use an item's own evidence only for that item's use.
- If a claim has no directly supporting passage, omit it rather than inventing a citation. If no citation is appropriate, use an empty list.
- Set see_doctor_now true only if retrieved passages support urgent care."""


def parse_guidance(text: str):
    text = re.sub(r"```(?:json)?", "", text or "")
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    body = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    clean = {"summary": str(data.get("summary", "")),
             "see_doctor_now": bool(data.get("see_doctor_now", False))}
    for key in ("do", "dont", "good_signs", "warning_signs"):
        value = data.get(key, [])
        clean[key] = [str(v) for v in value] if isinstance(value, list) else []
    values = data.get("items", [])
    clean["items"] = [x for x in values if isinstance(x, dict) and x.get("item") and x.get("how_to_use")] if isinstance(values, list) else []
    citations = data.get("citations", {})
    clean["citations"] = citations if isinstance(citations, dict) else {}
    return clean if clean["do"] or clean["dont"] else None


def _passage_key(passage):
    return (passage.get("source", ""), passage.get("title", ""), passage.get("url", ""), passage.get("text", ""))


def _source_object(passage):
    return {"name": f"{passage.get('source', 'Unknown publisher')}: {passage.get('title', 'Untitled source')}",
            "title": passage.get("title", "Untitled source"),
            "url": passage.get("url", "")}


def _tag_evidence(context):
    # Assign stable per-response IDs to passages and return lookup/allowed-ID sets.
    catalog, ids_by_key = {}, {}
    counter = 0

    def tag(passage):
        nonlocal counter
        key = _passage_key(passage)
        if key not in ids_by_key:
            counter += 1
            evidence_id = f"S{counter}"
            ids_by_key[key] = evidence_id
            catalog[evidence_id] = {**passage, "id": evidence_id}
        return catalog[ids_by_key[key]]

    injury_passages = [tag(p) for p in context.get("retrieved_chunks", [])]
    item_evidence = []
    item_allowed = {}
    for entry in context.get("item_evidence", []):
        tagged = [tag(p) for p in entry.get("passages", [])]
        item_evidence.append({"item": entry.get("item", ""), "passages": tagged})
        item_allowed[entry.get("item", "").strip().casefold()] = {p["id"] for p in tagged}
    injury_allowed = {p["id"] for p in injury_passages}
    return injury_passages, item_evidence, catalog, injury_allowed, item_allowed


def _normalise_id_list(value, allowed, catalog):
    if not isinstance(value, list):
        return []
    result, seen = [], set()
    for evidence_id in value:
        if isinstance(evidence_id, str) and evidence_id in allowed and evidence_id in catalog and evidence_id not in seen:
            url = catalog[evidence_id].get("url", "")
            if url.startswith("https://") or url.startswith("http://"):
                result.append(_source_object(catalog[evidence_id]))
                seen.add(evidence_id)
    return result


def _validated_citations(raw, structured, injury_allowed, item_allowed, catalog):
    raw = raw if isinstance(raw, dict) else {}
    output = {"summary": _normalise_id_list(raw.get("summary", []), injury_allowed, catalog)}
    for key in ("do", "dont", "good_signs", "warning_signs"):
        raw_rows = raw.get(key, [])
        raw_rows = raw_rows if isinstance(raw_rows, list) else []
        output[key] = [_normalise_id_list(raw_rows[i] if i < len(raw_rows) else [], injury_allowed, catalog)
                       for i in range(len(structured.get(key, [])))]
    item_rows = raw.get("items", [])
    item_rows = item_rows if isinstance(item_rows, list) else []
    output["items"] = []
    for i, item in enumerate(structured.get("items", [])):
        allowed = item_allowed.get(str(item.get("item", "")).strip().casefold(), set())
        ids = item_rows[i] if i < len(item_rows) else []
        output["items"].append(_normalise_id_list(ids, allowed, catalog))
    return output


@router.get("/health")
def health():
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model_loaded": predictor.model is not None,
        "rag_mode": "ready" if retriever.collection is not None and retriever.embedding_model is not None else "unavailable",
        "rag_error": retriever.rag_error,
        "classes": CLASS_NAMES,
    }


@router.post("/analyze")
async def analyze(image: UploadFile = File(...), items: str = Form("")):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload a valid image.")
    raw = await image.read()
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image must be smaller than {MAX_IMAGE_MB} MB.")
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The uploaded file is not a readable image.")

    prediction = await run_in_threadpool(predictor.predict, pil)
    if prediction.get("injury") not in CLASS_NAMES:
        raise HTTPException(500, "Model returned a class that is not in the configured class list.")

    # Non-injury labels must never enter the medical-guidance pipeline.
    if prediction["injury"] in {"no_injury", "not_known_image"}:
        return {
            "rejected": True, "reason": "not_injury", "prediction": prediction,
            "provider": LLM_PROVIDER,
            "message": "This image was classified as having no visible injury or as an unrelated image. Please upload a clear photo of a cut, burn, or bruise.",
        }

    sorted_probs = sorted(prediction["probabilities"].values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 100.0
    if prediction["confidence"] < MIN_CONFIDENCE or margin < MIN_MARGIN:
        return {
            "rejected": True, "reason": "unsure", "prediction": prediction,
            "provider": LLM_PROVIDER,
            "message": "The model is uncertain. Please upload a clear, well-lit photo. This model is an educational prototype, not a diagnosis.",
        }

    item_list = [part.strip() for part in items.split(",") if part.strip()]
    context = await run_in_threadpool(retriever.retrieve, prediction["injury"], item_list)
    if context.get("mode") != "rag" or not context.get("retrieved_chunks"):
        raise HTTPException(503, "Reviewed RAG sources are unavailable or insufficient. No guidance was generated. Check the source index and try again.")

    injury_passages, tagged_item_evidence, catalog, injury_allowed, item_allowed = _tag_evidence(context)
    llm_context = {
        "detected_injury": prediction["injury"],
        "available_items": item_list,
        "injury_passages": injury_passages,
        "item_evidence": tagged_item_evidence,
        "citation_instruction": "Cite the exact evidence IDs in the citations object. IDs are not interchangeable; cite only passages that support the associated claim.",
    }
    prompt = (f"{SYSTEM_RULES}\n\nDetected injury category: {prediction['injury']}\n"
              f"Classifier confidence: {prediction['confidence']}%\n\n"
              f"Retrieved source passages with evidence IDs:\n{json.dumps(llm_context, indent=2)}\n\n"
              f"User's available items: {item_list or 'Not provided'}\n\n{JSON_FORMAT}")
    try:
        provider = get_provider()
        raw_guidance = await run_in_threadpool(provider.generate, prompt)
        structured = parse_guidance(raw_guidance)
        if structured is None:
            raw_guidance = await run_in_threadpool(provider.generate, prompt)
            structured = parse_guidance(raw_guidance)
    except Exception as exc:
        print(f"RAG guidance generation failed: {exc}")
        raise HTTPException(502, "The configured LLM provider failed. No fallback medical guidance was generated.") from exc
    if structured is None:
        raise HTTPException(502, "The provider did not return valid structured guidance. No fallback guidance was generated.")

    raw_citations = structured.pop("citations", {})
    structured["citations"] = _validated_citations(raw_citations, structured, injury_allowed, item_allowed, catalog)

    # Keep item evidence explicitly tied to the selected item's retrieved evidence.
    # The model's cited IDs are validated against that item's allowed evidence IDs.
    safe_items = []
    raw_item_citations = raw_citations.get("items", []) if isinstance(raw_citations, dict) else []
    for index, entry in enumerate(structured["items"]):
        label = str(entry.get("item", "")).strip()
        allowed = item_allowed.get(label.casefold(), set())
        cited_ids = raw_item_citations[index] if isinstance(raw_item_citations, list) and index < len(raw_item_citations) else []
        sources = _normalise_id_list(cited_ids, allowed, catalog)
        if not sources:
            # No verified per-claim citation: do not show an unrelated/global source list.
            continue
        safe_items.append({"item": label, "how_to_use": str(entry.get("how_to_use", "")), "sources": sources})
    structured["items"] = safe_items

    # The global references list remains for traceability, but per-claim citations are authoritative in the UI.
    return {
        "prediction": prediction,
        "low_confidence": prediction["confidence"] < 60.0,
        "guidance": raw_guidance,
        "structured": structured,
        "guidance_source": "rag",
        "matched_materials": context["matched_materials"],
        "unmatched_items": context["unmatched_items"],
        "sources": context["sources"],
        "provider": LLM_PROVIDER,
    }
