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
    "omit it. Use clear language. Encourage professional care for severe, worsening, or uncertain injuries."
)
JSON_FORMAT = '''Return ONLY one valid JSON object with these keys:
{
  "summary": "two short sentences",
  "do": ["action supported by the retrieved passages"],
  "dont": ["avoidance supported by the retrieved passages"],
  "items": [{"item": "exact item label from item evidence", "how_to_use": "source-supported use or caution"}],
  "good_signs": ["source-supported signs"],
  "warning_signs": ["source-supported warning signs"],
  "see_doctor_now": false
}
Use injury passages only for general guidance. Use an item's own evidence only for that item's use. Do not invent missing details. Set see_doctor_now true only if the retrieved passages support urgent care.'''


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
    return clean if clean["do"] or clean["dont"] else None


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

    # The two non-injury labels must never enter the medical-guidance pipeline.
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
    # Fail closed: never generate first-aid guidance if reviewed RAG evidence is unavailable.
    if context.get("mode") != "rag" or not context.get("retrieved_chunks"):
        raise HTTPException(503, "Reviewed RAG sources are unavailable or insufficient. No guidance was generated. Check the source index and try again.")

    llm_context = {
        "detected_injury": prediction["injury"],
        "available_items": item_list,
        "injury_passages": context["retrieved_chunks"],
        "item_evidence": context["item_evidence"],
    }
    prompt = (f"{SYSTEM_RULES}\n\nDetected injury category: {prediction['injury']}\n"
              f"Classifier confidence: {prediction['confidence']}%\n\n"
              f"Retrieved source passages:\n{json.dumps(llm_context, indent=2)}\n\n"
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

    # Filter any item guidance not backed by evidence retrieved specifically for that item.
    evidence_by_item = {entry["item"].strip().casefold(): entry for entry in context.get("item_evidence", [])}
    safe_items = []
    for entry in structured["items"]:
        label = str(entry.get("item", "")).strip()
        evidence = evidence_by_item.get(label.casefold())
        if not evidence:
            continue
        links = {}
        for passage in evidence.get("passages", []):
            key = (passage.get("source", ""), passage.get("title", ""), passage.get("url", ""))
            links[key] = {"name": f"{key[0]}: {key[1]}", "url": key[2]}
        safe_items.append({"item": label, "how_to_use": str(entry.get("how_to_use", "")), "sources": list(links.values())})
    structured["items"] = safe_items

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
