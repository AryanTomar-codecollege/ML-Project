import json
import re
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.config import (
    MAX_IMAGE_MB,
    MIN_CONFIDENCE,
    MIN_MARGIN,
    LLM_PROVIDER,
)
from app.ml.predictor import WoundPredictor
from app.rag.retriever import KnowledgeRetriever
from app.llm.provider import get_provider

router = APIRouter(prefix="/api", tags=["analysis"])
predictor = WoundPredictor()
retriever = KnowledgeRetriever()

LOW_CONFIDENCE = 60.0

SYSTEM_RULES = (
    "You are the guidance layer of a first-aid education prototype. "
    "The image classifier is not a diagnosis. Use ONLY supplied "
    "knowledge-base facts; do not invent treatments, dosages, materials "
    "or procedures. Use simple language. For severe or uncertain situations, "
    "recommend professional or emergency care."
)

JSON_FORMAT = '''Reply with ONLY one valid JSON object:
{
  "summary": "2 short sentences",
  "do": ["short action step"],
  "dont": ["short thing to avoid"],
  "items": [{"item": "user item", "how_to_use": "verified use"}],
  "good_signs": ["normal healing signs"],
  "warning_signs": ["signs needing medical attention"],
  "see_doctor_now": false
}
Cover every matched material. For unmatched items, say there is no verified guidance.
Set see_doctor_now true only if urgent care is likely.'''


def parse_guidance(raw_text: str):
    raw_text = re.sub(r"```(?:json)?", "", raw_text or "")
    match = re.search(r"\{.*\}", raw_text, re.S)
    if not match:
        return None

    body = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None

    clean = {
        "summary": str(data.get("summary", "")),
        "see_doctor_now": bool(data.get("see_doctor_now", False)),
    }

    for key in ["do", "dont", "good_signs", "warning_signs"]:
        value = data.get(key, [])
        clean[key] = [str(v) for v in value] if isinstance(value, list) else []

    items = data.get("items", [])
    clean["items"] = (
        [item for item in items if isinstance(item, dict) and "item" in item]
        if isinstance(items, list) else []
    )
    return clean if (clean["do"] or clean["dont"]) else None


def kb_fallback(context: dict) -> dict:
    guidance = context["injury_guidance"]
    items = []

    for material in context["matched_materials"]:
        text = " ".join(
            material.get("safe_uses", []) + material.get("avoid", [])
        )
        items.append({"item": material["item"], "how_to_use": text})

    for name in context.get("unmatched_items", []):
        items.append({
            "item": name,
            "how_to_use": "No verified guidance for this item in the knowledge base.",
        })

    first = guidance.get("immediate_steps", [""])[0]
    return {
        "summary": f"{guidance.get('what_it_is', '')} First: {first}".strip(),
        "do": guidance.get("immediate_steps", []),
        "dont": guidance.get("avoid", []),
        "items": items,
        "good_signs": guidance.get("good_signs", []),
        "warning_signs": guidance.get("warning_signs", []),
        "see_doctor_now": False,
    }


@router.get("/health")
def health():
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model_loaded": True,
    }


@router.post("/analyze")
async def analyze(image: UploadFile = File(...), items: str = Form("")):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload a valid image.")

    raw = await image.read()
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(
            413, f"Image must be smaller than {MAX_IMAGE_MB} MB."
        )

    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "The uploaded file is not a readable image.")

    prediction = await run_in_threadpool(predictor.predict, pil)

    probabilities = sorted(
        prediction["probabilities"].values(), reverse=True
    )
    margin = probabilities[0] - probabilities[1] if len(probabilities) > 1 else 100

    # Explicitly reject the two non-injury classes from the five-class model.
    if prediction["injury"] in {"no_injury", "not_known_image"}:
        return {
            "rejected": True,
            "reason": "not_injury",
            "prediction": prediction,
            "provider": LLM_PROVIDER,
            "message": (
                "This image was classified as a normal image or an unrelated "
                "image. Please upload a clear photo of a cut, burn or bruise."
            ),
        }

    if (
        prediction["confidence"] < MIN_CONFIDENCE
        or margin < MIN_MARGIN
    ):
        return {
            "rejected": True,
            "reason": "unsure",
            "prediction": prediction,
            "provider": LLM_PROVIDER,
            "message": (
                "The model is not confident enough. Please upload a clear, "
                "well-lit close-up photo of a cut, burn or bruise."
            ),
        }

    item_list = [x.strip() for x in items.split(",") if x.strip()]
    context = retriever.retrieve(prediction["injury"], item_list)

    prompt = (
        f"{SYSTEM_RULES}\n\n"
        f"Detected injury category: {prediction['injury']}\n"
        f"Classifier confidence: {prediction['confidence']}%\n\n"
        f"Knowledge base:\n{json.dumps(context, indent=2)}\n\n"
        f"User's available items: {item_list or 'Not provided'}\n\n"
        f"{JSON_FORMAT}"
    )

    guidance, structured = "", None
    try:
        provider = get_provider()
        guidance = await run_in_threadpool(provider.generate, prompt)
        structured = parse_guidance(guidance)
        if structured is None:
            guidance = await run_in_threadpool(provider.generate, prompt)
            structured = parse_guidance(guidance)
    except Exception as exc:
        print(f"LLM failed, using knowledge-base fallback: {exc}")

    source = "llm"
    if structured is None:
        structured = kb_fallback(context)
        source = "knowledge_base"

    return {
        "prediction": prediction,
        "low_confidence": prediction["confidence"] < LOW_CONFIDENCE,
        "guidance": guidance,
        "structured": structured,
        "guidance_source": source,
        "matched_materials": context["matched_materials"],
        "sources": context["sources"],
        "provider": LLM_PROVIDER,
    }
