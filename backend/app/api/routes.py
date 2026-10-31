import json
import os
import re
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.config import MAX_IMAGE_MB, LLM_PROVIDER
from app.ml.predictor import WoundPredictor
from app.rag.retriever import KnowledgeRetriever
from app.llm.provider import get_provider

router = APIRouter(prefix='/api', tags=['analysis'])
predictor = WoundPredictor()
retriever = KnowledgeRetriever()

LOW_CONFIDENCE = 60.0  # below this %, the UI shows a caution
# "Not an injury photo" gate. Photos below these limits are rejected instead of being forced into a class.
# Tune them in backend/.env (MIN_CONFIDENCE, MIN_MARGIN) after testing a few of your own photos.
MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '55'))  # top class must be at least this %
MIN_MARGIN = float(os.getenv('MIN_MARGIN', '12'))          # and beat the 2nd class by this many points

SYSTEM_RULES = '''You are the guidance layer of a first-aid education prototype. The image classifier is not a diagnosis. Use ONLY the retrieved source passages; do not use prior knowledge or invent treatments, dosages, materials, or procedures. Keep language simple. If the passages do not support guidance, omit it. For severe or uncertain situations, recommend professional/emergency care only when supported by the passages.'''

JSON_FORMAT = '''Reply with ONLY one valid JSON object (no markdown, no text outside it) with exactly these keys:
{
  "summary": "2 short sentences: what this category means and the first thing to do",
  "do": ["short action step", "..."],
  "dont": ["short thing to avoid", "..."],
    "items": [{"item": "exact item name from item evidence", "how_to_use": "use or caution supported by that item's passages"}],
  "good_signs": ["signs that healing is going normally"],
  "warning_signs": ["signs that need a doctor / emergency care"],
  "see_doctor_now": false
}
Only use injury passages for general injury guidance. For an item, use only passages in that item's item_evidence group; include it only when those passages directly support the stated use or caution. Use the exact item label from the group. Omit items without evidence and do not add placeholder or generic replies. Set "see_doctor_now" to true only when retrieved passages support urgent care.'''


def parse_guidance(text: str):
    """Read the LLM reply as JSON (tolerates ```json fences and trailing commas). None if unusable."""
    text = re.sub(r'```(?:json)?', '', text or '')
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        return None
    body = re.sub(r',\s*([}\]])', r'\1', match.group(0))
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    clean = {'summary': str(data.get('summary', '')), 'see_doctor_now': bool(data.get('see_doctor_now', False))}
    for key in ['do', 'dont', 'good_signs', 'warning_signs']:
        value = data.get(key, [])
        clean[key] = [str(v) for v in value] if isinstance(value, list) else []
    items = data.get('items', [])
    clean['items'] = [i for i in items if isinstance(i, dict) and 'item' in i] if isinstance(items, list) else []
    return clean if (clean['do'] or clean['dont']) else None


@router.get('/health')
def health():
    return {'status': 'ok', 'provider': LLM_PROVIDER, 'model_loaded': True}


@router.post('/analyze')
async def analyze(image: UploadFile = File(...), items: str = Form('')):
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(400, 'Please upload a valid image.')
    raw = await image.read()
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f'Image must be smaller than {MAX_IMAGE_MB} MB.')
    try:
        pil = Image.open(BytesIO(raw)).convert('RGB')
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, 'The uploaded file is not a readable image.')

    prediction = await run_in_threadpool(predictor.predict, pil)

    # Gate: reject photos that are not clearly one of the supported injuries
    top = sorted(prediction['probabilities'].values(), reverse=True)
    margin = top[0] - top[1] if len(top) > 1 else 100
    reason = None
    if prediction['injury'] == 'not_wound':  # only exists after retraining with the 4th class
        reason = 'not_wound'
    elif prediction['confidence'] < MIN_CONFIDENCE or margin < MIN_MARGIN:
        reason = 'unsure'
    if reason:
        return {
            'rejected': True, 'reason': reason, 'prediction': prediction, 'provider': LLM_PROVIDER,
            'message': 'Please upload a clear, well-lit, close-up photo of a cut, burn or bruise on the skin.',
        }
    item_list = [x.strip() for x in items.split(',') if x.strip()]
    context = retriever.retrieve(prediction['injury'], item_list)
    if context['mode'] != 'rag':
        raise HTTPException(503, 'RAG is unavailable; no fallback guidance was generated.')

    llm_context = {
        'detected_injury': prediction['injury'],
        'available_items': item_list,
        'injury_passages': context['retrieved_chunks'],
        'item_evidence': context['item_evidence'],
    }

    prompt = (
        f"{SYSTEM_RULES}\n\nDetected injury category: {prediction['injury']}\n"
        f"Classifier confidence: {prediction['confidence']}%\n\n"
        f"Retrieved source passages:\n{json.dumps(llm_context, indent=2)}\n\n"
        f"User's available items: {item_list or 'Not provided'}\n\n{JSON_FORMAT}"
    )

    try:
        guidance = await run_in_threadpool(get_provider().generate, prompt)
        structured = parse_guidance(guidance)
        if structured is None:  # one retry if the reply was not valid JSON
            guidance = await run_in_threadpool(get_provider().generate, prompt)
            structured = parse_guidance(guidance)
    except Exception as exc:
        print(f'RAG guidance generation failed: {exc}')
        raise HTTPException(502, 'The RAG guidance provider failed; no fallback guidance was generated.') from exc
    if structured is None:
        raise HTTPException(502, 'The provider did not return valid RAG-based guidance.')

    evidence_by_item = {
        entry['item'].strip().casefold(): entry
        for entry in context['item_evidence']
    }
    supported_items = []
    for item in structured['items']:
        label = str(item.get('item', '')).strip()
        evidence = evidence_by_item.get(label.casefold())
        how_to_use = str(item.get('how_to_use', '')).strip()
        if not evidence or not how_to_use:
            continue
        source_links = {}
        for passage in evidence['passages']:
            key = (passage['source'], passage['title'], passage['url'])
            source_links[key] = {'name': f'{key[0]}: {key[1]}', 'url': key[2]}
        supported_items.append({
            'item': evidence['item'], 'how_to_use': how_to_use,
            'sources': list(source_links.values()),
        })
    structured['items'] = supported_items

    return {
        'prediction': prediction,
        'low_confidence': prediction['confidence'] < LOW_CONFIDENCE,
        'guidance': guidance,
        'structured': structured,
        'guidance_source': 'rag',
        'matched_materials': context['matched_materials'],
        'sources': context['sources'],
        'provider': LLM_PROVIDER,
    }
