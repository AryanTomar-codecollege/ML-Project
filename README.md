# WoundGuide – AI First-Aid Assistant

A college machine-learning project. The user uploads a photo of a **cut, burn or bruise** and lists the items they have. The system classifies the injury with a fine-tuned **EfficientNet-B0**, retrieves matching facts from reviewed sources in a local vector database, and an **LLM** (Groq or OmniRoute) turns those facts into clear first-aid guidance: what to do, what not to do, how to use the items, good signs and warning signs.

> **Safety note.** This is an educational prototype, not a medical device. It cannot diagnose injuries. Seek professional care for severe, worsening or uncertain injuries.

## Features
- Image classification: bruise, burn, cut (optional 4th class `not_wound`, see section 7)
- Rejects unclear or non-injury photos and asks for a better one
- Semantic retrieval from reviewed first-aid sources, with item aliases (e.g. "ice" finds "ice pack", "haldi" finds "turmeric")
- Structured guidance cards: Do, Don't, Your items, Good signs, Warning signs
- Fails closed if reviewed source passages or valid LLM guidance are unavailable; it does not generate fallback medical guidance
- Switchable LLM provider (Groq or OmniRoute) through `.env`, no automatic fallback between providers
- Responsive React interface with camera capture on phones

## Tech stack
| Layer | Tools |
|---|---|
| ML | PyTorch, torchvision EfficientNet-B0 (transfer learning + fine-tuning) |
| Backend | FastAPI, Uvicorn, Pillow, requests |
| Retrieval | Chroma vector database, sentence-transformer embeddings, JSON material aliases |
| LLM | Groq (`openai/gpt-oss-120b`) or OmniRoute (OpenAI-compatible) |
| Frontend | React, Vite, plain CSS |

## Project structure
```text
backend/
  app/
    main.py              FastAPI app and CORS
    config.py            settings from .env, CLASS_NAMES, RAG options
    api/routes.py        /api/health and /api/analyze (gate, RAG prompt, JSON parsing)
    schemas/response.py  AnalysisResponse base response schema
    ml/predictor.py      loads the model and predicts
    rag/retriever.py     retrieves reviewed passages from the local vector store
    llm/                 base.py, groq.py, omniroute.py, provider.py
  data/
    knowledge_base.json  material names and aliases; legacy guidance data
    sources/              reviewed source snapshots
    chroma/               persistent vector database
  scripts/                ingest, approve, index and test source passages
  models/                put best_wound_model_finetuned.pth here
  verify_model.py        checks the model file loads
frontend/
  index.html, src/main.jsx, src/style.css
```

## How it works
```text
Photo + items -> React -> FastAPI /api/analyze
   1. EfficientNet-B0 predicts class + probabilities
   2. Gate: unclear / not_wound photo -> "please upload a proper photo"
   3. Retriever finds relevant reviewed passages and matching materials
   4. LLM receives ONLY these facts and returns JSON
   5. JSON is parsed; invalid guidance is retried once, then the request fails
   6. React shows the result cards
```
The LLM is told to use only the supplied facts, so it does not invent medical advice.

## 1. Requirements
- Python 3.11 or 3.12
- Node.js 20.19+ or 22.12+ (check with `node -v`)
- A Groq API key (free at console.groq.com) or a running OmniRoute server
- The trained model file `best_wound_model_finetuned.pth`

## 2. Backend setup (Windows PowerShell)
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```
1. Copy `best_wound_model_finetuned.pth` into `backend/models/`.
2. Open `backend/.env` in Notepad and paste your key: `GROQ_API_KEY=your_key_here`.
3. Start the server:
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
4. Open `http://127.0.0.1:8000/api/health`. You should see `"status":"ok"`.

macOS/Linux: use `python3 -m venv .venv`, `source .venv/bin/activate` and `cp .env.example .env`.

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope Process Bypass` once and try again.

## 3. Frontend setup
Open a second terminal:
```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```
Open `http://localhost:5173`.

## 4. Configuration (`backend/.env`)
```env
LLM_PROVIDER=groq              # groq or omniroute
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
OMNIROUTE_API_KEY=
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=auto
MODEL_PATH=models/best_wound_model_finetuned.pth
MAX_IMAGE_MB=10
RAG_ENABLED=true
RAG_DB_PATH=data/chroma
RAG_COLLECTION=first_aid_guidance
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_MIN_SCORE=0.25
MIN_CONFIDENCE=55              # optional: reject photos below this top-class %
MIN_MARGIN=12                  # optional: reject if top class leads 2nd by less than this
```
Restart the backend after changing `.env`. To switch to OmniRoute set `LLM_PROVIDER=omniroute` and fill its three values.

## 5. Retrieval and safety behavior
WoundGuide uses a local Retrieval-Augmented Generation (RAG) corpus for cuts, burns and bruises. It searches approved, reviewed source passages using semantic vectors before asking the LLM to write guidance. The LLM receives retrieved passages, not the legacy guidebook entries.

The local corpus is stored in `backend/data/sources/snapshots.jsonl`, and the persistent Chroma vector database is `backend/data/chroma/`. Every snapshot keeps its URL, publisher, injury category and retrieval date so results can be cited in the interface.

To refresh the corpus, from `backend/` run:

```powershell
python scripts/ingest_sources.py
# Inspect snapshots.jsonl, then approve it only after review:
python scripts/approve_sources.py --approve-reviewed
python scripts/index_sources.py
python scripts/test_retrieval.py
```

The source catalogue has only the three supported injury types. If a source blocks automated download (as Mayo Clinic currently does), it is skipped and reported; do not bypass that restriction. Add a permitted export or a separately approved source instead.

`backend/data/knowledge_base.json` retains material aliases, allowing plain-language inputs such as `haldi` to be recognized. Although the retriever can prepare legacy guidance when RAG is unavailable, the `/api/analyze` endpoint requires RAG mode and does not send that fallback to the LLM. Missing or insufficient RAG passages return HTTP 503; provider errors or invalid guidance return HTTP 502. No substitute guidance is generated.

## 6. Photos that are not injuries
A softmax classifier must always pick one of its classes, so an unrelated photo would still be labelled "cut". Two protections are used:
1. **Confidence gate (works now).** If the top class is below `MIN_CONFIDENCE`, or leads the second class by less than `MIN_MARGIN` points, the API returns `rejected: true` and the page asks for a better photo. Limitation: a random image can still get a high score by chance.
2. **Fourth class `not_wound` (better).** Train the model on non-injury images too. See section 7.

## 7. Adding the `not_wound` class (optional, recommended)
1. In Colab, run the cells from `colab/colab_add_not_wound_class.py`. Upload `own_negatives.zip` with 60+ of your own photos (healthy skin, hands, legs, rooms, objects) first. Own photos matter more than the CIFAR images.
2. Download `best_wound_model_v2.pth`.
3. Rename your old model to `best_wound_model_old.pth` as a backup, then save the new file as `backend/models/best_wound_model_finetuned.pth`.
4. In `backend/app/config.py` set `CLASS_NAMES = ['bruise', 'burn', 'cut', 'not_wound']` (alphabetical order is required because `ImageFolder` sorts class names).
5. In `backend/verify_model.py` change `classes` to the same list, run `python verify_model.py`, then restart the backend.

## 8. Model results (from the training notebook)
- Data: Kaggle "wound-dataset" (yasinpratomo), 231 images: 122 bruise, 59 burn, 50 cut. Split 70/15/15.
- Stage 1 trained only the new classifier layer; stage 2 fine-tuned the last three feature blocks at a lower learning rate.
- Best validation accuracy 90.91%, test accuracy 78.38% (37 test images).
- Per-class test F1: bruise 0.87, burn 0.74, cut 0.62.

The dataset is small, so results vary between photos. More images and more classes would improve it.

## 9. API
`POST /api/analyze` (multipart form: `image`, `items` as comma-separated text)

For an accepted photo, the route returns:
- `prediction`: predicted injury, confidence and class probabilities.
- `guidance`: the provider's raw JSON response as a string.
- `structured`: parsed guidance with `summary`, `do`, `dont`, `items`, `good_signs`, `warning_signs` and `see_doctor_now`.
- `low_confidence`: whether prediction confidence is below 60%.
- `guidance_source`: currently `rag`.
- `matched_materials`, `sources` and `provider`.

The `AnalysisResponse` schema in `backend/app/schemas/response.py` declares the common fields `prediction`, `guidance`, `matched_materials`, `sources` and `provider`. The route currently returns additional fields and does not bind this schema as its FastAPI `response_model`.

Rejected photo returns `rejected: true`, `reason` (`unsure` or `not_wound`), `message`, `prediction` and `provider`. Invalid uploads return HTTP 400, oversized images return HTTP 413, and unavailable RAG or guidance generation returns HTTP 503 or 502 respectively.

`GET /api/health` returns status and the active provider.

## 10. Troubleshooting
| Problem | Fix |
|---|---|
| "Cannot reach the backend" | Start the backend; check `VITE_API_URL` in `frontend/.env` |
| 502 "LLM provider failed" | Check the API key and `LLM_PROVIDER` in `backend/.env`; this endpoint does not generate fallback guidance |
| 503 "RAG is unavailable" | Index the reviewed source corpus and confirm the RAG settings in `backend/.env` |
| `Model not found` | Put the `.pth` file in `backend/models/` |
| `size mismatch` when loading | `CLASS_NAMES` length differs from the model's output classes |
| Good photo gets rejected | Lower `MIN_CONFIDENCE` / `MIN_MARGIN` in `.env` |

## 11. Limitations and future work
- Only three injury types, small dataset, small test set.
- Guidance depends on the coverage and relevance of the reviewed passages in the local vector store.
- Possible additions: more classes (abrasion, laceration), severity estimation, multilingual output, user feedback logging.

## References
American Red Cross first aid, Mayo Clinic first aid, NHS, MedlinePlus (NIH), Kaggle wound dataset by yasinpratomo, Tan and Le (2019) EfficientNet.
