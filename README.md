WoundGuide --- AI First-Aid Assistant

WoundGuide is an educational college-project prototype that classifies
an uploaded skin-injury image with an EfficientNet-B0 model and
retrieves first-aid information from a local RAG index. The application
has a React/Vite frontend, a FastAPI backend, ChromaDB and Sentence
Transformers for retrieval, and optional Groq or OpenAI-compatible
OmniRoute integration for formatting retrieved evidence with source
links.

Safety: This is not a medical device or diagnosis. Do not use it
to determine injury severity or replace professional medical advice.
Seek appropriate medical care for severe bleeding, deep or large
wounds, serious burns, worsening symptoms, or uncertainty.

Features

Five-class image classifier.

Rejects no_injury and not_known_image predictions rather than
providing injury guidance.

RAG retrieval from a locally indexed set of reviewed first-aid
sources.

Item-aware guidance for supplies such as clean water, cloth, gauze,
bandage, ice pack, and soap.

Clickable source citations beside individual guidance statements
when the model returns valid citations.

FastAPI health endpoint and interactive API docs.

Class order

The checkpoint and backend/app/config.py must use this exact order:

bruise

burn

cut

no_injury

not_known_image

Only bruise, burn, and cut proceed to first-aid retrieval.
Confidence is not medical accuracy, and rejection is not guaranteed for
every unfamiliar image.

Technology

Frontend: React 18, Vite

Backend: Python 3.11 recommended, FastAPI, Uvicorn

Classifier: PyTorch, Torchvision, EfficientNet-B0

Retrieval: ChromaDB, Sentence Transformers

Source ingestion: Requests, Beautiful Soup

Optional LLM: Groq or a reachable OpenAI-compatible OmniRoute
endpoint

Repository layout

ML-Project/
├── backend/
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── llm/
│   │   ├── ml/predictor.py
│   │   ├── rag/retriever.py
│   │   ├── config.py
│   │   └── main.py
│   ├── data/sources/approved_sources.json
│   ├── models/best_wound_model_v2.pth   # obtain separately
│   ├── scripts/
│   │   ├── ingest_sources.py
│   │   ├── approve_sources.py
│   │   ├── index_sources.py
│   │   └── test_retrieval.py
│   ├── .env.example
│   ├── requirements.txt
│   └── verify_model.py
├── frontend/
│   ├── src/
│   ├── .env.example
│   ├── package.json
│   └── vite.config.js
└── README.md

The trained checkpoint, downloaded embedding-model cache, and generated
ChromaDB index may not be committed to Git. A new user should follow the
steps below.

Requirements

Git

Python 3.11 recommended

Node.js 20+ and npm

Internet access for dependency installation, source ingestion, and
the first embedding-model download

The correct five-class checkpoint best_wound_model_v2.pth

A Groq API key or a running OmniRoute endpoint if LLM-formatted
guidance is required

1. Clone the RAG integration branch

git clone --branch integrate-rag-system --single-branch https://github.com/AryanTomar-codecollege/ML-Project.git
cd ML-Project

These instructions assume the integrate-rag-system branch.

2. Install backend dependencies

Windows PowerShell

Run from the repository root:

cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env

Only copy .env.example if backend/.env does not already exist. If
PowerShell blocks activation, run
Set-ExecutionPolicy -Scope Process Bypass, then activate again.

macOS / Linux

Run from the repository root:

cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env

Do not overwrite an existing .env; edit it instead. If PyTorch
installation fails, install a compatible PyTorch/Torchvision build for
your Python and operating system, then rerun
pip install -r requirements.txt.

3. Add and verify the trained model

Obtain the five-class checkpoint from the project owner/training run and
copy it to:

backend/models/best_wound_model_v2.pth

Create the models directory if needed. Do not use the old three-class
checkpoint.

With the backend virtual environment active and the current directory
still backend, run:

python verify_model.py models/best_wound_model_v2.pth

It should print Model loaded successfully and the class mapping:

0 -> bruise
1 -> burn
2 -> cut
3 -> no_injury
4 -> not_known_image

Resolve any checkpoint or class-order error before proceeding.

4. Configure the LLM provider

Edit backend/.env. Never commit this file or share API keys.

Groq

LLM_PROVIDER=groq
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b

The model is a configured default; check your provider account for
current model availability, access, limits, and any costs.

OmniRoute

Use this only if an OpenAI-compatible OmniRoute endpoint is already
running:

LLM_PROVIDER=omniroute
OMNIROUTE_API_KEY=
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=auto

Set the actual URL, model, and API key required by your endpoint.

Other .env settings have defaults in backend/.env.example. In
particular, keep the RAG collection and embedding model consistent with
the index:

RAG_ENABLED=true
RAG_DB_PATH=data/chroma
RAG_COLLECTION=first_aid_guidance
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_MIN_SCORE=0.25
MAX_IMAGE_MB=10
MIN_CONFIDENCE=55
MIN_MARGIN=12

5. Prepare the RAG index

If a working backend/data/chroma index and the matching
embedding-model cache are already available, run the retrieval test at
the end of this section first. Otherwise, build the index as follows.
Keep the backend virtual environment active and run commands from
backend/.

5.1 Ingest source snapshots

python scripts/ingest_sources.py

The script downloads pages listed in
data/sources/approved_sources.json and writes extracted text to
data/sources/snapshots.jsonl. Some sites may block automated requests
or change their page structure, so some sources may be skipped.

5.2 Manually review snapshots

Open backend/data/sources/snapshots.jsonl. Check every record's URL,
extracted text, and injury category. Remove or correct unsuitable or
irrelevant records before approval.

Do not approve records you have not reviewed. The approval script
records a human review decision; it does not independently validate
medical accuracy.

5.3 Approve the reviewed snapshots

Only after reviewing all records currently in the file, run:

python scripts/approve_sources.py --approve-reviewed

This command marks all records in snapshots.jsonl as reviewed. Do not
run it before completing the review.

5.4 Build the index

python scripts/index_sources.py

The script downloads/loads the configured embedding model, chunks
approved text, and creates the ChromaDB index. It recreates the
configured vector database directory, so back up an existing index
before running if you need to preserve it. The first run needs internet
access to download the embedding model.

5.5 Test retrieval

python scripts/test_retrieval.py

A successful result should show:

Mode: rag
RAG error: None

It should also print retrieved passages for a cut query.

Known source-quality issue: The current approved-source list maps
the bruise category to an NHS black-eye page as well as other bruise
sources. That may lead to an eye-specific source being used for a hand
bruise. Review retrieved passages and source relevance; a clickable URL
does not prove the page supports the exact instruction. Improve the
approved source list and rebuild the index if you find mismatched
sources.

6. Start the backend

From backend/, with the virtual environment active:

python -m compileall app scripts
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Keep this terminal running.

Health: http://127.0.0.1:8000/api/health

API documentation: http://127.0.0.1:8000/docs

Root status: http://127.0.0.1:8000/

Check that health reports status: "ok", model_loaded: true, and
rag_mode: "ready".

7. Install and start the frontend

Open a second terminal; leave the backend running.

Windows PowerShell

From the repository root:

cd frontend
npm install
Copy-Item .env.example .env
npm run build
npm run dev

Only copy .env.example if frontend/.env does not already exist.

macOS / Linux

From the repository root:

cd frontend
npm install
cp .env.example .env
npm run build
npm run dev

The frontend environment file should contain:

VITE_API_URL=http://127.0.0.1:8000

If you edit .env, restart Vite. Open the local URL printed by Vite,
usually http://127.0.0.1:5173.

8. Test the application

Check /api/health before opening the frontend.

Upload a clear image showing one skin injury.

Select or enter available supplies.

Submit the analysis.

Review the predicted class and confidence.

Check the summary, "What to do," "What not to do," item
instructions, good signs, and warning signs.

Click source links and verify that the pages are relevant to the
exact instruction and injury.

Test a cut, burn, and bruise image, as well as normal skin and
unrelated objects.

Check that duplicate links are not unnecessarily repeated.

API endpoints

GET /api/health

Reports API status, provider, model-loaded status, RAG readiness/error,
and configured classes.

POST /api/analyze

Accepts multipart fields:

image: image file

items: comma-separated list of available items

The response can contain the prediction, class probabilities, structured
guidance, citations, sources, and provider information. no_injury,
not_known_image, low-confidence, and uncertain predictions can be
rejected.

Use http://127.0.0.1:8000/docs to inspect the current API schema.

Source citations and limitations

The backend assigns IDs to retrieved passages and checks that
returned citation IDs belong to the allowed retrieved evidence set.

The frontend shows clickable source links when valid citations are
returned.

Citation-ID validation does not prove semantic support for a claim.
Review source passages and citations.

Retrieval quality depends on source coverage, extraction, chunking,
metadata, embeddings, and query design.

Some websites may block ingestion. Inspect snapshots before
approving and indexing.

Model-provider names, availability, rate limits, and pricing may
change.

The classifier can misclassify unfamiliar images and has not been
clinically validated.

Troubleshooting

Problem                             Check

Model not found                   Put best_wound_model_v2.pth in
backend/models/ and check
MODEL_PATH.

Checkpoint size/class-order error   Use the correct five-class
checkpoint, not the older
three-class one.

Missing Python module               Activate .venv and run
pip install -r requirements.txt.

PyTorch install fails               Install a compatible
PyTorch/Torchvision build for Python
3.11 and your OS.

model_loaded: false               Read the backend terminal error and
verify the checkpoint.

rag_mode: unavailable             Check RAG_ENABLED, RAG_DB_PATH,
RAG_COLLECTION, the
embedding-model cache, and
/api/health's rag_error.

Retrieval test fails                Run ingestion, manual review,
approval, and indexing in order.

Source ingestion skips pages        A site may block requests or have
changed its HTML; inspect snapshots
and source coverage.

Groq error                          Check API key, model access, account
limits, and connectivity.

OmniRoute error                     Check that the endpoint is running
and the configured base URL/model
are correct.

Frontend cannot reach backend       Ensure Uvicorn is running on port
8000 and frontend/.env points to
it; restart Vite after changing
.env.

Vite uses a different port          Open the URL printed in the
terminal.

Duplicate source links              Inspect citation output and
deduplicate identical URLs/source
identities without removing distinct
sources.

Security and privacy

Keep .env files and API keys private.

Do not commit sensitive images or personal information.

Only ingest sources you are permitted to use.

Review sources and generated guidance before relying on the output.

Do not expose the development server publicly without appropriate
security controls.

Disclaimer

WoundGuide is an educational college project. It is not a diagnostic
system, has not been clinically validated, and does not replace a
qualified healthcare professional. Model confidence and source citations
do not guarantee medical accuracy.