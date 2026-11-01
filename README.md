::: {align="center"}

🩹 WoundGuide

AI-Powered First-Aid Guidance with Image Classification & Retrieval-Augmented Generation

Classify an injury image. Retrieve reviewed first-aid evidence.
Explore clickable source citations.

<br>{=html}








:::

[!IMPORTANT] Educational prototype only --- not a medical
device. WoundGuide does not diagnose injuries, determine severity,
or replace a qualified healthcare professional. Model confidence and
source citations do not guarantee medical accuracy. Seek urgent
medical care for severe bleeding, deep or large wounds, serious burns,
rapidly worsening symptoms, or any emergency.

✨ Overview

WoundGuide is a web-based AI project that combines image classification
with Retrieval-Augmented Generation (RAG) to present first-aid
information grounded in a locally indexed collection of reviewed
sources.

The application classifies an uploaded image, retrieves relevant
passages, and displays guidance with clickable source links where valid
citations are available.

🚀 Features

🖼️ Image classification using an EfficientNet-B0 model.

🧭 Five prediction classes: bruise, burn, cut,
no_injury, and not_known_image.

🛑 Rejection handling for predictions classified as no_injury
or not_known_image.

📚 RAG retrieval with ChromaDB and Sentence Transformers.

🔎 Evidence-linked guidance with clickable sources beside
individual instructions when valid citations are returned.

🧰 Item-aware suggestions based on available supplies, such as
clean water, cloth, gauze, and bandages.

⚡ FastAPI backend with health status and interactive API
documentation.

🧩 Configurable LLM provider using Groq or an OpenAI-compatible
OmniRoute endpoint.

🧱 Tech Stack

Layer              Technologies

Frontend           React, Vite, JavaScript, CSS
Backend API        Python, FastAPI, Uvicorn
Image model        PyTorch, Torchvision, EfficientNet-B0
Retrieval          ChromaDB, Sentence Transformers
Source ingestion   Requests, Beautiful Soup
LLM provider       Groq or OpenAI-compatible OmniRoute

🖼️ Application Flow

flowchart TD
    A[Upload injury image] --> B[EfficientNet-B0 classifier]
    B --> C{Prediction accepted?}
    C -->|No injury / unknown / uncertain| D[Reject injury guidance]
    C -->|Bruise / burn / cut| E[Retrieve reviewed passages]
    E --> F[ChromaDB + Sentence Transformers]
    F --> G[Configured LLM formats evidence]
    G --> H[Guidance with source citations]
    H --> I[React frontend]

🏷️ Model Classes

The expected class order is defined by the model configuration and must
match the trained checkpoint exactly:

Index Class               Behavior

  `0` `bruise`            Continue to injury guidance
  `1` `burn`              Continue to injury guidance
  `2` `cut`               Continue to injury guidance
  `3` `no_injury`         Reject injury guidance
  `4` `not_known_image`   Reject injury guidance

[!NOTE] These classes are model outputs, not medical diagnoses. The
rejection classes cannot guarantee that every unfamiliar or unrelated
image will be rejected correctly.

📁 Repository Structure

ML-Project/
├── backend/
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── llm/
│   │   ├── ml/predictor.py
│   │   ├── rag/retriever.py
│   │   ├── config.py
│   │   └── main.py
│   ├── data/
│   │   └── sources/
│   │       ├── approved_sources.json
│   │       └── README.md
│   ├── models/
│   │   └── best_wound_model_v2.pth   # Obtain separately if not committed
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

Generated files such as the trained checkpoint, ChromaDB index, Python
virtual environment, and embedding-model cache may not be included in
Git. Follow the setup steps below for a fresh clone.

✅ Prerequisites

Install these before starting:

Git

Python 3.11 (recommended)

Node.js 20 or later and npm

Internet access for package installation, source ingestion, and the
first embedding-model download

The correct five-class checkpoint: best_wound_model_v2.pth

A Groq API key or a running OmniRoute endpoint if you want
LLM-formatted guidance

⚙️ Setup & Installation

1. Clone the integration branch

git clone --branch integrate-rag-system --single-branch https://github.com/AryanTomar-codecollege/ML-Project.git
cd ML-Project

These instructions use integrate-rag-system, which contains the RAG
integration. If the project has moved to a different branch, check the
repository's branch list.

2. Set up the backend

<details>

<summary>

<strong>{=html}Windows (PowerShell)</strong>{=html}

</summary>

Run from the repository root:

cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Create the environment file only if it does not already exist:

Copy-Item .env.example .env

If PowerShell blocks activation, run:

Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

</details>

<details>

<summary>

<strong>{=html}macOS / Linux</strong>{=html}

</summary>

Run from the repository root:

cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

Create the environment file only if it does not already exist:

cp .env.example .env

</details>

[!TIP] Keep the backend virtual environment activated while running
backend setup, RAG scripts, verification, and the API server. Do not
overwrite an existing .env file; edit it instead.

3. Add and verify the trained model

The trained model checkpoint may be excluded from Git because of its
size. Obtain the correct five-class checkpoint from the project
owner/training run and place it here:

backend/models/best_wound_model_v2.pth

Create the models directory if it does not exist.

From the backend/ directory, with the virtual environment activated,
run:

python verify_model.py models/best_wound_model_v2.pth

The checkpoint must match this class order:

0 -> bruise
1 -> burn
2 -> cut
3 -> no_injury
4 -> not_known_image

Do not continue if model verification fails. An older three-class
checkpoint is not compatible with this five-class setup.

4. Configure the language-model provider

Edit backend/.env. Never commit API keys or .env files.

<details>

<summary>

<strong>{=html}Option A --- Groq</strong>{=html}

</summary>

Set the provider and your own API key:

LLM_PROVIDER=groq
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b

The model shown is a configured default, not a guarantee of current
availability. Check your provider account for model access, limits, and
possible charges.

</details>

<details>

<summary>

<strong>{=html}Option B --- OmniRoute</strong>{=html}

</summary>

Use this option only if an OpenAI-compatible OmniRoute endpoint is
already running. Adjust the URL, model, and key to match your setup:

LLM_PROVIDER=omniroute
OMNIROUTE_API_KEY=
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=auto

</details>

Keep the RAG settings consistent with the index you build. Typical
settings in the example configuration are:

RAG_ENABLED=true
RAG_DB_PATH=data/chroma
RAG_COLLECTION=first_aid_guidance
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_MIN_SCORE=0.25
MAX_IMAGE_MB=10
MIN_CONFIDENCE=55
MIN_MARGIN=12

Use backend/.env.example as the source of truth for supported variable
names and defaults.

5. Prepare the RAG knowledge index

If the local ChromaDB index is absent, create it by following the steps
below. Run all commands from backend/ with the virtual environment
activated.

5.1 Ingest source pages

python scripts/ingest_sources.py

This reads the configured source list and writes extracted text to:

backend/data/sources/snapshots.jsonl

Some websites may block automated requests or change their page
structure, so not every source is guaranteed to download.

5.2 Review snapshots before approval

Open backend/data/sources/snapshots.jsonl. Review the URL, extracted
text, and injury category for every record. Correct or remove
unsuitable, irrelevant, or poorly extracted content.

[!WARNING] Do not approve records you have not reviewed. The
approval script records a review decision; it does not independently
establish medical accuracy.

5.3 Approve reviewed snapshots

Only after reviewing every record currently in the snapshot file, run:

python scripts/approve_sources.py --approve-reviewed

This command marks all records in the current snapshot file as reviewed.
Do not run it until the review is complete.

5.4 Build the index

python scripts/index_sources.py

This embeds approved passages and builds the ChromaDB index. The first
run may download the Sentence Transformers model and therefore needs
internet access.

[!CAUTION] The index script recreates the configured vector database
directory. Back up an existing index before running this command if
you need to preserve it.

5.5 Test retrieval

python scripts/test_retrieval.py

A healthy retrieval test should report:

Mode: rag
RAG error: None

It should also display retrieved passages for a sample query.

6. Start the backend API

From backend/, run:

python -m compileall app scripts
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Keep this terminal open.

Endpoint                             Purpose

http://127.0.0.1:8000/             API root/status
http://127.0.0.1:8000/api/health   Backend, model, and RAG status
http://127.0.0.1:8000/docs         Interactive API documentation

Before using the frontend, check that the health endpoint reports the
API is healthy, the model is loaded, and RAG is ready.

7. Set up and start the frontend

Open a second terminal. Leave the backend terminal running.

From the repository root:

cd frontend
npm install

Create frontend/.env only if it does not already exist:

Windows PowerShell

Copy-Item .env.example .env

macOS / Linux

cp .env.example .env

Check that frontend/.env contains:

VITE_API_URL=http://127.0.0.1:8000

Then build and start Vite:

npm run build
npm run dev

Open the local URL printed by Vite, usually http://127.0.0.1:5173. If
you change the frontend .env, restart Vite.

🧪 Test Checklist

/api/health reports the model is loaded and RAG is ready.

The frontend opens without build or browser-console errors.

A bruise image returns the expected type of result.

A burn image returns the expected type of result.

A cut image returns the expected type of result.

Normal skin and unrelated objects are tested for rejection
behavior.

Source links open the intended webpages.

Each instruction's citation is relevant to that instruction and
injury.

Duplicate links are not unnecessarily repeated.

RAG-unavailable/provider-error states are handled without
pretending unsupported guidance is verified.

🔌 API Reference

GET /api/health

Reports API health, model status, RAG status, and relevant configuration
details.

POST /api/analyze

Accepts a multipart request with:

Field     Description

image   Image file to classify
items   Available supplies, typically a comma-separated string

The response may contain the prediction, class probabilities, structured
guidance, source citations, and provider/retrieval status. Consult
/docs for the live response schema.

🔎 Citation Quality & Known Limitations

The backend can validate that a returned citation ID belongs to
retrieved evidence, but this does not prove the passage
semantically supports the associated statement.

Retrieval quality depends on approved-source coverage, page
extraction, chunking, metadata, embeddings, and query design.

An anatomically specific page (for example, guidance about a black
eye) may be retrieved for a bruise on another body part if source
coverage and ranking are not sufficiently specific. Review sources
and improve the approved-source list when this occurs.

Some websites may block automated ingestion. Always inspect
snapshots before approval.

Duplicate links should be deduplicated by URL/source identity
without hiding genuinely distinct sources.

LLM availability, model names, rate limits, and pricing can change.

The classifier may misclassify unfamiliar images and has not been
clinically validated.

🛠️ Troubleshooting

Problem                             What to check

Model checkpoint not found          Ensure
backend/models/best_wound_model_v2.pth
exists and MODEL_PATH is correct.

Checkpoint size/class mismatch      Use the five-class checkpoint, not the
older three-class model.

Python module missing               Activate .venv and run
pip install -r requirements.txt.

PyTorch installation fails          Install a compatible PyTorch/Torchvision
build for Python 3.11 and your OS, then
reinstall requirements.

model_loaded is false             Read the backend terminal error and verify
the checkpoint.

RAG unavailable                     Check RAG_ENABLED, database path,
collection name, embedding model, and
/api/health error details.

Retrieval test fails                Run ingestion, manual review, approval,
indexing, and retrieval test in order.

Some sources were skipped           The website may block requests or have
changed its markup; inspect snapshots.

Groq provider fails                 Check API key, model availability, account
limits, and network access.

OmniRoute provider fails            Check the endpoint is running and the
configured URL/model are correct.

Frontend cannot reach backend       Confirm Uvicorn runs on port 8000 and
VITE_API_URL is correct; restart Vite
after editing .env.

Vite uses another port              Open the URL printed in the frontend
terminal.

Duplicate citation links            Inspect returned citation data and
deduplicate identical source URLs.

🔐 Security & Privacy

Keep .env files, API keys, and private credentials out of Git.

Do not commit personal or sensitive images.

Only ingest sources you are permitted to use.

Review medical source content and generated guidance before relying
on it.

Do not expose the development server publicly without suitable
security controls.

⚖️ Disclaimer

WoundGuide is an educational college project. It is not a diagnostic
system, has not been clinically validated, and does not replace a
qualified healthcare professional. Predictions, confidence scores,
retrieved passages, and citations may be wrong or incomplete.