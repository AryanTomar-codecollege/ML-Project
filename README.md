# WoundGuide – AI First-Aid Assistant

WoundGuide is a college project that classifies uploaded images using a fine-tuned EfficientNet-B0 model and retrieves first-aid information from a local JSON knowledge base. It can optionally use Groq or an OpenAI-compatible OmniRoute endpoint to phrase the retrieved facts.

> **Safety:** This is an educational prototype, not a medical device and not a diagnosis. For severe, worsening, or uncertain injuries, seek professional care.

## Supported model classes

The trained five-class model uses this exact order:

1. `bruise`
2. `burn`
3. `cut`
4. `no_injury`
5. `not_known_image`

Only `bruise`, `burn`, and `cut` proceed to first-aid guidance. Predictions of `no_injury` and `not_known_image` are rejected.

The Colab checkpoint is a dictionary containing `state_dict`, `class_names`, and `use_natural_images`. The backend validates the class-name order before serving predictions.

## Project structure

```text
ML-Project/
  backend/
    app/
      api/routes.py
      config.py
      llm/
      ml/predictor.py
      rag/retriever.py
      main.py
    data/knowledge_base.json
    models/                  # put your new .pth file here
    .env.example
    requirements.txt
    verify_model.py
  frontend/
    src/main.jsx
    src/style.css
    .env.example
    package.json
    vite.config.js
  README.md
```

## 1. Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11 recommended (3.12 may also work with a compatible PyTorch build)
- Node.js 20 or later
- Your trained checkpoint from Colab: `best_wound_model_v2.pth`
- Optional: Groq API key, or an available OmniRoute endpoint

## 2. Install the model checkpoint

Download `best_wound_model_v2.pth` from Google Colab.

Copy it to:

```text
backend/models/best_wound_model_v2.pth
```

The model is intentionally not bundled in this source ZIP because the current repository's tracked `.pth` file is the older three-class checkpoint. Using that old model with the updated five-class backend would produce an output-size mismatch.

## 3. Backend setup (Windows PowerShell)

Open a terminal in the project root:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

Open `backend/.env` and configure the model path:

```env
MODEL_PATH=models/best_wound_model_v2.pth
```

### Configure the LLM provider

For Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=put_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

For a running OmniRoute endpoint:

```env
LLM_PROVIDER=omniroute
OMNIROUTE_API_KEY=
OMNIROUTE_BASE_URL=http://localhost:20128/v1
OMNIROUTE_MODEL=auto
```

Do not commit `.env` or expose API keys.

### Verify the model

While still inside `backend` with the virtual environment active:

```powershell
python verify_model.py models/best_wound_model_v2.pth
```

It should print `Model loaded successfully` and the five-class mapping in order. If it reports a class mismatch, check that you copied the five-class checkpoint from the latest Colab run.

### Start the backend

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check:

- API health: `http://127.0.0.1:8000/api/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`

Leave this terminal running.

## 4. Frontend setup

Open a second terminal in the project root:

```powershell
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The frontend calls the backend at `http://127.0.0.1:8000` by default. To change it, edit `frontend/.env`:

```env
VITE_API_URL=http://127.0.0.1:8000
```

Restart Vite after changing `.env`.

## 5. API behaviour

`POST /api/analyze` accepts multipart fields:

- `image`: image file
- `items`: comma-separated items the user has available

Possible responses:

- Injury recognised: prediction, class probabilities, structured first-aid guidance, knowledge-base sources and provider.
- `no_injury` or `not_known_image`: `rejected: true`, reason `not_injury`.
- Low confidence or low top-two class margin: `rejected: true`, reason `unsure`.

The model's confidence is not medical accuracy. The non-injury classes improve rejection, but cannot guarantee detection of every unfamiliar image.

## 6. Troubleshooting

| Problem | Check |
|---|---|
| `Model not found` | Ensure the `.pth` file is in `backend/models/` and `MODEL_PATH` is correct. |
| `size mismatch` | You may have copied the old three-class checkpoint. Use the new five-class Colab checkpoint. |
| `Checkpoint class order mismatch` | Keep `CLASS_NAMES` and checkpoint metadata in the same order. |
| Backend cannot start | Activate the Python virtual environment and install `backend/requirements.txt`. |
| Frontend cannot reach backend | Keep Uvicorn running on port 8000 and check `frontend/.env`. |
| LLM guidance unavailable | Check provider settings/API key. The backend falls back to knowledge-base content. |

## Model results from the supplied Colab run

The screenshots supplied for this version show:

- Training accuracy: **99.65%**
- Validation accuracy: **93.44%**
- Test accuracy: **93.55%**
- Test macro-F1: **0.907**
- Test support: **62 images**

These are results from one small stratified split, not a guarantee of performance on new images.

## Disclaimer

WoundGuide is for educational demonstration only. It should not be used to diagnose injuries, determine severity, or replace professional medical advice.
