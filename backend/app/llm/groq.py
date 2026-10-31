import requests

from app.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


class GroqProvider:
    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        self.url = f"{GROQ_BASE_URL}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            headers=self.headers,
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Follow the user's instruction. Return valid JSON "
                            "when the prompt requests JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
