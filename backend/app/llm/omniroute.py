import requests

from app.config import (
    OMNIROUTE_API_KEY,
    OMNIROUTE_BASE_URL,
    OMNIROUTE_MODEL,
)


class OmniRouteProvider:
    def __init__(self):
        self.url = f"{OMNIROUTE_BASE_URL}/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        if OMNIROUTE_API_KEY:
            self.headers["Authorization"] = f"Bearer {OMNIROUTE_API_KEY}"

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            headers=self.headers,
            json={
                "model": OMNIROUTE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
