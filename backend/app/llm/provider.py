from app.config import LLM_PROVIDER
from app.llm.groq import GroqProvider
from app.llm.omniroute import OmniRouteProvider


def get_provider():
    if LLM_PROVIDER == "groq":
        return GroqProvider()
    if LLM_PROVIDER == "omniroute":
        return OmniRouteProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
