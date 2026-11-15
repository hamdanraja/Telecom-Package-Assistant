import json
import re

import ollama
from langchain_ollama import ChatOllama

from config import OLLAMA_HOST, OLLAMA_MODEL, NUM_CTX


class LLMUnavailable(RuntimeError):
    """Ollama is not running, or the model is missing."""


def _strip_think(t: str) -> str:
    return re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()


def _friendly(e: Exception) -> str:
    if isinstance(e, ollama.ResponseError):
        return (f"Ollama error: {getattr(e, 'error', e)}. "
                f"Check that '{OLLAMA_MODEL}' appears in `ollama list`.")
    return (f"Cannot reach Ollama at {OLLAMA_HOST}. Start it (open the Ollama app or run "
            f"`ollama serve`) and try again. ({type(e).__name__})")


def _model(temp: float, think: bool = False, schema: dict | None = None) -> ChatOllama:
    kw = dict(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=temp,
              num_ctx=NUM_CTX, reasoning=bool(think))
    if schema is not None:
        kw["format"] = schema
    return ChatOllama(**kw)


def chat_json(messages, schema: dict) -> dict:
    for _ in range(2):
        try:
            r = _model(0.0, False, schema).invoke(messages)
            return json.loads(_strip_think(r.content))
        except json.JSONDecodeError:
            continue
        except Exception as e:
            raise LLMUnavailable(_friendly(e)) from e
    return {}


def chat_stream(messages, think: bool = False, temp: float = 0.1):
    try:
        in_think = False
        for chunk in _model(temp, think).stream(messages):
            tok = chunk.content or ""
            if "<think>" in tok: in_think = True
            if not in_think and tok: yield tok
            if "</think>" in tok: in_think = False
    except Exception as e:
        raise LLMUnavailable(_friendly(e)) from e