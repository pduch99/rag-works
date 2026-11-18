# one chat() for every provider. content = list of strings and image paths
import base64
import io
import os
import re
import time
from pathlib import Path

from PIL import Image

from .config import PROVIDERS, SEND_PX

_clients = {}


def set_key(provider, key):
    """bring your own key from the app: lives in this process only, never written to disk"""
    os.environ[PROVIDERS[provider]["key"]] = key.strip()
    _clients.pop(provider, None)


def client(provider):
    if provider not in _clients and provider == "ollama":
        # local model on your own gpu, no key and no rate limits
        from openai import OpenAI
        _clients[provider] = OpenAI(api_key="ollama", base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), timeout=900)
    if provider not in _clients:
        key = os.getenv(PROVIDERS[provider]["key"])
        if not key:
            raise RuntimeError(f"no {PROVIDERS[provider]['key']} in .env")
        if provider == "groq":
            from groq import Groq
            _clients[provider] = Groq(api_key=key)
        else:
            # gemini speaks the openai chat format, so the openai package is just the client here
            from openai import OpenAI
            _clients[provider] = OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    return _clients[provider]


_urls = {}


def image_url(path):
    """jpeg data url, remembered per file version so evals don't re-encode the same crop every question"""
    st = Path(path).stat()
    k = (str(path), st.st_mtime_ns, SEND_PX)
    if k not in _urls:
        if len(_urls) > 300:
            _urls.clear()
        _urls[k] = _encode(path)
    return _urls[k]


def _encode(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((SEND_PX, SEND_PX))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def to_parts(content):
    parts = []
    for c in content:
        if isinstance(c, Path):
            parts.append({"type": "image_url", "image_url": {"url": image_url(c)}})
        else:
            parts.append({"type": "text", "text": c})
    return parts


def clean(text):
    # some models leak their thinking into the answer
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def chat(provider, model, content, max_tokens=800, tries=4):
    """returns (text, seconds). waits and retries on rate limits"""
    msgs = [{"role": "user", "content": to_parts(content) if any(isinstance(c, Path) for c in content) else "\n".join(content)}]
    args = {"model": model, "messages": msgs, "temperature": 0.2}
    # gemini 2.5+ thinks before answering and that counts against max_tokens, so give it room
    args["max_completion_tokens" if provider == "groq" else "max_tokens"] = max_tokens * 5 if provider == "gemini" else max_tokens
    if "gpt-oss" in model:
        args["reasoning_effort"] = "low"   # gpt-oss thinks first and that eats the token budget
    start = time.time()   # includes rate-limit waits: that's api time, not ours
    for attempt in range(tries):
        try:
            r = client(provider).chat.completions.create(**args)
            return clean(r.choices[0].message.content), round(time.time() - start, 2)
        except Exception as e:
            code = getattr(e, "status_code", None)
            if code not in RETRY or attempt == tries - 1 or "limit: 0" in str(e) or "per day" in str(e).lower():
                hint = " (is ollama running? open the ollama app or run: ollama serve)" if provider == "ollama" and code is None else ""
                raise RuntimeError(f"{provider} {model} failed ({code}): {why(e)}{hint}") from e
            wait = 20 * (attempt + 1)
            try:
                wait = float(e.response.headers.get("retry-after", wait)) + 1
            except Exception:
                pass
            print(f"{'rate limited' if code == 429 else 'model busy'} ({code}), waiting {wait:.0f}s")
            time.sleep(wait)


RETRY = {429, 500, 502, 503, 504}   # rate limit or google/groq having a moment. anything else is a real error


def why(e):
    """the api's own message, so you can see things like 'quota exceeded, limit: 0' or 'model not found'"""
    try:
        body = e.response.json()
        if isinstance(body, list):
            body = body[0]
        msg = body.get("error", body)
        return (msg.get("message") if isinstance(msg, dict) else str(msg))[:400]
    except Exception:
        return str(e)[:400]


def ping(provider, model=None):
    """one tiny call. returns (ok, message)"""
    from .config import PROVIDERS
    model = model or PROVIDERS[provider]["text"]
    try:
        out, secs = chat(provider, model, ["Reply with the single word: ok"], max_tokens=20, tries=2)
        return True, f"{provider} {model} answered in {secs:.1f}s: {out[:40]!r}"
    except Exception as e:
        return False, str(e)