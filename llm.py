"""
llm.py
One function, ask(), that sends a prompt to the generation model and records the cost
(input/output tokens) and time of every call in CALLS.

The study uses one model: DeepSeek (deepseek-v4-pro) through its OpenAI-compatible API,
temperature 0. The API key is read from the environment variable DEEPSEEK_API_KEY
(Colab Secrets or a local .env file); it is never written in the code.
Two fake models ('mock', 'mockbad') let the whole pipeline be tested offline for free.
The pipeline never talks to the API directly; it only calls ask().
"""
import hashlib
import json
import os
import random
import re
import time

MODELS = {
    "deepseek": {"provider": "deepseek", "name": "deepseek-v4-pro"},  # the model used in the study
    "mock":     {"provider": "mock", "name": "mock"},                 # free fake model for testing
    "mockbad":  {"provider": "mock", "name": "mockbad"},              # fake model whose table misses stories
}
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TEMPERATURE = 0          # identical for every method and every call

MODEL_INFO = {}          # exact model version and run date, saved next to the results
CALLS = []               # every successful call: {"model", "seconds", "input_tokens", "output_tokens"}

try:                     # optional: read the API key from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_clients = {}


def _client():
    if "deepseek" not in _clients:
        from openai import OpenAI
        _clients["deepseek"] = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL)
    return _clients["deepseek"]


def _call(model_key, system, user):
    """Returns (text, input_tokens, output_tokens)."""
    cfg = MODELS[model_key]
    if cfg["provider"] == "mock":
        text = _mock(system, user, bad=(cfg["name"] == "mockbad"))
        return text, (len(system) + len(user)) // 4, len(text) // 4
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    r = _client().chat.completions.create(model=cfg["name"], messages=msgs, temperature=TEMPERATURE)
    if not MODEL_INFO:   # record the exact model version for the report
        MODEL_INFO.update({"model_id": cfg["name"], "model_version": getattr(r, "model", None),
                           "temperature": TEMPERATURE, "run_date": time.strftime("%Y-%m-%d")})
    u = r.usage          # completion_tokens already includes any reasoning tokens (billed as output)
    return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens


def ask(model_key, system, user, retries=8):
    """Send one request; retry with growing waits if the API is busy or rate-limited."""
    for attempt in range(retries):
        try:
            t = time.time()
            text, tin, tout = _call(model_key, system, user)
            CALLS.append({"model": model_key, "seconds": round(time.time() - t, 2),
                          "input_tokens": tin, "output_tokens": tout})
            return text
        except KeyError as e:
            raise SystemExit(f"Missing API key {e}. Put it in Colab Secrets or in a .env file.")
        except Exception as e:
            wait = min(15 * 2 ** attempt, 180) + random.randint(0, 10)   # 15s, 30s, 60s ... max 3 min
            msg = str(e).replace("\n", " ")[:60]
            print(f"   API busy ({e.__class__.__name__}: {msg}) -> retry {attempt + 1}/{retries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError("API failed after several retries")


# ---------------- fake models used only for testing the pipeline ----------------
# 'mock'    : deterministic fake replies (same prompt -> same reply), so test counts repeat
# 'mockbad' : the extraction table always leaves out some stories, to test that this
#             is retried and finally recorded as a failure
def _mock(system, user, bad=False):
    rng = random.Random(hashlib.sha256((system + user).encode()).hexdigest())
    stories = re.findall(r"^(S\d+): (As .*)$", user, flags=re.M | re.I)
    if "EXTRACTOR" in system:
        keep = stories[: len(stories) // 2] if bad else stories
        rows = [{"id": sid, "role": "user", "function": s[:40], "object": "not stated",
                 "conditions": "not stated"} for sid, s in keep]
        return json.dumps({"rows": rows})
    if "ReAct" in system:                       # search twice, then finish
        m = re.search(r"exactly (\d+)", user)
        n = int(m.group(1)) if m else 5
        if user.count("Observation: [") < 3 and "used all your searches" not in user:
            sid, s = rng.choice(stories) if stories else ("S1", "log in")
            return json.dumps({"thought": "check coverage", "action": "search_backlog", "query": s[10:60]})
        out = []
        for _ in range(n):
            sid, s = rng.choice(stories) if stories else ("S1", "As a user, I want to log in.")
            out.append({"story": s.replace("I want to", "I want to quickly"), "reason": "gap",
                        "support_ids": [sid]})
        return json.dumps({"thought": "done", "action": "finish", "suggestions": out})
    m = re.search(r"exactly (\d+)", user)
    n = int(m.group(1)) if m else 5
    out = []
    for _ in range(n):
        sid, s = rng.choice(stories) if stories else ("S1", "As a user, I want to log in.")
        out.append({"story": s.replace("I want to", "I want to be able to"),
                    "reason": "related to an existing story", "support_ids": [sid]})
    return json.dumps({"suggestions": out})
