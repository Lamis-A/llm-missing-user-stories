"""
pipeline.py
Four ways of suggesting missing user stories for one project, all with the same model:

  direct        one call:  visible stories -> n suggested stories                 (baseline)
  structured    two calls: (1) visible stories -> table of role / function / object /
                               conditions / source story (extraction only)
                           (2) table + the same visible stories -> n suggested stories
  agent         Critic Agent: the direct draft, then a tool flags suggestions that are
                near-copies (similarity >= 0.85) of a visible story and the model replaces
                them (at most 2 rounds)
  react         ReAct Agent: at every step the model either searches the visible backlog
                (search_backlog, top 5 stories) or finishes; at least 3 and at most 6 searches

Direct, structured and the Critic Agent use the SAME system prompt and the SAME output format
for suggestions (story, short reason, IDs of the visible stories it is based on).
With --fewshot, direct and react receive the same worked examples, built only from the
three example projects (g04, g13, g25), which are never tested.

Visible stories are shown with neutral IDs S1..Sk (not the dataset IDs), so gaps in
the dataset numbering cannot reveal where hidden stories were.
Hidden stories never enter this module; the tools search the visible stories only.

Any reply that is not valid JSON in the expected shape is asked again
(MAX_FORMAT_RETRIES times); if it still fails, PipelineFormatError is raised and the
project is recorded as failed (never as a result).
"""
import json
import re
import time

import llm

MAX_FORMAT_RETRIES = 2


class PipelineFormatError(Exception):
    """The model kept returning output that could not be used."""


EXTRACTOR_SYSTEM = """You are a requirements analyst (EXTRACTOR).
You receive the user stories of one software project, each with an ID.
Organise what they say into a table. For every story write at least one row with:
- "id": the ID of the source story,
- "role": the user role (actor),
- "function": the need or function the story asks for,
- "object": the item the function acts on, or "not stated",
- "conditions": important conditions or constraints, or "not stated".
This is extraction only:
- Do NOT add roles, functions, objects or conditions that the stories do not state.
- Keep differences between stories; do not merge stories that differ in role,
  function, object or condition.
- If information is missing, write "not stated" instead of guessing.
- Every story ID must appear in at least one row.
Reply ONLY with JSON: {"rows": [{"id": "S1", "role": "...", "function": "...", "object": "...", "conditions": "..."}]}"""

SUGGEST_SYSTEM = """You are an experienced agile requirements analyst.
You receive part of a software project's backlog. Other user stories of this project
are missing. Suggest user stories that are likely missing.
For every suggestion:
- "story": one user story in the format "As a <role>, I want <goal>, so that <benefit>."
- "reason": one short sentence explaining why this need is likely missing.
- "support_ids": the IDs of the given stories the suggestion is based on.
Rules:
- Stay within the project's scope; reuse its roles and terminology.
- Do not repeat or rephrase a need that the given stories already cover.
- One specific need per story.
- A suggestion is a possibility to be checked, not a confirmed requirement.
Reply ONLY with JSON:
{"suggestions": [{"story": "...", "reason": "...", "support_ids": ["S1"]}]}"""

FORMAT_REMINDER = ("\n\nIMPORTANT: your previous reply could not be used ({problem}). "
                   "Reply again with ONLY the JSON object in the exact format requested.")


# ---------------- helpers ----------------
def parse_json(text):
    text = re.sub(r"```(json)?", "", text or "")
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def ask_valid(model, system, user, validate, format_log, stage):
    prompt = user
    for _ in range(1 + MAX_FORMAT_RETRIES):
        reply = llm.ask(model, system, prompt)
        value, problem = validate(parse_json(reply))
        if problem is None:
            return value
        format_log.append({"stage": stage, "problem": problem, "reply_start": (reply or "")[:200]})
        prompt = user + FORMAT_REMINDER.format(problem=problem)
    raise PipelineFormatError(f"{stage}: invalid reply {1 + MAX_FORMAT_RETRIES} times: {problem}")


def numbered(visible):
    """Neutral IDs S1..Sk for the visible stories."""
    return [(f"S{i}", s) for i, s in enumerate(visible, 1)]


def stories_block(items):
    return "\n".join(f"{sid}: {s}" for sid, s in items)


# ---------------- stage 1: extraction table ----------------
def validate_table(data, ids):
    if data is None:
        return None, "not valid JSON"
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        return None, 'missing or empty "rows" list'
    clean = []
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str):
            return None, 'a row has no string "id"'
        clean.append({k: str(r.get(k, "not stated")).strip() or "not stated"
                      for k in ("id", "role", "function", "object", "conditions")})
    unknown = sorted({r["id"] for r in clean} - set(ids))
    if unknown:
        return None, f"unknown story IDs in the table: {', '.join(unknown[:10])}"
    missing = [i for i in ids if i not in {r["id"] for r in clean}]
    if missing:
        return None, f"these stories have no row: {', '.join(missing[:15])}"
    return clean, None


def extract_table(model, project, items, format_log):
    user = (f"Project: {project}\n\nUser stories ({len(items)}):\n{stories_block(items)}\n\n"
            "Build the table.")
    return ask_valid(model, EXTRACTOR_SYSTEM, user,
                     lambda d: validate_table(d, [i for i, _ in items]), format_log, "extraction")


def table_block(rows):
    lines = ["id | role | function | object | conditions"]
    lines += [f"{r['id']} | {r['role']} | {r['function']} | {r['object']} | {r['conditions']}" for r in rows]
    return "\n".join(lines)


# ---------------- stage 2 / direct: suggestions ----------------
def validate_suggestions(data, n):
    if data is None:
        return None, "not valid JSON"
    sugg = data.get("suggestions")
    if not isinstance(sugg, list):
        return None, 'missing "suggestions" list'
    out = []
    for s in sugg:
        if not isinstance(s, dict) or not isinstance(s.get("story"), str) or not s["story"].strip():
            return None, 'a suggestion has no "story" text'
        ids = s.get("support_ids", [])
        ids = [str(i) for i in ids] if isinstance(ids, list) else []
        out.append({"story": s["story"].strip(), "reason": str(s.get("reason", "")).strip(),
                    "support_ids": ids})
    if not out:
        return None, "empty suggestions list"
    return out[:n], None              # more than n are cut to n so both methods get the same budget


def suggest(model, project, items, n, format_log, table=None, fewshot=False):
    user = (fewshot_block() + "\n\n") if fewshot else ""
    user += f"Project: {project}\n\n"
    if table is not None:
        user += ("Table of the roles, functions, objects and conditions found in the stories:\n"
                 f"{table_block(table)}\n\n")
    user += (f"User stories ({len(items)}):\n{stories_block(items)}\n\n"
             f"Write exactly {n} suggestions of missing user stories.")
    return ask_valid(model, SUGGEST_SYSTEM, user, lambda d: validate_suggestions(d, n),
                     format_log, "suggestion")


# ---------------- Critic Agent: direct draft + a checking tool + revision ----------------
AGENT_THRESHOLD = 0.85     # similarity at which a suggestion counts as a near-copy
AGENT_MAX_ROUNDS = 2       # how many times the model may be asked to replace flagged suggestions
_embedder = {}


def tool_check_backlog(suggestions, visible):
    """
    The agent's TOOL. Uses only the visible backlog (never the hidden stories).
    For each suggestion, returns None if it is new, or the reason it is flagged:
    a near-copy of a visible story, or a near-copy of an earlier suggestion.
    """
    if "m" not in _embedder:
        from sentence_transformers import SentenceTransformer
        _embedder["m"] = SentenceTransformer("all-MiniLM-L6-v2")
    m = _embedder["m"]
    es = m.encode(suggestions, normalize_embeddings=True)
    ev = m.encode(visible, normalize_embeddings=True)
    flags = []
    for i, e in enumerate(es):
        sims = [float(e @ v) for v in ev]
        j = max(range(len(sims)), key=sims.__getitem__)
        if sims[j] >= AGENT_THRESHOLD:
            flags.append({"index": i, "kind": "copy of an existing story", "similar_to": f"S{j + 1}",
                          "similar_text": visible[j], "similarity": round(sims[j], 3)})
            continue
        prev = [(k, float(e @ es[k])) for k in range(i) if k not in {f["index"] for f in flags}]
        dup = max(prev, key=lambda x: x[1], default=None)
        if dup and dup[1] >= AGENT_THRESHOLD:
            flags.append({"index": i, "kind": "duplicate of another suggestion",
                          "similar_to": f"suggestion {dup[0] + 1}", "similar_text": suggestions[dup[0]],
                          "similarity": round(dup[1], 3)})
    return flags


def revise(model, project, items, kept, flagged, format_log):
    """Ask the model to replace the flagged suggestions with new ones."""
    lines = "\n".join(f'- "{f["story"]}" -> {f["kind"]} ({f["similar_to"]}: "{f["similar_text"]}")'
                       for f in flagged)
    user = (f"Project: {project}\n\nUser stories ({len(items)}):\n{stories_block(items)}\n\n"
            "Your accepted suggestions (do not repeat them):\n"
            + ("\n".join(f"- {s['story']}" for s in kept) or "- none") + "\n\n"
            "A checking tool found that these suggestions repeat a need that is already covered:\n"
            f"{lines}\n\n"
            f"Write exactly {len(flagged)} NEW suggestions to replace them. Each must be a need that is "
            "not in the user stories and not in your accepted suggestions.")
    return ask_valid(model, SUGGEST_SYSTEM, user, lambda d: validate_suggestions(d, len(flagged)),
                     format_log, "agent_revision")


def run_agent(model, project, visible, n):
    """
    Returns (direct_result, agent_result). The agent starts from the direct suggestions
    (the same call as the direct method), checks them with the tool, and asks the model to
    replace flagged ones, up to AGENT_MAX_ROUNDS times. Same budget n as direct.
    """
    direct = run_method("direct", model, project, visible, n)
    items = numbered(visible)
    valid_ids = {i for i, _ in items}
    sugg = [dict(d) for d in direct["details"]]
    format_log, first_call, t = [], len(llm.CALLS), time.time()
    log = []
    for rnd in range(1, AGENT_MAX_ROUNDS + 1):
        flags = tool_check_backlog([s["story"] for s in sugg], visible)
        log.append({"round": rnd, "flagged": len(flags),
                    "flags": [{k: f[k] for k in ("index", "kind", "similar_to", "similarity")} for f in flags]})
        if not flags:
            break
        bad = {f["index"] for f in flags}
        kept = [s for i, s in enumerate(sugg) if i not in bad]
        flagged = [dict(f, story=sugg[f["index"]]["story"]) for f in flags]
        new = revise(model, project, items, kept, flagged, format_log)
        for s in new:
            s["invalid_support_ids"] = [i for i in s["support_ids"] if i not in valid_ids]
        sugg = kept + new
    else:
        flags = tool_check_backlog([s["story"] for s in sugg], visible)
        log.append({"round": "final check", "flagged": len(flags)})
    calls = llm.CALLS[first_call:]
    agent = {
        "suggestions": [s["story"] for s in sugg],
        "details": sugg,
        "table": None,
        "n_requested": n,
        "n_returned": len(sugg),
        # cost of the agent = the direct call + the revision calls
        "calls": direct["calls"] + len(calls),
        "input_tokens": direct["input_tokens"] + sum(c["input_tokens"] for c in calls),
        "output_tokens": direct["output_tokens"] + sum(c["output_tokens"] for c in calls),
        "seconds": round(direct["seconds"] + time.time() - t, 1),
        "format_retries": direct["format_retries"] + format_log,
        "agent_log": log,
    }
    return direct, agent


# ---------------- few-shot examples (from the 3 example projects only) ----------------
FEWSHOT_PROJECTS = ["g04-recycling", "g13-planningpoker", "g25-duraspace"]
FEWSHOT_VISIBLE = 12       # visible stories shown per example
FEWSHOT_MISSING = 3        # missing stories shown per example
_fewshot = {}


def fewshot_block():
    """
    Worked examples built ONCE from the seed-1 split of the example projects
    (never from a test project), and identical for every method that uses few-shot.
    """
    if "text" in _fewshot:
        return _fewshot["text"]
    import csv, os
    rows = list(csv.DictReader(open(os.path.join("data", "split_seed1.csv"), encoding="utf-8")))
    parts = []
    for k, proj in enumerate(FEWSHOT_PROJECTS, 1):
        vis = [r["story"] for r in rows if r["project"] == proj and r["hidden"] != "True"][:FEWSHOT_VISIBLE]
        hid = [r["story"] for r in rows if r["project"] == proj and r["hidden"] == "True"][:FEWSHOT_MISSING]
        parts.append(f"Example {k} (another project)\nGiven stories (excerpt):\n"
                     + "\n".join(f"- {s}" for s in vis)
                     + "\nStories that were actually missing:\n" + "\n".join(f"- {s}" for s in hid))
    _fewshot["text"] = ("Worked examples from OTHER projects, showing the kind of stories that are "
                        "typically missing:\n\n" + "\n\n".join(parts))
    return _fewshot["text"]


# ---------------- ReAct agent: the model decides when to search the backlog ----------------
REACT_MIN_SEARCHES = 3     # the model must check candidates with the tool before finishing
REACT_MAX_STEPS = 6        # tool calls allowed before the model must finish
REACT_TOP_K = 5            # stories returned by one search

REACT_SYSTEM = """You are an experienced agile requirements analyst working as a ReAct agent.
Goal: find user stories that are likely MISSING from a project's backlog.
You can use one tool:
  search_backlog(query): returns the given stories most similar to the query, with their IDs
  and similarity scores. Use it to check whether a candidate need is already covered.
Work step by step: think of candidate missing needs, then CHECK them with search_backlog
before you finish. You must make at least 3 searches (at most 6). Drop or change candidates
that the search shows are already covered.
In EVERY reply output exactly one JSON object, either
  {"thought": "...", "action": "search_backlog", "query": "..."}
or, when you are ready,
  {"thought": "...", "action": "finish", "suggestions": [{"story": "...", "reason": "...", "support_ids": ["S1"]}]}
Rules for the final suggestions:
- "story": one user story in the format "As a <role>, I want <goal>, so that <benefit>."
- "reason": one short sentence explaining why this need is likely missing.
- "support_ids": the IDs of the given stories the suggestion is based on.
- Stay within the project's scope; reuse its roles and terminology.
- Do not repeat or rephrase a need that the given stories already cover.
- One specific need per story.
Reply ONLY with the JSON object."""


def tool_search_backlog(query, visible, k=REACT_TOP_K):
    """The agent's TOOL: semantic search over the VISIBLE stories only (never the hidden ones)."""
    if "m" not in _embedder:
        from sentence_transformers import SentenceTransformer
        _embedder["m"] = SentenceTransformer("all-MiniLM-L6-v2")
    m = _embedder["m"]
    q = m.encode([query], normalize_embeddings=True)[0]
    ev = m.encode(visible, normalize_embeddings=True)
    sims = sorted(((float(q @ v), j) for j, v in enumerate(ev)), reverse=True)[:k]
    return [{"id": f"S{j + 1}", "story": visible[j], "similarity": round(s, 3)} for s, j in sims]


def validate_react_step(data, n):
    if data is None:
        return None, "not valid JSON"
    act = data.get("action")
    if act == "search_backlog":
        q = data.get("query")
        if not isinstance(q, str) or not q.strip():
            return None, 'search_backlog needs a non-empty "query"'
        return {"action": act, "query": q.strip(), "thought": str(data.get("thought", ""))}, None
    if act == "finish":
        sugg, problem = validate_suggestions(data, n)
        if problem:
            return None, problem
        return {"action": act, "suggestions": sugg, "thought": str(data.get("thought", ""))}, None
    return None, 'action must be "search_backlog" or "finish"'


def run_react(model, project, visible, n):
    """
    ReAct loop: the model reasons, chooses to call search_backlog (or to finish), sees the
    observation, and decides again. At most REACT_MAX_STEPS searches; then it must finish.
    The few-shot examples are the same ones the few-shot direct method receives.
    """
    items = numbered(visible)
    valid_ids = {i for i, _ in items}
    format_log, first_call, t = [], len(llm.CALLS), time.time()
    base = (f"{fewshot_block()}\n\nProject: {project}\n\nUser stories ({len(items)}):\n"
            f"{stories_block(items)}\n\nTask: find exactly {n} missing user stories.")
    transcript, log, sugg = [], [], None
    searches, step = 0, 0
    for _ in range(REACT_MAX_STEPS + REACT_MIN_SEARCHES + 2):
        step += 1
        must_finish = searches >= REACT_MAX_STEPS
        user = base + ("\n\nYour previous steps:\n" + "\n".join(transcript) if transcript else "")
        user += ("\n\nYou have used all your searches: reply now with action \"finish\" and exactly "
                 f"{n} suggestions." if must_finish else "\n\nNext step:")
        out = ask_valid(model, REACT_SYSTEM, user, lambda d: validate_react_step(d, n), format_log,
                        f"react_step{step}")
        rejected = sum(1 for x in log if x["action"] == "finish_rejected")
        if out["action"] == "finish" and searches < REACT_MIN_SEARCHES and rejected < 2:
            # finishing without checking: not accepted (at most twice), the model is told to use the tool
            log.append({"step": step, "thought": out["thought"][:500], "action": "finish_rejected"})
            transcript.append(json.dumps({"thought": out["thought"], "action": "finish"}, ensure_ascii=False)
                              + f"\nObservation: not accepted - you made {searches} search(es); check your "
                              f"candidates with search_backlog at least {REACT_MIN_SEARCHES} times first.")
            continue
        if out["action"] == "finish":
            sugg = out["suggestions"]
            log.append({"step": step, "thought": out["thought"][:500], "action": "finish"})
            break
        if must_finish:              # the model kept searching after being told to finish
            raise PipelineFormatError("react: no finish after the search limit")
        searches += 1
        obs = tool_search_backlog(out["query"], visible)
        log.append({"step": step, "thought": out["thought"][:500], "action": "search_backlog",
                    "query": out["query"], "observation_ids": [o["id"] for o in obs],
                    "top_similarity": obs[0]["similarity"] if obs else None})
        transcript.append(json.dumps({"thought": out["thought"], "action": "search_backlog",
                                      "query": out["query"]}, ensure_ascii=False)
                          + "\nObservation: " + json.dumps(obs, ensure_ascii=False))
    if sugg is None:
        raise PipelineFormatError("react: the model never finished")
    for s in sugg:
        s["invalid_support_ids"] = [i for i in s["support_ids"] if i not in valid_ids]
    calls = llm.CALLS[first_call:]
    return {
        "suggestions": [s["story"] for s in sugg],
        "details": sugg,
        "table": None,
        "n_requested": n,
        "n_returned": len(sugg),
        "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "seconds": round(time.time() - t, 1),
        "format_retries": format_log,
        "agent_log": log,
        "tool_calls": searches,
        "finish_rejected": sum(1 for x in log if x["action"] == "finish_rejected"),
    }


# ---------------- one project, one method ----------------
def run_method(method, model, project, visible, n, fewshot=False):
    items = numbered(visible)
    valid_ids = {i for i, _ in items}
    format_log, first_call = [], len(llm.CALLS)
    t = time.time()
    table = None
    if method == "structured":
        table = extract_table(model, project, items, format_log)
    sugg = suggest(model, project, items, n, format_log, table=table, fewshot=fewshot)
    calls = llm.CALLS[first_call:]
    for s in sugg:
        s["invalid_support_ids"] = [i for i in s["support_ids"] if i not in valid_ids]
    return {
        "suggestions": [s["story"] for s in sugg],
        "details": sugg,
        "table": table,
        "n_requested": n,
        "n_returned": len(sugg),
        "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "seconds": round(time.time() - t, 1),
        "format_retries": format_log,
    }
