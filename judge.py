"""
judge.py
Functional evaluation with an LLM judge (Llama 3.3 70B via OpenRouter). Second measure,
next to the similarity-based one (evaluate.py); it does not replace it.

For every project / seed / method the judge receives the hidden stories and the suggested
stories (no method name, suggestions shuffled with a fixed seed) and decides, for each
hidden story, whether a suggestion expresses the SAME need, covers it PARTLY, or NONE does.
A suggestion can be credited to one hidden story only (as in the one-to-one matching).

    functional recall (strict)  = hidden stories with a "same" suggestion / hidden stories
    functional recall (lenient) = ("same" + "partial") / hidden stories

Suggestion sets judged (identical sets are judged once and reused):
    direct, structured             results/deepseek_seed<n>.json          Experiment 1, budget N
    agent_draft, agent             results/deepseek_agent_seed<n>.json    Critic Agent (draft = before the tool)
    direct_fs, react_fs            results/deepseek_fs_seed<n>.json       Experiment 2, budget N
    direct_k2_fs, react_k2_fs      results/deepseek_k2_fs_seed<n>.json    Experiment 2, budget 2N
Tests (two-sided Wilcoxon, paired by project, seed-averaged, on the 19 test projects; the
few-shot example projects g04, g13, g25 are excluded everywhere):
    structured vs direct, agent vs agent_draft, agent vs direct,
    react_fs vs direct_fs, react_k2_fs vs direct_k2_fs,
    direct_k2_fs vs direct_fs and react_k2_fs vs react_fs (budget 2N vs N), direct_fs vs direct

Usage (Colab, OPENROUTER_API_KEY set):
    python judge.py --validate     # 1. judge the 60 labelled pairs, agreement with the human labels
    python judge.py                # 2. judge all methods (resumable; cached in results/judge/)
    python judge.py --report       # only recompute tables from the cache
    python judge.py --mock         # offline pipeline test with a fake judge (do not report)
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import time

JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"
OUT = os.path.join("results", "judge")
CACHE = os.path.join(OUT, "judgments.json")
DEV_PROJECTS = ["g04-recycling", "g13-planningpoker", "g25-duraspace"]   # few-shot example projects
JUDGED_METHODS = ("direct", "structured", "agent_draft", "agent", "direct_fs", "react_fs",
                  "direct_k2_fs", "react_k2_fs")
TESTS = (("structured", "direct"), ("agent", "agent_draft"), ("agent", "direct"),
         ("react_fs", "direct_fs"), ("react_k2_fs", "direct_k2_fs"),
         ("direct_k2_fs", "direct_fs"), ("react_k2_fs", "react_fs"), ("direct_fs", "direct"))
LABELS = ("same", "partial", "none")

SYSTEM = """You are an expert requirements engineer acting as an impartial judge.
You receive HIDDEN user stories of a software project and a list of SUGGESTED user stories.
For each hidden story decide whether one suggestion expresses the same need.
- "same": the suggestion asks for the same capability for the same purpose, even if worded
  differently or with a slightly different role name.
- "partial": the suggestion covers a clear part of the hidden need (e.g. one of two functions,
  or the same function with an important difference), but not all of it.
- "none": no suggestion covers it. Sharing the topic, object or words is NOT enough:
  e.g. "view opening hours" and "book an appointment" are different needs.
Each suggestion may be credited to at most ONE hidden story; use the best fitting one.
Judge only meaning. Do not reward more detailed or longer suggestions.
Reply ONLY with JSON:
{"judgments": [{"hidden": "H1", "label": "same|partial|none", "suggestion": "A3 or empty", "reason": "short"}]}"""

PAIR_SYSTEM = """You are an expert requirements engineer acting as an impartial judge.
For each numbered pair, decide whether story A and story B express the same need.
- "same": same capability for the same purpose, even if worded differently.
- "partial": B covers a clear part of A's need, but not all of it.
- "none": different needs. Sharing the topic, object or words is NOT enough.
Reply ONLY with JSON: {"judgments": [{"pair": "P1", "label": "same|partial|none", "reason": "short"}]}"""


# ---------------- model call ----------------
_client = {"c": None}
USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "providers": {}}


def call(system, user):
    if _client["c"] is None:
        from openai import OpenAI
        _client["c"] = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=BASE_URL)
    for attempt in range(6):
        try:
            r = _client["c"].chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=4000,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            USAGE["calls"] += 1
            if r.usage:
                USAGE["input_tokens"] += r.usage.prompt_tokens or 0
                USAGE["output_tokens"] += r.usage.completion_tokens or 0
            prov = (getattr(r, "model_extra", None) or {}).get("provider", "unknown")
            USAGE["providers"][prov] = USAGE["providers"].get(prov, 0) + 1
            return r.choices[0].message.content or "", prov
        except Exception as e:
            wait = 5 * 2 ** attempt
            print(f"   API error ({e.__class__.__name__}: {str(e)[:120]}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("judge API failed 6 times")


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------- judging one suggestion set ----------------
def set_key(project, hidden, suggestions):
    blob = json.dumps([project, list(hidden), sorted(suggestions)], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def judge_set(project, hidden, suggestions, seed_for_order):
    order = list(range(len(suggestions)))
    random.Random(seed_for_order).shuffle(order)              # blind order, reproducible
    sid = {f"A{k + 1}": i for k, i in enumerate(order)}
    user = (f"Project: {project}\n\nHIDDEN stories ({len(hidden)}):\n" +
            "\n".join(f"H{j + 1}: {h}" for j, h in enumerate(hidden)) +
            f"\n\nSUGGESTED stories ({len(suggestions)}):\n" +
            "\n".join(f"{a}: {suggestions[i]}" for a, i in sid.items()) +
            "\n\nJudge every hidden story.")
    last = ""
    for attempt in range(3):
        text, prov = call(SYSTEM, user if attempt == 0 else
                          user + f"\n\nYour previous reply was not usable ({last}). Reply ONLY with the JSON.")
        data = parse_json(text)
        js = data.get("judgments") if isinstance(data, dict) else None
        if not isinstance(js, list):
            last = "no judgments list"
            continue
        out, used, problems = {}, set(), []
        # "same" first, so a suggestion claimed twice goes to a "same" judgment
        for j in sorted((x for x in js if isinstance(x, dict)), key=lambda x: str(x.get("label")) != "same"):
            hid, lab, a = str(j.get("hidden", "")), str(j.get("label", "")).lower().strip(), str(j.get("suggestion") or "").strip()
            if not re.fullmatch(r"H\d+", hid) or int(hid[1:]) > len(hidden) or hid in out:
                continue
            if lab not in LABELS:
                lab = "none"
            if lab != "none" and (a not in sid or a in used):
                problems.append(f"{hid}: suggestion '{a}' invalid or used twice -> none")
                lab, a = "none", ""
            if lab != "none":
                used.add(a)
            out[hid] = {"label": lab, "suggestion": suggestions[sid[a]] if lab != "none" else "",
                        "reason": str(j.get("reason", ""))[:300]}
        missing = [f"H{k + 1}" for k in range(len(hidden)) if f"H{k + 1}" not in out]
        if len(missing) > len(hidden) // 2:
            last = f"no judgment for {len(missing)} hidden stories"
            continue
        for h in missing:
            out[h] = {"label": "none", "suggestion": "", "reason": "NOT JUDGED (missing in reply)"}
            problems.append(f"{h}: missing in reply -> none")
        return {"judgments": [dict(hidden=hidden[k], **out[f"H{k + 1}"]) for k in range(len(hidden))],
                "problems": problems, "provider": prov, "attempts": attempt + 1}
    raise RuntimeError(f"{project}: no usable judge reply after 3 attempts ({last})")


# ---------------- data ----------------
def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def suggestion_sets():
    """(seed, project, method) -> (hidden, suggestions)"""
    sets = {}
    for s in (1, 2, 3):
        base = load(f"results/deepseek_seed{s}.json")
        ag = load(f"results/deepseek_agent_seed{s}.json")
        for fname, suf in (("deepseek_fs", "_fs"), ("deepseek_k2_fs", "_k2_fs")):   # few-shot, budget N and 2N
            for p, r in load(f"results/{fname}_seed{s}.json").items():
                if "direct" in r:
                    sets[(s, p, "direct" + suf)] = (r["hidden"], r["direct"]["suggestions"])
                if "react" in r:
                    sets[(s, p, "react" + suf)] = (r["hidden"], r["react"]["suggestions"])
        for p, r in base.items():
            for m in ("direct", "structured"):
                if m in r:
                    sets[(s, p, m)] = (r["hidden"], r[m]["suggestions"])
        for p, r in ag.items():
            if "direct" in r:
                sets[(s, p, "agent_draft")] = (r["hidden"], r["direct"]["suggestions"])
            if "agent" in r:
                sets[(s, p, "agent")] = (r["hidden"], r["agent"]["suggestions"])
    return sets


# ---------------- main run ----------------
def run_all():
    os.makedirs(OUT, exist_ok=True)
    cache = load(CACHE)
    sets = suggestion_sets()
    todo = {}
    index = {}
    for (s, p, m), (hidden, sugg) in sets.items():
        k = set_key(p, hidden, sugg)
        index[f"{s}|{p}|{m}"] = k
        if k not in cache:
            todo.setdefault(k, (p, hidden, sugg))
    print(f"{len(sets)} suggestion sets, {len(set(index.values()))} distinct, {len(todo)} still to judge "
          f"(judge: {JUDGE_MODEL})")
    for n, (k, (p, hidden, sugg)) in enumerate(todo.items(), 1):
        try:
            res = judge_set(p, hidden, sugg, seed_for_order=int(k[:8], 16))
        except RuntimeError as e:
            print(f"   FAILED {p}: {e} -> run again later")
            continue
        res.update(project=p, judge_model=JUDGE_MODEL, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        cache[k] = res
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        same = sum(j["label"] == "same" for j in res["judgments"])
        part = sum(j["label"] == "partial" for j in res["judgments"])
        print(f"[{n}/{len(todo)}] {p}: same {same}, partial {part} / {len(hidden)} | provider {res['provider']}"
              + (f" | {len(res['problems'])} problem(s)" if res["problems"] else ""))
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    json.dump(dict(USAGE, judge_model=JUDGE_MODEL, date=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(OUT, f"usage_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
    print(f"Judge usage this run: {USAGE['calls']} calls, {USAGE['input_tokens']}+{USAGE['output_tokens']} tokens, "
          f"providers {USAGE['providers']}")


def report():
    from scipy.stats import wilcoxon
    cache, index = load(CACHE), load(os.path.join(OUT, "index.json"))
    vals, rows, pairs_out = {}, [], []
    for key, k in index.items():
        if k not in cache:
            continue
        s, p, m = key.split("|")
        if m not in JUDGED_METHODS or p in DEV_PROJECTS:
            continue
        js = cache[k]["judgments"]
        strict = sum(j["label"] == "same" for j in js) / len(js)
        lenient = sum(j["label"] in ("same", "partial") for j in js) / len(js)
        vals.setdefault((m, p), []).append((strict, lenient))
        rows.append([m, s, p, len(js), round(strict, 4), round(lenient, 4)])
        for j in js:
            pairs_out.append([m, s, p, j["label"], j["hidden"], j["suggestion"], j["reason"]])
    if not vals:
        raise SystemExit("No judgments yet - run: python judge.py")
    per = {mp: (statistics.mean(x[0] for x in v), statistics.mean(x[1] for x in v)) for mp, v in vals.items()}
    methods = [m for m in JUDGED_METHODS if any(mm == m for mm, _ in per)]
    summary = []
    for m in methods:
        ps = [v for (mm, p), v in per.items() if mm == m]
        summary.append([m, "19 test projects", len(ps), round(100 * statistics.mean(x[0] for x in ps), 1),
                        round(100 * statistics.mean(x[1] for x in ps), 1)])
    tests = []
    for a, b in TESTS:
        ps = sorted(p for (m, p) in per if m == a and (b, p) in per)
        for i, name in ((0, "strict"), (1, "lenient")):
            x, y = [per[(a, p)][i] for p in ps], [per[(b, p)][i] for p in ps]
            if not ps:
                continue
            d = [u - v for u, v in zip(x, y)]
            bt, wt = sum(v > 1e-12 for v in d), sum(v < -1e-12 for v in d)
            pv = round(float(wilcoxon(x, y).pvalue), 4) if bt + wt and len(ps) >= 10 else ""
            tests.append([f"{a} vs {b}", name, len(ps), round(100 * statistics.mean(x), 1),
                          round(100 * statistics.mean(y), 1), round(100 * statistics.mean(d), 1),
                          f"{bt}/{len(d) - bt - wt}/{wt}", pv])

    def write(name, header, data):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(data)
    write("judge_per_project.csv", ["method", "seed", "project", "hidden", "recall_strict", "recall_lenient"], rows)
    write("judge_summary.csv", ["method", "projects_scope", "projects", "recall_strict_%", "recall_lenient_%"], summary)
    write("judge_tests.csv", ["comparison", "measure", "projects", "method_%", "baseline_%", "diff_pts",
                              "better/equal/worse", "p_value"], tests)
    write("judge_pairs.csv", ["method", "seed", "project", "label", "hidden_story", "credited_suggestion", "reason"],
          pairs_out)
    print(f"Judge: {JUDGE_MODEL} (functional measure; second to the similarity-based one)\n")
    print(f"{'method':16s} {'projects':12s} {'n':>3s} {'strict':>7s} {'lenient':>8s}")
    for r in summary:
        print(f"{r[0]:16s} {r[1]:12s} {r[2]:3d} {r[3]:6.1f}% {r[4]:7.1f}%")
    print("\nWilcoxon (paired by project, seed-averaged):")
    for t in tests:
        print(f"  {t[0]:28s} {t[1]:8s} n={t[2]:2d}  {t[3]:5.1f}% vs {t[4]:5.1f}%  diff {t[5]:+5.1f} | "
              f"{t[6]:>8s} | p {t[7] if t[7] != '' else 'n/a'}")
    print(f"\nSaved {OUT}/judge_summary.csv, judge_tests.csv, judge_per_project.csv, judge_pairs.csv")


# ---------------- validation against human labels ----------------
def validate(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    col = next(c for c in rows[0] if c.startswith("human_label"))
    rows = [r for r in rows if r[col].strip().lower() in ("yes", "no")]
    os.makedirs(OUT, exist_ok=True)
    preds = []
    for b in range(0, len(rows), 10):
        batch = rows[b:b + 10]
        user = "\n\n".join(f"P{i + 1}\nA (hidden story): {r['closest_hidden_story']}\nB (suggestion): {r['suggestion']}"
                           for i, r in enumerate(batch))
        for attempt in range(3):
            data = parse_json(call(PAIR_SYSTEM, user)[0])
            js = {str(j.get("pair")): str(j.get("label", "")).lower() for j in (data or {}).get("judgments", [])
                  if isinstance(j, dict)}
            if all(f"P{i + 1}" in js for i in range(len(batch))):
                break
        preds += [js.get(f"P{i + 1}", "none") for i in range(len(batch))]
    human = [r[col].strip().lower() == "yes" for r in rows]
    judge_same = [p == "same" for p in preds]
    judge_any = [p in ("same", "partial") for p in preds]

    def stats(j):
        agree = sum(a == b for a, b in zip(human, j)) / len(human)
        pe = (sum(human) * sum(j) + (len(human) - sum(human)) * (len(j) - sum(j))) / len(human) ** 2
        kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
        return agree, kappa
    out = os.path.join(OUT, "validation.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["similarity", "suggestion", "hidden_story", "human_label", "judge_label"])
        for r, p in zip(rows, preds):
            w.writerow([r["similarity"], r["suggestion"], r["closest_hidden_story"], r[col], p])
    print(f"Validation on {len(rows)} labelled pairs (human yes: {sum(human)}), judge {JUDGE_MODEL}")
    for name, j in (("judge 'same' = yes", judge_same), ("judge 'same' or 'partial' = yes", judge_any)):
        a, k = stats(j)
        print(f"  {name:34s} agreement {100 * a:5.1f}% | Cohen's kappa {k:.2f}")
    print(f"Saved {out}")


def main(argv=None):
    global call
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", nargs="?", const="validation/threshold_check_labelled.csv",
                    help="judge the labelled pairs (default file: threshold_check_labelled.csv)")
    ap.add_argument("--report", action="store_true", help="only recompute the tables from the cache")
    ap.add_argument("--mock", action="store_true", help="offline test with a fake judge (do not report)")
    args = ap.parse_args(argv)
    if args.mock:
        global OUT, CACHE
        OUT = os.path.join("results", "judge_mock")
        CACHE = os.path.join(OUT, "judgments.json")

        def fake(system, user):
            hs = re.findall(r"^(H\d+): ", user, flags=re.M)
            ps = re.findall(r"^(P\d+)$", user, flags=re.M)
            if ps:
                return json.dumps({"judgments": [{"pair": p, "label": ["same", "none"][i % 2], "reason": "m"}
                                                 for i, p in enumerate(ps)]}), "mock"
            return json.dumps({"judgments": [{"hidden": h, "label": ["same", "partial", "none"][i % 3],
                                              "suggestion": f"A{i + 1}", "reason": "m"}
                                             for i, h in enumerate(hs)]}), "mock"
        call = fake
    if args.validate:
        validate(args.validate)
    elif args.report:
        report()
    else:
        run_all()
        report()


if __name__ == "__main__":
    main()
