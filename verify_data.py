"""
verify_data.py
Independent checks of the dataset, the splits and the model inputs.
It recounts everything from the raw files and the data files, instead of trusting the
cleaning log, and runs the real experiment code with a fake model to record every
prompt the agents receive.

Usage:
    python verify_data.py "<folder with the 22 raw .txt files>"
Writes verification_report.txt and stops with an error if any check fails.
"""
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

RAW = sys.argv[1]
SEEDS = [1, 2, 3]
report = []


def check(ok, msg):
    report.append(("PASS " if ok else "FAIL ") + msg)
    print(report[-1])
    if not ok:
        raise SystemExit("Verification failed: " + msg)


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def raw_lines(path):
    b = open(path, "rb").read()
    try:
        t = b.decode("utf-8")
    except UnicodeDecodeError:
        t = b.decode("cp1252")
    t = t.replace("﻿", "").replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    return [re.sub(r"\s+", " ", l).strip() for l in t.splitlines()]


# ---------- 1. counts from the raw files ----------
files = sorted(glob.glob(os.path.join(RAW, "*.txt")))
nonempty = story_lines = 0
raw_texts = defaultdict(set)
for f in files:
    p = os.path.splitext(os.path.basename(f))[0]
    for l in raw_lines(f):
        if l:
            nonempty += 1
            if re.match(r"(?i)^as\s", l):
                story_lines += 1
                raw_texts[p].add(l)
check(len(files) == 22, f"22 raw project files found ({len(files)})")
report.append(f"     raw non-empty lines: {nonempty}; lines starting a story: {story_lines}")

clean = read("data/clean_stories.csv")
log = read("data/cleaning_log.csv")
dup = [r for r in read("review/duplicate_review.csv") if r["decision"] == "DUPLICATE"]
exact = sum(r["action"] == "REMOVED_EXACT_DUPLICATE" for r in log)
semantic = sum(r["action"] == "REMOVED_SEMANTIC_DUPLICATE" for r in log)
check(semantic == len(dup), f"semantic removals in log ({semantic}) = DUPLICATE decisions ({len(dup)})")
expected = story_lines - exact - semantic
check(len(clean) == expected,
      f"clean stories = story lines {story_lines} - exact duplicates {exact} - semantic duplicates {semantic} = {expected} (file has {len(clean)})")
ids = [r["story_id"] for r in clean]
check(len(ids) == len(set(ids)), "story_id values are unique")
check(len({(r['project'], r['story'].lower()) for r in clean}) == len(clean), "no exact duplicate text left in any project")

# every clean story comes from the raw file (merged stories: their first line is in the raw file)
missing = [r["story_id"] for r in clean
           if r["story"] not in raw_texts[r["project"]]
           and not any(r["story"].startswith(t) for t in raw_texts[r["project"]])]
check(not missing, f"every clean story traces back to its raw file ({len(missing)} missing)")

# ---------- 2. review decisions applied ----------
idset = set(ids)
check(all(d["drop"] not in idset and d["keep"] in idset for d in dup),
      "every DUPLICATE: the dropped story is gone and the kept one is present")
flags = read("review/scope_flags.csv")
check(all(f["story_id"] in idset for f in flags),
      f"scope_flags.csv lists final stories only ({len(flags)} rows)")
scope_in_data = {r["story_id"]: r["scope"] for r in clean}
check(sum(v == "process" for v in scope_in_data.values()) == len(flags),
      "scope column in the data matches scope_flags.csv")
group_pairs = [(r["id_a"], r["id_b"]) for r in read("review/duplicate_review.csv")
               if r["decision"] == "GROUP" and r["id_a"] in idset and r["id_b"] in idset]

# ---------- 3. splits ----------
by_project = defaultdict(list)
for r in clean:
    by_project[r["project"]].append(r["story_id"])
for s in SEEDS:
    sp = read(f"data/split_seed{s}.csv")
    check(sorted(r["story_id"] for r in sp) == sorted(ids), f"seed {s}: same stories as clean_stories.csv")
    hidden = {r["story_id"] for r in sp if r["hidden"] == "True"}
    bad = [p for p, sids in by_project.items()
           if len(hidden & set(sids)) != int(len(sids) * 0.20 + 0.5)]
    check(not bad, f"seed {s}: every project hides round_half_up(20% of n) stories (total {len(hidden)})")
    split_pairs = [(a, b) for a, b in group_pairs if (a in hidden) != (b in hidden)]
    check(not split_pairs, f"seed {s}: no GROUP pair has one story visible and the other hidden")
hid = [{r["story_id"] for r in read(f"data/split_seed{s}.csv") if r["hidden"] == "True"} for s in SEEDS]
check(hid[0] != hid[1] and hid[1] != hid[2], "the three seeds hide different stories")

# ---------- 4. deterministic rebuild ----------
tmp = tempfile.mkdtemp()
subprocess.run([sys.executable, "prepare_data.py", RAW, tmp], check=True, capture_output=True)
same = all(hashlib.md5(open(os.path.join("data", f), "rb").read()).hexdigest()
           == hashlib.md5(open(os.path.join(tmp, f), "rb").read()).hexdigest()
           for f in ["clean_stories.csv", "split_seed1.csv", "split_seed2.csv", "split_seed3.csv"])
shutil.rmtree(tmp)
check(same, "running prepare_data.py again gives byte-identical data and splits")

# ---------- 5. what the model actually receives ----------
# Scope of this test: it runs the REAL data path and prompt-building code with a FAKE,
# deterministic model ('mock'), so prompt counts repeat between runs. It proves the
# experiment code never puts hidden stories into prompts; it says nothing about how the
# real models behave, and it detects copies of hidden stories, not paraphrases.
import contextlib, io      # noqa: E401,E402
import llm                 # noqa: E402
import pipeline            # noqa: E402
import run_experiment      # noqa: E402


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


captured = []              # (system, user) of every call
real_call = llm._call
llm._call = lambda model, system, user: (captured.append((system, user)), real_call(model, system, user))[1]
for s in SEEDS:
    sp = read(f"data/split_seed{s}.csv")
    hidden_text, visible_text = defaultdict(list), defaultdict(list)
    for r in sp:
        (hidden_text if r["hidden"] == "True" else visible_text)[r["project"]].append(r["story"])
    captured.clear()
    run_experiment.RESULTS_DIR = tempfile.mkdtemp()
    sys.argv = ["run_experiment.py", "mock", str(s)]
    with contextlib.redirect_stdout(io.StringIO()):
        run_experiment.main()
    shutil.rmtree(run_experiment.RESULTS_DIR)
    users = [u for _, u in captured]
    joined = norm("\n".join(users))
    leaks = sum(norm(h) in joined and h not in visible_text[p] for p, hs in hidden_text.items() for h in hs)
    n_ext = sum(sys_ == pipeline.EXTRACTOR_SYSTEM for sys_, _ in captured)
    n_sug = sum(sys_ == pipeline.SUGGEST_SYSTEM for sys_, _ in captured)
    check(leaks == 0, f"seed {s}: no hidden story appears (normalized text) in any of {len(users)} prompts "
                      f"({n_ext} extraction + {n_sug} suggestion)")
    check(n_ext == 22 and n_sug == 44, f"seed {s}: 1 extraction call and 2 suggestion calls (direct + structured) per project")
    ok = all(all(v in u for v in visible_text[re.search(r"Project: (\S+)", u).group(1)]) for u in users)
    check(ok, f"seed {s}: every prompt (both methods, both stages) contains all visible stories of its project")
    ids_in_prompts = any(re.search(r"\bg\d\d-[a-z0-9-]+-\d{3}\b", u) for u in users)
    check(not ids_in_prompts, f"seed {s}: prompts use neutral IDs (S1..Sk), never dataset IDs that could reveal gaps")

# negative control: leak one hidden story per project and make sure the check sees it
orig_load = run_experiment.load_split


def leaky(seed):
    p = orig_load(seed)
    for d in p.values():
        d["visible"] = d["visible"] + d["hidden"][:1]
    return p


run_experiment.load_split = leaky
run_experiment.RESULTS_DIR = tempfile.mkdtemp()
captured.clear()
sys.argv = ["run_experiment.py", "mock", "1"]
with contextlib.redirect_stdout(io.StringIO()):
    run_experiment.main()
shutil.rmtree(run_experiment.RESULTS_DIR)
run_experiment.load_split = orig_load
joined = norm("\n".join(u for _, u in captured))
caught = sum(norm(x["story"]) in joined for x in read("data/split_seed1.csv") if x["hidden"] == "True")
check(caught >= 22, f"negative control: an injected leak of 1 hidden story per project is detected ({caught} found)")

# ---------- 6. method design and failure handling ----------
sug_sys = {sys_ for sys_, u in captured if "suggestions of missing user stories" in u}
check(sug_sys == {pipeline.SUGGEST_SYSTEM},
      "direct and structured suggestions use the identical system prompt and output format")
run_experiment.RESULTS_DIR = tempfile.mkdtemp()
sys.argv = ["run_experiment.py", "mockbad", "1", "--only", "g04"]
with contextlib.redirect_stdout(io.StringIO()):
    run_experiment.main()
res = json.load(open(os.path.join(run_experiment.RESULTS_DIR, "mockbad_seed1.json")))
fails = json.load(open(os.path.join(run_experiment.RESULTS_DIR, "mockbad_seed1_failures.json")))
shutil.rmtree(run_experiment.RESULTS_DIR)
check("structured" not in res.get("g04-recycling", {}) and "g04-recycling/structured" in fails,
      "an extraction table that misses stories (after retries) is recorded as a failure, not as a result")
check("missing" in json.dumps(fails) or "no row" in json.dumps(fails),
      "the failure record names the stories the table left out")

with open("verification_report.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(report) + "\n\nALL CHECKS PASSED\n")
print("\nALL CHECKS PASSED -> verification_report.txt")
