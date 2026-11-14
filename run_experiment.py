"""
run_experiment.py
Runs the prompting methods on every test project for one model and one split (seed).

Methods (names in the report):
    direct      Direct prompting (baseline, 1 call)
    structured  Structured prompting (extraction table, then suggestions)
    agent       Critic Agent (draft, a tool flags near-copies of the backlog, the model revises)
    react       ReAct Agent (decides when to search the visible backlog; needs --fewshot)

Examples:
    python run_experiment.py deepseek 1 --methods direct,structured            # Experiment 1, budget N
    python run_experiment.py deepseek 1 --methods agent                        # Critic Agent, budget N
    python run_experiment.py deepseek 1 --fewshot --methods direct,react       # Experiment 2, budget N
    python run_experiment.py deepseek 1 --fewshot --methods direct,react --budget 2   # Experiment 2, budget 2N
    python run_experiment.py mock 1                                            # free offline test

Results: results/<run>_seed<seed>.json (one entry per project, every method inside), where
<run> = deepseek [+ _k2 for --budget 2] [+ _fs for --fewshot] [+ _agent for the Critic Agent].
Failures: results/<run>_seed<seed>_failures.json (never saved as results).
If a run stops, run the same command again: finished project/method pairs are skipped.
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict

import llm
from llm import MODELS
from pipeline import run_method, run_agent, run_react, FEWSHOT_PROJECTS

DATA_DIR = "data"
RESULTS_DIR = "results"
METHODS = ["direct", "structured", "agent", "react"]
PAUSE_SECONDS = 4          # small pause between projects to respect rate limits


def load_split(seed):
    """Only hidden=False stories ever reach the model; hidden ones are kept for evaluation."""
    projects = defaultdict(lambda: {"visible": [], "hidden": [], "hidden_ids": []})
    with open(os.path.join(DATA_DIR, f"split_seed{seed}.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["hidden"] == "True":
                projects[row["project"]]["hidden"].append(row["story"])
                projects[row["project"]]["hidden_ids"].append(row["story_id"])
            else:
                projects[row["project"]]["visible"].append(row["story"])
    for name, d in projects.items():
        assert not set(d["hidden"]) & set(d["visible"]), f"hidden story leaked into visible set: {name}"
    return projects


def main(argv=None):
    """argv lets a notebook call main() directly."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS.keys())
    ap.add_argument("seed", type=int, choices=[1, 2, 3])
    ap.add_argument("--only", help="run only projects whose name starts with this, e.g. g04")
    ap.add_argument("--methods", default="direct,structured",
                    help="comma-separated, e.g. --methods direct,structured")
    ap.add_argument("--budget", type=int, default=1,
                    help="suggestions per hidden story (default 1). --budget 2 asks for twice as many "
                         "suggestions; results are saved as <model>_k2_seed<n>.json")
    ap.add_argument("--fewshot", action="store_true",
                    help="give the same worked examples (from g04, g13 and g25) to every "
                         "method; those 3 projects are skipped; saved as <model>_fs_seed<n>.json")
    args = ap.parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",")]
    assert all(m in METHODS for m in methods), f"methods must be from {METHODS}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_name = args.model if args.budget == 1 else f"{args.model}_k{args.budget}"
    if "react" in methods and not args.fewshot:
        raise SystemExit("the ReAct agent is defined with few-shot examples: add --fewshot")
    if args.fewshot:
        run_name += "_fs"
    if "agent" in methods:            # the agent run has its own file with its own direct baseline
        assert methods == ["agent"], "run the agent alone: --methods agent"
        run_name += "_agent"
    out_path = os.path.join(RESULTS_DIR, f"{run_name}_seed{args.seed}.json")
    fail_path = os.path.join(RESULTS_DIR, f"{run_name}_seed{args.seed}_failures.json")
    results = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    failures = json.load(open(fail_path, encoding="utf-8")) if os.path.exists(fail_path) else {}

    projects = load_split(args.seed)
    not_done = []
    for name, d in projects.items():
        if args.only and not name.startswith(args.only):
            continue
        if args.fewshot and name in FEWSHOT_PROJECTS:
            continue                  # their stories are the few-shot examples: never tested
        entry = results.setdefault(name, {"hidden": d["hidden"], "hidden_ids": d["hidden_ids"]})
        todo = [m for m in methods if m not in entry]
        if not todo:
            print(f"skip {name} (already done)")
            continue
        n = args.budget * len(d["hidden"])   # same number of suggestions for every method
        print(f"{name}: {len(d['visible'])} visible, {n} suggestions per method")
        for method in todo:
            try:
                if method == "agent":
                    res_direct, res = run_agent(args.model, name, d["visible"], n)
                    entry["direct"] = res_direct       # same first call, kept as the paired baseline
                elif method == "react":
                    res = run_react(args.model, name, d["visible"], n)
                else:
                    res = run_method(method, args.model, name, d["visible"], n, fewshot=args.fewshot)
            except Exception as e:
                not_done.append(f"{name}/{method}")
                failures.setdefault(f"{name}/{method}", []).append(
                    {"error": f"{e.__class__.__name__}: {e}", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                json.dump(failures, open(fail_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"   {method:10s} FAILED ({e.__class__.__name__}: {e}) -> recorded; run again later")
                continue
            entry[method] = res
            json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            if method == "react":
                print(f"   react: {res['tool_calls']} searches"
                      + (f", {res['finish_rejected']} early finish rejected" if res["finish_rejected"] else "") + ": "
                      + " | ".join(x.get("query", "finish")[:40] for x in res["agent_log"]))
            if method == "agent":
                print("   agent tool: " + ", ".join(f"round {x['round']}: {x['flagged']} flagged"
                                                    for x in res["agent_log"]))
            print(f"   {method:10s} {res['n_returned']}/{n} suggestions | calls {res['calls']} | "
                  f"tokens {res['input_tokens']}+{res['output_tokens']} | {res['seconds']}s"
                  + (f" | format retries {len(res['format_retries'])}" if res["format_retries"] else ""))
        if args.model not in ("mock", "mockbad"):
            time.sleep(PAUSE_SECONDS)

    if llm.MODEL_INFO:                      # exact model version and settings, for the report
        json.dump(llm.MODEL_INFO, open(os.path.join(RESULTS_DIR, f"{run_name}_model_info.json"), "w"),
                  indent=1)
    results = {k: v for k, v in results.items() if any(m in v for m in METHODS)}
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    done = sum(all(m in v for m in methods) for v in results.values())
    total = len([p for p in projects if not (args.fewshot and p in FEWSHOT_PROJECTS)])
    print(f"\nSaved: {out_path}  |  {'+'.join(methods)} finished for {done}/{total} projects")
    if not_done:
        print(f"NOT finished: {', '.join(not_done)}  -> run the same command again to complete them.")


if __name__ == "__main__":
    main()
