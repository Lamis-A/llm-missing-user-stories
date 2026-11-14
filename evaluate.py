"""
evaluate.py
Scores every results file (results/*_seed<n>.json) and writes the tables and tests.

Test projects: the 19 projects that are never used as few-shot examples. The three example
projects (g04, g13, g25) are excluded from EVERY comparison, so all methods and budgets are
compared on the same 19 projects.

Matching (fixed before the experiments, never changed after seeing results):
  A suggestion matches a hidden story if the cosine similarity of their sentence
  embeddings (all-MiniLM-L6-v2) is >= 0.60. Each hidden story can be matched by at most
  one suggestion (greedy one-to-one, best pairs first).
  If the embedding model cannot be loaded, evaluation STOPS. A TF-IDF matcher exists only
  for offline testing (--test-tfidf); its results are labelled and must not be reported.

Metrics, per project, then averaged over projects, then over the 3 seeds:
  Recall      = matched hidden stories / hidden stories                    (semantic recall)
  Precision   = matched suggestions / suggestions
  F1          = 2PR/(P+R) per project, then averaged (equal to recall at budget N)
  Overlap     = suggestions whose closest VISIBLE story has similarity >= 0.60
                (also counts new needs from the same domain: read with care)
  Near-copy   = suggestions whose closest VISIBLE story has similarity >= 0.85 (lower is better)
  TermRecall, Unsupported (no or invalid support IDs): extra diagnostics
  Cost per project: model calls, input/output tokens, seconds, format retries.
Reliability: failed projects and projects that needed a format retry -> results/reliability.csv

Statistical tests (two-sided Wilcoxon signed-rank, paired by project, seed-averaged, n = 19):
  stat_tests.csv    each method vs direct, within the same results file (same budget)
  budget_tests.csv  budget 2N vs budget N, for the same method (<run>_k2 vs <run>)

Method names: direct = Direct prompting, structured = Structured prompting,
agent = Critic Agent, react = ReAct Agent (with few-shot, in the *_fs files).

Usage:
    python evaluate.py                # final evaluation (embeddings required)
    python evaluate.py --test-tfidf   # offline pipeline test only
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import statistics
from collections import defaultdict

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
THRESHOLD = 0.60
HIGH_REDUNDANCY = 0.85        # stricter cut: near-copies of a visible story (rephrasing)
LOWER_IS_BETTER = {"overlap", "near_copy"}
TFIDF_THRESHOLD = 0.35          # test mode only

STOP = set("""a an the and or of to in on for with by from at as is are be been being i want
so that can my our their his her its it this these those me we you they will would should
able use using via into about all any each more most other some such than too very not no
user users system""".split())


class Similarity:
    def __init__(self, test_tfidf=False):
        if test_tfidf:
            self.model, self.method, self.threshold = None, "TF-IDF (TEST ONLY - do not report)", TFIDF_THRESHOLD
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            raise SystemExit(f"Cannot load the embedding model '{EMBEDDING_MODEL}' ({e.__class__.__name__}: {e}).\n"
                             "Final evaluation stops here instead of switching method. "
                             "Fix the installation/network and run again.")
        self.method, self.threshold = f"embeddings ({EMBEDDING_MODEL})", THRESHOLD

    def matrix(self, a, b):
        if not a or not b:
            return [[0.0] * len(b) for _ in a]
        if self.model:
            ea = self.model.encode(a, normalize_embeddings=True)
            eb = self.model.encode(b, normalize_embeddings=True)
            return (ea @ eb.T).tolist()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        v = TfidfVectorizer(stop_words="english").fit(a + b)
        return cosine_similarity(v.transform(a), v.transform(b)).tolist()


def greedy_match(sim, threshold):
    pairs = sorted(((sim[i][j], i, j) for i in range(len(sim)) for j in range(len(sim[i]))
                    if sim[i][j] >= threshold), reverse=True)
    used_s, used_h, out = set(), set(), []
    for score, i, j in pairs:
        if i not in used_s and j not in used_h:
            used_s.add(i); used_h.add(j); out.append((i, j, score))
    return out


def terms(texts):
    words = set()
    for t in texts:
        for w in re.findall(r"[a-z]+", t.lower()):
            if len(w) > 3 and w not in STOP:
                words.add(w[:-1] if w.endswith("s") else w)
    return words


def term_recall(visible, hidden, suggestions):
    missing = terms(hidden) - terms(visible)
    return None if not missing else len(missing & terms(suggestions)) / len(missing)


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def load_split(seed):
    vis, scope = defaultdict(list), {}
    with open(f"data/split_seed{seed}.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scope[row["story_id"]] = row.get("scope", "product")
            if row["hidden"] != "True":
                vis[row["project"]].append(row["story"])
    return vis, scope


METHODS = ["direct", "structured", "agent", "react"]
EXCLUDED = {"g04-recycling", "g13-planningpoker", "g25-duraspace"}   # few-shot example projects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-tfidf", action="store_true", help="local pipeline test only")
    args = ap.parse_args()
    sim = Similarity(args.test_tfidf)
    print(f"Matching method: {sim.method}, threshold {sim.threshold}\n")

    seed_scores = defaultdict(lambda: defaultdict(list))   # (model, method, scope) -> metric -> [per seed]
    paired = defaultdict(list)       # (model, scope, metric, method, project) -> [one value per seed]
    per_rows, match_rows, near, redundant_rows = [], [], [], []

    for path in sorted(glob.glob("results/*_seed[123].json")):
        model, seed = re.match(r".*[\\/](\w+)_seed(\d)\.json", path).groups()
        if model.startswith("mock") and not args.test_tfidf:
            continue
        data = json.load(open(path, encoding="utf-8"))
        visible, scope = load_split(seed)
        # methods this model was run with (a model may be run with direct only)
        run_methods = [m for m in METHODS if any(m in r for r in data.values())]
        # only projects where ALL its methods finished, so the methods are compared on the same projects
        complete = {p: r for p, r in data.items() if all(m in r for m in run_methods) and p not in EXCLUDED}
        n_test = len([p for p in visible if p not in EXCLUDED])
        if len(complete) < n_test:
            print(f"WARNING {path}: all methods finished for {len(complete)}/{n_test} projects "
                  "(see the _failures.json file); only those projects are scored.")
        for method in run_methods:
            for sc in ("all", "product"):
                per = defaultdict(list)
                for proj, r in complete.items():
                    res = r[method]
                    hidden = [h for h, i in zip(r["hidden"], r["hidden_ids"])
                              if sc == "all" or scope.get(i, "product") == "product"]
                    if not hidden:
                        continue
                    sugg = res["suggestions"]
                    S = sim.matrix(sugg, hidden)
                    m = greedy_match(S, sim.threshold)
                    rec = len(m) / len(hidden)
                    pre = len(m) / len(sugg) if sugg else 0.0
                    per["recall"].append(rec)
                    per["precision"].append(pre)
                    per["f1"].append(f1(pre, rec))
                    tr = term_recall(visible[proj], hidden, sugg)
                    if tr is not None:
                        per["term_recall"].append(tr)
                    for k, v in (("recall", rec), ("precision", pre), ("f1", f1(pre, rec))):
                        paired[(model, sc, k, method, proj)].append(v)
                    per_rows.append([model, seed, method, sc, proj, len(hidden), len(sugg), len(m),
                                     round(rec, 4), round(pre, 4), round(f1(pre, rec), 4)])
                    if sc != "all":
                        continue
                    # measures that do not depend on the scope: computed once
                    V = sim.matrix(sugg, visible[proj])
                    red = [i for i, row in enumerate(V) if row and max(row) >= sim.threshold]
                    per["overlap"].append(len(red) / len(sugg) if sugg else 0.0)
                    paired[(model, sc, "overlap", method, proj)].append(per["overlap"][-1])
                    # 0.60 also flags new needs from the same domain; >= 0.85 is close to a copy
                    red_hi = [i for i, row in enumerate(V) if row and max(row) >= HIGH_REDUNDANCY]
                    per["near_copy"].append(len(red_hi) / len(sugg) if sugg else 0.0)
                    paired[(model, sc, "near_copy", method, proj)].append(per["near_copy"][-1])
                    for i in red:
                        j = max(range(len(V[i])), key=lambda k: V[i][k])
                        redundant_rows.append([model, seed, method, proj, round(V[i][j], 3), sugg[i],
                                               visible[proj][j]])
                    unsup = [d for d in res["details"] if not d["support_ids"] or d["invalid_support_ids"]]
                    per["unsupported"].append(len(unsup) / len(sugg) if sugg else 0.0)
                    per["calls"].append(res["calls"])
                    per["input_tokens"].append(res["input_tokens"])
                    per["output_tokens"].append(res["output_tokens"])
                    per["seconds"].append(res["seconds"])
                    per["retries"].append(len(res["format_retries"]))
                    per["retried"].append(1.0 if res["format_retries"] else 0.0)
                    for i, j, s in m:
                        match_rows.append([model, seed, method, proj, round(s, 3), sugg[i], hidden[j]])
                    for i, srow in enumerate(S):
                        if srow:
                            j = max(range(len(srow)), key=lambda k: srow[k])
                            near.append([model, seed, method, proj, round(srow[j], 3), sugg[i], hidden[j]])
                for k, v in per.items():
                    seed_scores[(model, method, sc)][k].append(statistics.mean(v))

    if not seed_scores:
        raise SystemExit("No results to evaluate in results/.")

    rows = []
    for (model, method, sc), d in sorted(seed_scores.items()):
        pct = lambda k: round(100 * statistics.mean(d[k]), 1) if d.get(k) else ""
        sd = lambda k: round(100 * statistics.stdev(d[k]), 1) if len(d.get(k, [])) > 1 else ""
        avg = lambda k, n=1: round(statistics.mean(d[k]), n) if d.get(k) else ""
        rows.append({
            "model": model, "method": method, "scope": sc, "seeds": len(d["recall"]),
            "recall_%": pct("recall"), "recall_sd": sd("recall"),
            "precision_%": pct("precision"), "precision_sd": sd("precision"),
            "macro_f1_%": pct("f1"), "macro_f1_sd": sd("f1"),
            "term_recall_%": pct("term_recall"),
            "overlap_%": pct("overlap"),
            "near_copy_%": pct("near_copy"), "unsupported_%": pct("unsupported"),
            "calls_per_project": avg("calls", 2), "input_tokens_per_project": avg("input_tokens", 0),
            "output_tokens_per_project": avg("output_tokens", 0), "seconds_per_project": avg("seconds"),
            "format_retries_per_project": avg("retries", 2), "projects_with_retry_%": pct("retried"),
            "matching": sim.method, "threshold": sim.threshold,
        })

    def write(name, header, data):
        with open(f"results/{name}", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(data)

    with open("results/summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    write("per_project.csv", ["model", "seed", "method", "scope", "project", "hidden", "suggestions",
                              "matches", "recall", "precision", "f1"], per_rows)
    write("matches.csv", ["model", "seed", "method", "project", "similarity", "suggestion", "hidden_story"],
          match_rows)
    write("redundant.csv", ["model", "seed", "method", "project", "similarity", "suggestion",
                            "closest_visible_story"], redundant_rows)

    # stratified sample of redundant pairs for the manual check (10 per similarity band)
    rng = random.Random(0)
    red_sample = []
    for lo, hi in [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]:
        pool = [x for x in redundant_rows if lo <= x[4] < hi]
        for x in rng.sample(pool, min(10, len(pool))):
            red_sample.append(x + [f"{lo:.2f}-{min(hi, 1):.2f}", ""])
    write("redundancy_check.csv", ["model", "seed", "method", "project", "similarity", "suggestion",
                                   "closest_visible_story", "band",
                                   "human_label (same need as the visible story? yes/no)"], red_sample)

    t = sim.threshold
    bands = [(t - 0.15, t - 0.05), (t - 0.05, t), (t, t + 0.05), (t + 0.05, t + 0.15)]
    rng = random.Random(0)
    sample = []
    for lo, hi in bands:
        pool = [x for x in near if lo <= x[4] < hi]
        for x in rng.sample(pool, min(15, len(pool))):
            sample.append(x + [f"{lo:.2f}-{hi:.2f}", "MATCH" if x[4] >= t else "NO MATCH", ""])
    write("threshold_check.csv", ["model", "seed", "method", "project", "similarity", "suggestion",
                                  "closest_hidden_story", "band", "automatic_decision",
                                  "human_label (same need? yes/no)"], sample)

    hdr = (f"{'model':18s} {'method':11s} {'scope':8s} {'recall':>7s} {'prec.':>7s} {'mF1':>6s} "
           f"{'overlap':>8s} {'nearcp':>7s} {'calls':>6s} {'tokens':>8s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        tok = (r["input_tokens_per_project"] or 0) + (r["output_tokens_per_project"] or 0)
        print(f"{r['model']:18s} {r['method']:11s} {r['scope']:8s} {r['recall_%']:6.1f}% {r['precision_%']:6.1f}% "
              f"{r['macro_f1_%']:5.1f}% {str(r['overlap_%']):>7s}% {str(r['near_copy_%']):>6s}% {str(r['calls_per_project']):>6s} "
              f"{tok if r['scope'] == 'all' else '':>8}")
    print(f"\nSaved results/summary.csv, per_project.csv, matches.csv, redundant.csv, "
          f"threshold_check.csv ({len(sample)} pairs), redundancy_check.csv ({len(red_sample)} pairs)")
    reliability()
    stat_tests(paired)
    budget_tests(paired)


def reliability():
    """
    How often each method could not produce a usable reply (failed after all format
    retries) and how often it needed a retry, per model and seed. Failed project/method
    pairs are excluded from the comparison above (both methods are scored only on the
    projects where both finished), so they are reported here instead.
    """
    rows = []
    for path in sorted(glob.glob("results/*_seed[123].json")):
        model, seed = re.match(r".*[\\/](\w+)_seed(\d)\.json", path).groups()
        if model.startswith("mock"):
            continue
        data = json.load(open(path, encoding="utf-8"))
        fpath = path.replace(".json", "_failures.json")
        fails = json.load(open(fpath, encoding="utf-8")) if os.path.exists(fpath) else {}
        projects = [p for p in load_split(seed)[0] if p not in EXCLUDED]
        for method in METHODS:
            if not any(method in r for r in data.values()):
                continue                    # this model was not run with this method
            done = [p for p in projects if method in data.get(p, {})]
            failed = [p for p in projects if method not in data.get(p, {}) and f"{p}/{method}" in fails]
            retried = [p for p in done if data[p][method]["format_retries"]]
            rows.append({"model": model, "seed": seed, "method": method, "projects": len(projects),
                         "finished": len(done), "failed": len(failed), "failed_projects": " ".join(failed),
                         "finished_after_retry": len(retried)})
    if not rows:
        return
    with open("results/reliability.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("\nReliability (failed = no valid reply after 3 attempts; retry = valid on 2nd/3rd attempt):")
    tot = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        print(f"  {r['model']:8s} seed {r['seed']} {r['method']:11s} finished {r['finished']:2d}/{r['projects']} "
              f"| failed {r['failed']} {r['failed_projects']} | needed a retry {r['finished_after_retry']}")
        t = tot[(r["model"], r["method"])]
        t[0] += r["projects"]; t[1] += r["finished"]; t[2] += r["failed"]; t[3] += r["finished_after_retry"]
    for (model, method), (n, ok, fl, rt) in sorted(tot.items()):
        print(f"  {model:8s} all seeds {method:11s} failed {fl}/{n} ({100 * fl / n:.1f}%) | "
              f"needed a retry {rt}/{ok} ({100 * rt / max(ok, 1):.1f}%)")
    print("Saved results/reliability.csv")


def stat_tests(paired):
    """
    Paired comparison of each method against direct (structured vs direct, agent vs direct),
    one pair per PROJECT: each project's value is first averaged over the 3 seeds, so
    projects are the independent units (n = 19).
    Two-sided Wilcoxon signed-rank test (scipy, zero_method='wilcox': ties are dropped),
    plus the number of projects where the method is better / equal / worse than direct.
    For overlap and near-copies, lower is better, so "better" means a lower value.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
        print("scipy is not installed: p-values are skipped.")
    keys = sorted({(m, sc, k) for (m, sc, k, _, _) in paired})
    out = []
    for other in [m for m in METHODS if m != "direct"]:
        for model, sc, metric in keys:
            projects = sorted({p for (m, s, k, meth, p) in paired if (m, s, k) == (model, sc, metric)})
            d, s_ = [], []
            for p in projects:
                a = paired.get((model, sc, metric, other, p))
                b = paired.get((model, sc, metric, "direct", p))
                if a and b and len(a) == len(b):
                    d.append(statistics.mean(b)); s_.append(statistics.mean(a))
            if not d:
                continue
            diffs = [x - y for x, y in zip(s_, d)]
            sign = -1 if metric in LOWER_IS_BETTER else 1
            better = sum(sign * x > 1e-12 for x in diffs)
            worse = sum(sign * x < -1e-12 for x in diffs)
            equal = len(diffs) - better - worse
            p_value, stat = "", ""
            if wilcoxon and better + worse > 0:
                r = wilcoxon(s_, d, zero_method="wilcox", alternative="two-sided")
                p_value, stat = round(float(r.pvalue), 4), round(float(r.statistic), 2)
            out.append({"model": model, "comparison": f"{other} vs direct", "scope": sc, "metric": metric,
                        "projects": len(diffs),
                        "direct_mean_%": round(100 * statistics.mean(d), 1),
                        "method_mean_%": round(100 * statistics.mean(s_), 1),
                        "mean_diff_pts": round(100 * statistics.mean(diffs), 1),
                        "median_diff_pts": round(100 * statistics.median(diffs), 1),
                        "method_better": better, "equal": equal, "method_worse": worse,
                        "wilcoxon_W": stat, "p_value": p_value,
                        "significant_at_0.05": (p_value < 0.05) if p_value != "" else ""})
    if not out:
        return
    with open("results/stat_tests.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print("\nMethod vs direct (paired by project, seed-averaged, two-sided Wilcoxon):")
    print(f"{'model':12s} {'comparison':22s} {'scope':8s} {'metric':14s} {'direct':>7s} {'method':>7s} "
          f"{'diff':>6s} {'b/e/w':>9s} {'p':>7s}")
    for r in out:
        print(f"{r['model']:12s} {r['comparison']:22s} {r['scope']:8s} {r['metric']:14s} "
              f"{r['direct_mean_%']:6.1f}% {r['method_mean_%']:6.1f}% {r['mean_diff_pts']:+5.1f} "
              f"{r['method_better']:>3d}/{r['equal']}/{r['method_worse']:<3d} {str(r['p_value']):>7s}")
    print("b/e/w = projects where the method is better / equal / worse than direct. Saved results/stat_tests.csv")


def _wilcoxon_row(x, y, metric):
    """x = new values, y = baseline values (one per project). Returns the test summary."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    diffs = [a - b for a, b in zip(x, y)]
    sign = -1 if metric in LOWER_IS_BETTER else 1
    better = sum(sign * v > 1e-12 for v in diffs)
    worse = sum(sign * v < -1e-12 for v in diffs)
    p_value = ""
    if wilcoxon and better + worse > 0:
        p_value = round(float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue), 4)
    return {"projects": len(diffs), "baseline_mean_%": round(100 * statistics.mean(y), 1),
            "new_mean_%": round(100 * statistics.mean(x), 1), "mean_diff_pts": round(100 * statistics.mean(diffs), 1),
            "better": better, "equal": len(diffs) - better - worse, "worse": worse, "p_value": p_value}


def budget_tests(paired):
    """
    Budget 2N vs budget N for the same method: <run>_k2 is compared with <run>
    (deepseek_k2 vs deepseek, deepseek_k2_agent vs deepseek_agent, deepseek_k2_fs vs deepseek_fs),
    paired by project, seed-averaged, two-sided Wilcoxon. "better" = higher recall / precision / F1,
    lower overlap / near-copies.
    """
    models = {m for (m, _, _, _, _) in paired}
    out = []
    for m2 in sorted(x for x in models if "_k2" in x):
        m1 = m2.replace("_k2", "")
        if m1 not in models:
            continue
        for sc, metric, method in sorted({(s, k, me) for (m, s, k, me, _) in paired if m == m2}):
            ps = sorted(p for (m, s, k, me, p) in paired if (m, s, k, me) == (m2, sc, metric, method)
                        and (m1, sc, metric, method, p) in paired)
            if not ps:
                continue
            x = [statistics.mean(paired[(m2, sc, metric, method, p)]) for p in ps]
            y = [statistics.mean(paired[(m1, sc, metric, method, p)]) for p in ps]
            out.append({"run_N": m1, "run_2N": m2, "method": method, "scope": sc, "metric": metric,
                        **_wilcoxon_row(x, y, metric)})
    if not out:
        return
    with open("results/budget_tests.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
    print("\nBudget 2N vs N, same method (paired by project, seed-averaged, two-sided Wilcoxon), scope = all:")
    for r in out:
        if r["scope"] == "all":
            print(f"  {r['run_N']:16s} {r['method']:11s} {r['metric']:10s} {r['baseline_mean_%']:5.1f}% -> "
                  f"{r['new_mean_%']:5.1f}%  b/e/w {r['better']}/{r['equal']}/{r['worse']}  p {r['p_value']}")
    print("Saved results/budget_tests.csv")


if __name__ == "__main__":
    main()
