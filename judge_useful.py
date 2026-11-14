"""
judge_useful.py
Complementary measure: are the suggested stories plausible and useful, whether or not they
are one of the hidden stories? Same judge as judge.py (Llama 3.3 70B via OpenRouter).
It does NOT replace recall: a high value means "reasonable ideas", not "recovered stories".

For every suggestion the judge sees the project's VISIBLE stories (never the hidden ones)
and answers three yes/no criteria:
    relevant   - in the project's scope; fits its roles, terms and domain
    novel      - not already covered by a visible story (a rephrasing is not novel)
    well_formed- one clear need written as a user story (role, goal, benefit)
    useful     = relevant AND novel AND well_formed
Suggestion sets are shuffled, the method is never shown, identical sets are judged once.

Steps (Colab, OPENROUTER_API_KEY set):
    python judge_useful.py --sample      # 1. writes results/judge_useful/validation_sample.csv (no API)
                                         #    -> label it yourself BEFORE step 2
    python judge_useful.py --validate    # 2. judge the sample, agreement with your labels
    python judge_useful.py               # 3. all methods (resumable), then the tables
    python judge_useful.py --report      #    tables only, from the cache
    python judge_useful.py --mock        #    offline test with a fake judge (do not report)
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

import judge as J            # same model, client, JSON parsing and suggestion sets as judge.py
import evaluate as ev        # load_split (visible stories) and the similarity model

OUT = os.path.join("results", "judge_useful")
CRITERIA = ("relevant", "novel", "well_formed")
SAMPLE_SIZE = 40

SYSTEM = """You are an expert agile requirements engineer acting as an impartial judge.
You receive the EXISTING user stories of a software project and a list of SUGGESTED new stories.
For each suggestion answer three questions with "yes" or "no":
- "relevant": is it within this project's scope, consistent with its roles, terms and domain?
  ("no" for generic features that do not fit this particular project.)
- "novel": is the need NOT already covered by an existing story? A rephrasing, a role swap of
  the same need, or a part of an existing story is "no".
- "well_formed": does it state ONE clear need as a user story (role, goal, benefit)?
  ("no" if it combines several unrelated functions or is vague.)
Judge each suggestion on its own. Do not reward length or technical detail.
Reply ONLY with JSON:
{"judgments": [{"id": "A1", "relevant": "yes", "novel": "no", "well_formed": "yes", "reason": "short"}]}"""


def yes(x):
    return str(x).strip().lower() in ("yes", "true", "1")


CHUNK = 20      # at most 20 suggestions per call: longer replies are cut off by the judge's output limit


def judge_suggestions(project, visible, suggestions, order_seed):
    """Returns {suggestion text: {criteria..., useful, reason}} for every suggestion.
    Large sets (budget 2N) are judged in chunks of CHUNK suggestions, each chunk against
    the same visible stories, so every suggestion is judged exactly as in a small set."""
    if len(suggestions) <= CHUNK:
        return _judge_chunk(project, visible, suggestions, order_seed)
    order = list(range(len(suggestions)))
    random.Random(order_seed).shuffle(order)
    items, provs, attempts = {}, [], 0
    for c in range(0, len(order), CHUNK):
        part = [suggestions[i] for i in order[c:c + CHUNK]]
        r = _judge_chunk(project, visible, part, order_seed + c)
        items.update(r["items"]); provs.append(r["provider"]); attempts += r["attempts"]
    return {"items": items, "provider": "+".join(sorted(set(map(str, provs)))), "attempts": attempts,
            "chunks": len(provs)}


def _judge_chunk(project, visible, suggestions, order_seed):
    order = list(range(len(suggestions)))
    random.Random(order_seed).shuffle(order)
    ids = {f"A{k + 1}": i for k, i in enumerate(order)}
    user = (f"Project: {project}\n\nEXISTING stories ({len(visible)}):\n" +
            "\n".join(f"S{i + 1}: {t}" for i, t in enumerate(visible)) +
            f"\n\nSUGGESTED stories ({len(suggestions)}):\n" +
            "\n".join(f"{a}: {suggestions[i]}" for a, i in ids.items()) + "\n\nJudge every suggestion.")
    last = ""
    for attempt in range(3):
        text, prov = J.call(SYSTEM, user if attempt == 0 else
                            user + f"\n\nYour previous reply was not usable ({last}). Reply ONLY with the JSON.")
        data = J.parse_json(text)
        js = data.get("judgments") if isinstance(data, dict) else None
        if not isinstance(js, list):
            last = "no judgments list"
            continue
        got = {str(j.get("id")): j for j in js if isinstance(j, dict) and str(j.get("id")) in ids}
        if len(got) < len(ids):
            last = f"judgments for {len(got)} of {len(ids)} suggestions"
            continue
        out = {}
        for a, i in ids.items():
            j = got[a]
            c = {k: yes(j.get(k)) for k in CRITERIA}
            out[suggestions[i]] = dict(c, useful=all(c.values()), reason=str(j.get("reason", ""))[:300])
        return {"items": out, "provider": prov, "attempts": attempt + 1}
    raise RuntimeError(f"{project}: no usable reply after 3 attempts ({last})")


def key(project, visible, suggestions):
    return hashlib.sha256(json.dumps([project, sorted(visible), sorted(suggestions)]).encode()).hexdigest()[:24]


def all_sets():
    """(seed, project, method) -> (visible, suggestions)"""
    vis = {s: ev.load_split(s)[0] for s in (1, 2, 3)}
    return {(s, p, m): (vis[s][p], sugg) for (s, p, m), (_, sugg) in J.suggestion_sets().items()}


# ---------------- run and report ----------------
def run_all():
    os.makedirs(OUT, exist_ok=True)
    cpath = os.path.join(OUT, "judgments.json")
    cache = J.load(cpath)
    sets, index, todo = all_sets(), {}, {}
    for (s, p, m), (visible, sugg) in sets.items():
        k = key(p, visible, sugg)
        index[f"{s}|{p}|{m}"] = k
        if k not in cache:
            todo.setdefault(k, (p, visible, sugg))
    print(f"{len(sets)} suggestion sets, {len(set(index.values()))} distinct, {len(todo)} still to judge")
    for n, (k, (p, visible, sugg)) in enumerate(todo.items(), 1):
        try:
            res = judge_suggestions(p, visible, sugg, int(k[:8], 16))
        except RuntimeError as e:
            print(f"   FAILED {p}: {e} -> run again later")
            continue
        res.update(project=p, judge_model=J.JUDGE_MODEL, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        cache[k] = res
        json.dump(cache, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        u = sum(v["useful"] for v in res["items"].values())
        print(f"[{n}/{len(todo)}] {p}: useful {u}/{len(sugg)} | provider {res['provider']}")
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    json.dump(dict(J.USAGE, judge_model=J.JUDGE_MODEL, date=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(OUT, f"usage_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"), indent=1)
    print(f"Usage this run: {J.USAGE['calls']} calls, {J.USAGE['input_tokens']}+{J.USAGE['output_tokens']} tokens")


def report():
    from scipy.stats import wilcoxon
    cache = J.load(os.path.join(OUT, "judgments.json"))
    index = J.load(os.path.join(OUT, "index.json"))
    # suggestions credited to a hidden story by judge.py (same/partial), to count "useful beyond the hidden stories"
    jc, ji = J.load(os.path.join(J.OUT, "judgments.json")), J.load(os.path.join(J.OUT, "index.json"))
    sets = all_sets()
    vals, rows = {}, []
    measures = ("useful",) + CRITERIA + ("useful_not_hidden",)
    for kk, k in index.items():
        if k not in cache:
            continue
        s, p, m = kk.split("|")
        if m not in J.JUDGED_METHODS or p in J.DEV_PROJECTS or (int(s), p, m) not in sets:
            continue
        items = cache[k]["items"]
        sugg = sets[(int(s), p, m)][1]
        credited = set()
        if kk in ji and ji[kk] in jc:
            credited = {j["suggestion"] for j in jc[ji[kk]]["judgments"] if j["label"] in ("same", "partial")}
        r = {c: statistics.mean(items[x][c] for x in sugg) for c in ("useful",) + CRITERIA}
        r["useful_not_hidden"] = statistics.mean(items[x]["useful"] and x not in credited for x in sugg)
        vals.setdefault((m, p), []).append(r)
        rows.append([m, s, p, len(sugg)] + [round(r[c], 4) for c in measures])
    if not vals:
        raise SystemExit("No judgments yet - run: python judge_useful.py")
    per = {mp: {c: statistics.mean(x[c] for x in v) for c in measures} for mp, v in vals.items()}
    methods = [m for m in J.JUDGED_METHODS if any(mm == m for mm, _ in per)]
    summary = []
    for m in methods:
        ps = [v for (mm, p), v in per.items() if mm == m]
        summary.append([m, "19 test projects", len(ps)] + [round(100 * statistics.mean(x[c] for x in ps), 1)
                                                           for c in measures])
    tests = []
    for a, b in J.TESTS:
        ps = sorted(p for (m, p) in per if m == a and (b, p) in per)
        if not ps:
            continue
        for c in ("useful", "novel"):
            x, y = [per[(a, p)][c] for p in ps], [per[(b, p)][c] for p in ps]
            d = [u - v for u, v in zip(x, y)]
            bt, wt = sum(v > 1e-12 for v in d), sum(v < -1e-12 for v in d)
            pv = round(float(wilcoxon(x, y).pvalue), 4) if bt + wt and len(ps) >= 10 else ""
            tests.append([f"{a} vs {b}", c, len(ps), round(100 * statistics.mean(x), 1),
                          round(100 * statistics.mean(y), 1), round(100 * statistics.mean(d), 1),
                          f"{bt}/{len(d) - bt - wt}/{wt}", pv])

    def write(name, header, data):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(data)
    hdr = [f"{c}_%" for c in measures]
    write("useful_per_project.csv", ["method", "seed", "project", "suggestions"] + list(measures), rows)
    write("useful_summary.csv", ["method", "projects_scope", "projects"] + hdr, summary)
    write("useful_tests.csv", ["comparison", "measure", "projects", "method_%", "baseline_%", "diff_pts",
                               "better/equal/worse", "p_value"], tests)
    print(f"Judge: {J.JUDGE_MODEL} - share of suggestions (complementary to recall)\n")
    print(f"{'method':16s} {'projects':12s} {'n':>3s} " + " ".join(f"{c:>11s}" for c in measures))
    for r in summary:
        print(f"{r[0]:16s} {r[1]:12s} {r[2]:3d} " + " ".join(f"{v:10.1f}%" for v in r[3:]))
    print("\n'useful_not_hidden' = useful AND not credited to any hidden story by judge.py")
    print("\nWilcoxon (paired by project, seed-averaged):")
    for t in tests:
        print(f"  {t[0]:28s} {t[1]:7s} n={t[2]:2d}  {t[3]:5.1f}% vs {t[4]:5.1f}%  diff {t[5]:+5.1f} | "
              f"{t[6]:>8s} | p {t[7] if t[7] != '' else 'n/a'}")
    print(f"\nSaved {OUT}/useful_summary.csv, useful_tests.csv, useful_per_project.csv")


# ---------------- human validation ----------------
def make_sample(test_tfidf=False):
    """A stratified sample (methods x projects, seed 1) for YOUR labels, with the closest visible stories."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "validation_sample.csv")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists already - delete it first if you really want a new sample")
    sets = all_sets()
    rng = random.Random(42)
    pool = [(p, m, x) for (s, p, m), (_, sugg) in sets.items() if s == 1 and m != "agent_draft"
            for x in sugg]
    seen, picks = set(), []
    for p, m, x in rng.sample(pool, len(pool)):           # one suggestion per (project, method) first
        if (p, m) not in seen and x not in {q[2] for q in picks}:
            seen.add((p, m)); picks.append((p, m, x))
        if len(picks) == SAMPLE_SIZE:
            break
    sim = ev.Similarity(test_tfidf)
    rows = []
    for i, (p, m, x) in enumerate(picks, 1):
        visible = sets[(1, p, "direct")][0]
        S = sim.matrix([x], visible)[0]
        top = sorted(range(len(visible)), key=lambda k: -S[k])[:3]
        rows.append([f"U{i}", p, x] + [f"{visible[k]} ({S[k]:.2f})" for k in top] + ["", "", "", ""])
    json.dump({x: m for p, m, x in picks}, open(os.path.join(OUT, "validation_sample_methods.json"), "w"), indent=1)
    rng.shuffle(rows)                                       # the method is not shown in the file
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "project", "suggestion", "closest_visible_1", "closest_visible_2", "closest_visible_3",
                    "relevant (yes/no)", "novel (yes/no)", "well_formed (yes/no)", "notes"])
        w.writerows(rows)
    print(f"Saved {path} ({len(rows)} suggestions). Label the three yes/no columns yourself, then run --validate.\n"
          "Tip: the 3 closest visible stories help to judge 'novel'; the full list is in data/split_seed1.csv.")


def validate():
    path = os.path.join(OUT, "validation_sample.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = {c: next(h for h in rows[0] if h.startswith(c)) for c in CRITERIA}
    rows = [r for r in rows if all(r[cols[c]].strip().lower() in ("yes", "no") for c in CRITERIA)]
    if not rows:
        raise SystemExit("No labelled rows: fill the three yes/no columns in validation_sample.csv first")
    vis = ev.load_split(1)[0]
    by_proj = {}
    for r in rows:
        by_proj.setdefault(r["project"], []).append(r["suggestion"])
    pred = {}
    for p, sugg in by_proj.items():
        pred.update(judge_suggestions(p, vis[p], sugg, 7)["items"])
    print(f"Validation on {len(rows)} labelled suggestions, judge {J.JUDGE_MODEL}")
    out = []
    for c in CRITERIA + ("useful",):
        h = [all(r[cols[k]].strip().lower() == "yes" for k in CRITERIA) if c == "useful"
             else r[cols[c]].strip().lower() == "yes" for r in rows]
        j = [pred[r["suggestion"]][c] for r in rows]
        agree = sum(a == b for a, b in zip(h, j)) / len(h)
        pe = (sum(h) * sum(j) + (len(h) - sum(h)) * (len(j) - sum(j))) / len(h) ** 2
        kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
        print(f"  {c:12s} you yes {sum(h):2d} | judge yes {sum(j):2d} | agreement {100 * agree:5.1f}% | kappa {kappa:.2f}")
        out.append([c, len(h), sum(h), sum(j), round(agree, 3), round(kappa, 3)])
    with open(os.path.join(OUT, "validation_result.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["criterion", "n", "human_yes", "judge_yes", "agreement", "kappa"]); w.writerows(out)
        w.writerow([])
        w.writerow(["suggestion"] + [f"judge_{c}" for c in CRITERIA] + ["judge_reason"])
        for r in rows:
            q = pred[r["suggestion"]]
            w.writerow([r["suggestion"]] + [q[c] for c in CRITERIA] + [q["reason"]])
    print(f"Saved {OUT}/validation_result.csv")


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--mock", action="store_true", help="offline test with a fake judge (do not report)")
    args = ap.parse_args(argv)
    if args.mock:
        OUT = os.path.join("results", "judge_useful_mock")
        J.OUT = os.path.join("results", "judge_mock")

        def fake(system, user):
            ids = re.findall(r"^(A\d+): ", user, flags=re.M)
            return json.dumps({"judgments": [{"id": a, "relevant": "yes", "novel": ["yes", "no"][i % 2],
                                              "well_formed": "yes", "reason": "m"} for i, a in enumerate(ids)]}), "mock"
        J.call = fake
    if args.sample:
        make_sample(test_tfidf=args.mock)
    elif args.validate:
        validate()
    elif args.report:
        report()
    else:
        run_all()
        report()


if __name__ == "__main__":
    main()
