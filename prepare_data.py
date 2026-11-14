"""
prepare_data.py  (version 2)
Builds the clean dataset and the three hide/visible splits from the raw Dalpiaz files
and the two manual review files.

Usage:
    python prepare_data.py "<folder with the 22 raw .txt files>" data

Inputs (review/ folder, filled by manual review):
    line_review.csv        decision for every source line that is not a story
    duplicate_review.csv   candidate pairs: DUPLICATE (clearly equivalent: drop one),
                           GROUP (one reveals most of the other's need: keep both,
                           always on the same split side) or DISTINCT (related or not,
                           different needs: split independently)

Splitting is done at GROUP level ("group-level hiding"): a group of near-equivalent
stories is hidden or shown as a whole, so a hidden need never has an equivalent in
the visible part. Related but different stories are NOT grouped: they are the context
the agents are expected to use.
    scope_flags.csv        stories marked 'process' (team activity, not software
                           behaviour). They are kept; evaluation reports results
                           with and without them.

Outputs (data/ folder):
    clean_stories.csv      project, story_id, source_line, story, scope, group_id
    split_seed{1,2,3}.csv  same columns + hidden (True/False)
    dataset_stats.csv      counts per project
    cleaning_log.csv       every merge, removal and change, with its reason
"""
import csv
import glob
import os
import random
import re
import sys
from collections import defaultdict

HIDE_RATIO = 0.20
SEEDS = [1, 2, 3]
STORY_START = re.compile(r"^as\s", re.IGNORECASE)
REVIEW_DIR = "review"


def n_hidden(n):
    """Round half up (not Python's round-half-to-even): 52 -> 10.4 -> 10, 53 -> 10.6 -> 11."""
    return int(n * HIDE_RATIO + 0.5)


def read_lines(path):
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    text = text.replace("﻿", "")
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"'))
    return [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]


def load_line_review():
    """Manual decision for every source line that does not start with 'As'."""
    out = {}
    with open(os.path.join(REVIEW_DIR, "line_review.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["project"], int(r["source_line"]))] = (r["decision"], r["reason"])
    return out


def parse_project(project, lines, log, line_review):
    """'As ...' starts a story. Every other non-empty line must have a manual decision in
    review/line_review.csv: CONTINUATION (joined to the previous story) or HEADING (dropped).
    An unreviewed line stops the script, so nothing is merged or dropped by a blind rule."""
    stories = []                                   # [text, first source line]
    for no, line in enumerate(lines, 1):
        if not line:
            continue
        if STORY_START.match(line):
            stories.append([line, no])
            continue
        if (project, no) not in line_review:
            raise SystemExit(f"Line {no} of {project} is not a story and has no decision "
                             f"in review/line_review.csv: {line!r}")
        decision, reason = line_review[(project, no)]
        if decision == "CONTINUATION":
            log.append([project, "", no, "MERGED_CONTINUATION",
                        f"joined to the story on line {stories[-1][1]}: {reason}", line])
            stories[-1][0] += " " + line
        else:
            log.append([project, "", no, "REMOVED_HEADING", reason, line])
    unique, seen = [], {}
    for text, no in stories:
        key = text.lower()
        if key in seen:
            log.append([project, "", no, "REMOVED_EXACT_DUPLICATE",
                        f"identical to the story on line {seen[key]}", text])
            continue
        seen[key] = no
        unique.append((text, no))
    return unique


def load_review():
    drops, groups = {}, []
    with open(os.path.join(REVIEW_DIR, "duplicate_review.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["decision"] == "DUPLICATE":
                drops[r["drop"]] = (r["keep"], r["reason"])
            elif r["decision"] == "GROUP":
                groups.append((r["id_a"], r["id_b"]))
    scope = {}
    with open(os.path.join(REVIEW_DIR, "scope_flags.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            scope[r["story_id"]] = r["scope"]
    return drops, groups, scope


def components(ids, pairs):
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    comp = defaultdict(list)
    for i in ids:
        comp[find(i)].append(i)
    return list(comp.values())


def split(project, units, n_hide, seed):
    """Hide whole units (a single story or a GROUP of related stories) until ~20% is hidden."""
    rng = random.Random(f"{seed}:{project}")
    order = sorted(units, key=lambda u: u[0])
    rng.shuffle(order)
    hidden = set()
    for u in order:
        if len(hidden) + len(u) <= n_hide:
            hidden.update(u)
        if len(hidden) == n_hide:
            break
    return hidden


def main(src, out):
    os.makedirs(out, exist_ok=True)
    drops, group_pairs, scope = load_review()
    line_review = load_line_review()
    log, data = [], {}

    for f in sorted(glob.glob(os.path.join(src, "*.txt"))):
        project = os.path.splitext(os.path.basename(f))[0]
        stories = parse_project(project, read_lines(f), log, line_review)
        rows = []
        for i, (text, line_no) in enumerate(stories, 1):
            sid = f"{project}-{i:03d}"              # same IDs as the first cleaning
            if sid in drops:
                keep, why = drops[sid]
                log.append([project, sid, line_no, "REMOVED_SEMANTIC_DUPLICATE",
                            f"equivalent to {keep}: {why}", text])
                continue
            rows.append({"project": project, "story_id": sid, "source_line": line_no,
                         "story": text, "scope": scope.get(sid, "product")})
        ids = [r["story_id"] for r in rows]
        comps = components(ids, group_pairs)
        gid = {}
        for k, comp in enumerate(sorted(comps, key=lambda c: c[0]), 1):
            for sid in comp:
                gid[sid] = f"{project}-G{k:03d}" if len(comp) > 1 else ""
        for r in rows:
            r["group_id"] = gid[r["story_id"]]
        data[project] = (rows, comps)

    cols = ["project", "story_id", "source_line", "story", "scope", "group_id"]
    with open(os.path.join(out, "clean_stories.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rows, _ in data.values():
            w.writerows(rows)

    hidden_by_seed = {}
    for seed in SEEDS:
        hidden_by_seed[seed] = {}
        with open(os.path.join(out, f"split_seed{seed}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols + ["hidden"])
            w.writeheader()
            for project, (rows, comps) in data.items():
                hid = split(project, comps, n_hidden(len(rows)), seed)
                hidden_by_seed[seed][project] = hid
                for r in rows:
                    w.writerow({**r, "hidden": r["story_id"] in hid})

    # leakage check: related stories must never be split between visible and hidden
    for seed, per in hidden_by_seed.items():
        for project, (rows, comps) in data.items():
            for comp in comps:
                states = {sid in per[project] for sid in comp}
                assert len(states) == 1, f"group split across sides: {comp} seed {seed}"

    with open(os.path.join(out, "dataset_stats.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "stories", "product_scope", "process_scope", "groups",
                    "hidden_seed1", "hidden_seed2", "hidden_seed3", "visible_seed1"])
        tot = [0, 0, 0]
        for project, (rows, comps) in data.items():
            n = len(rows)
            proc = sum(r["scope"] == "process" for r in rows)
            hs = [len(hidden_by_seed[s][project]) for s in SEEDS]
            w.writerow([project, n, n - proc, proc, sum(len(c) > 1 for c in comps), *hs, n - hs[0]])
            tot[0] += n; tot[1] += proc; tot[2] += hs[0]
            print(f"{project:24s} {n:4d} stories | process {proc:3d} | hidden {hs}")
        print(f"\nProjects: {len(data)}  Stories: {tot[0]}  Process-flagged: {tot[1]}  Hidden (seed 1): {tot[2]}")

    with open(os.path.join(out, "cleaning_log.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "story_id", "source_line", "action", "reason", "text"])
        w.writerows(log)
    print(f"Cleaning log: {len(log)} entries")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
