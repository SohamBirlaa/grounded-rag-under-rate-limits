"""Re-grade saved eval answers OFFLINE (no API calls) and check graders against your human grades.

  uv run python scripts/regrade.py                 # compare policies, write summary
  uv run python scripts/regrade.py --apply judge_primary   # also rewrite answer_grade in results.csv

Inputs : data/live/traces.jsonl, data/live/eval_review.csv (your human_grade column), questions.json
Outputs: regrade_summary.json next to the traces file; with --apply, data/live/results.csv (backup kept as .bak)

Policies compared:
  rules            key-fact recall >= 2/3 and cites the target paper (the original grader)
  rules_and_judge  rules AND the LLM judge
  judge_primary    the LLM judge (when it ran) and cites the target paper
For each policy: Table 3 percentages per setting, and agreement with human_grade where you filled it.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from _live_common import ROOT, load_questions

from relay.live.evals import grade_answer

POLICIES = ("rules", "rules_and_judge", "judge_primary")
LABELS = ("correct", "wrong", "refused_correctly", "refused_wrongly")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(ROOT / "data/live/traces.jsonl"))
    ap.add_argument("--review", default=str(ROOT / "data/live/eval_review.csv"))
    ap.add_argument("--results", default=str(ROOT / "data/live/results.csv"))
    ap.add_argument("--apply", choices=POLICIES, default=None)
    a = ap.parse_args()

    qs = {q["question_id"]: q for q in load_questions()}
    human = {}
    if Path(a.review).exists():
        # Excel's "CSV UTF-8" adds a BOM, and some regional settings save with ';' instead of ','.
        with open(a.review, newline="", encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            header = sample.splitlines()[0] if sample else ""
            delim = ";" if header.count(";") > header.count(",") else ","  # decide from the header only
            for r in csv.DictReader(f, delimiter=delim):
                r = {(k or "").strip().lstrip("\ufeff"): v for k, v in r.items()}
                if "request_id" not in r:
                    raise SystemExit(f"Could not find a request_id column in {a.review}; columns: {list(r)}")
                h = (r.get("human_grade") or "").strip().lower()
                if h in LABELS:
                    human[r["request_id"]] = h

    rows = []
    for line in open(a.traces, encoding="utf-8"):
        t = json.loads(line)
        if t.get("scenario") != "retrieval_eval" or t.get("question_id") not in qs:
            continue
        rows.append(t)
    latest = {}
    for t in rows:  # a resumed or re-run eval may log a request twice: keep the last
        latest[t["request_id"]] = t

    out = {p: defaultdict(lambda: defaultdict(int)) for p in POLICIES}
    agree = {p: [0, 0] for p in POLICIES}
    regraded = {p: {} for p in POLICIES}
    for rid, t in latest.items():
        q = qs[t["question_id"]]
        jc = (t.get("eval") or {}).get("judge_correct")
        judge = None if jc is None else {"correct": jc}
        for p in POLICIES:
            g = grade_answer(q, t.get("answer", ""), t.get("cited_ids", []), t.get("status", "ok"),
                             t.get("gate_emitted", 0), t.get("gate_rejected", 0), judge, policy=p).grade
            regraded[p][rid] = g
            if g:
                out[p][t["setting_label"]][g] += 1
                out[p][t["setting_label"]]["_n"] += 1
            if rid in human:
                agree[p][1] += 1
                agree[p][0] += int(g == human[rid])

    summary = {"human_graded_rows": len(human), "policies": {}}
    for p in POLICIES:
        per = {}
        for s, c in out[p].items():
            n = c["_n"]
            per[s] = {lab: round(100 * c[lab] / n, 1) if n else None for lab in LABELS} | {"graded": n}
        acc = round(agree[p][0] / agree[p][1], 3) if agree[p][1] else None
        summary["policies"][p] = {"agreement_with_human": acc, "n_compared": agree[p][1], "per_setting": per}
        print(f"\n{p}: agreement with your grades = {acc} (n={agree[p][1]})")
        for s, v in per.items():
            print(f"  {s:50s} correct {v['correct']}%  wrong {v['wrong']}%  refused_ok {v['refused_correctly']}%  "
                  f"refused_wrong {v['refused_wrongly']}%  (n={v['graded']})")
    summary_path = Path(a.traces).parent / "regrade_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nwrote {summary_path}")
    if not human:
        print("\nNo human grades found yet: fill human_grade in data/live/eval_review.csv, then rerun.")

    if a.apply:
        res = Path(a.results)
        shutil.copy(res, res.with_suffix(".csv.bak"))
        with open(res, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields, data = reader.fieldnames, list(reader)
        n = 0
        for r in data:
            if r["request_id"] in regraded[a.apply]:
                r["answer_grade"] = regraded[a.apply][r["request_id"]]
                n += 1
        with open(res, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
        print(f"\nRewrote answer_grade for {n} rows in {res} using '{a.apply}' (backup: {res.name}.bak)")


if __name__ == "__main__":
    main()
