"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    t0 = time.monotonic()
    try:
        resp = httpx.post(agent_url, json={"question": question["question"], "db": db_id}, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            "db_id": db_id,
            "question": question["question"],
            "gold_sql": gold_sql,
            "final_sql": "",
            "iterations": 0,
            "ok": False,
            "error": str(e),
            "per_iter_correct": [],
            "correct": False,
            "latency_seconds": time.monotonic() - t0,
        }

    latency = time.monotonic() - t0
    history = data.get("history", [])

    # Evaluate each SQL produced during the run (generate_sql + any revise calls).
    per_iter_correct: list[bool] = []
    for entry in history:
        if entry.get("node") in ("generate_sql", "revise"):
            sql = entry.get("sql", "")
            _, pred_rows, _ = run_sql(db_id, sql)
            per_iter_correct.append(matches(gold_rows, pred_rows))

    # If history is missing (shouldn't happen but be safe), fall back to final SQL.
    if not per_iter_correct:
        final_sql = data.get("sql", "")
        _, pred_rows, _ = run_sql(db_id, final_sql)
        per_iter_correct = [matches(gold_rows, pred_rows)]

    return {
        "db_id": db_id,
        "question": question["question"],
        "gold_sql": gold_sql,
        "final_sql": data.get("sql", ""),
        "iterations": data.get("iterations", 0),
        "ok": data.get("ok", False),
        "per_iter_correct": per_iter_correct,
        "correct": per_iter_correct[-1],
        "latency_seconds": latency,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n": 0, "overall_pass_rate": 0.0, "per_iter_pass_rate": []}

    max_iter = max((len(r.get("per_iter_correct", [])) for r in results), default=1)

    per_iter_pass: list[dict] = []
    for k in range(max_iter):
        correct_at_k = 0
        for r in results:
            iters = r.get("per_iter_correct", [])
            if not iters:
                continue
            # Carry forward the last available result for iterations past termination.
            idx = min(k, len(iters) - 1)
            correct_at_k += int(iters[idx])
        per_iter_pass.append({"iteration": k, "pass_rate": correct_at_k / n})

    overall = sum(1 for r in results if r.get("correct", False)) / n

    return {
        "n": n,
        "overall_pass_rate": overall,
        "per_iter_pass_rate": per_iter_pass,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
