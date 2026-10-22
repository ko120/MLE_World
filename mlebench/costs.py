#!/usr/bin/env python3
"""Offline attempt/cost diagnostics, without submission snapshots or model calls.

Usage: python real_execution_report.py RUN_GROUP... --out-dir OUTPUT
Writes trials.csv, attempts.csv, breakdowns.csv, publications.csv, model_usage.csv,
and paired_modes.csv. Existing journals are a labelled, incomplete duration proxy.
Published test_metric values are raw task scores, never averaged across tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"(openai|sim) call: in_tokens=(\d+) out_tokens=(\d+) req_time=([0-9.]+)s(?: model=(.*))?$")
SCORING_RE = re.compile(r"exc_posterior: .*seconds=([0-9.]+)")


def quantile(values, probability):
    if not values:
        return None
    values = sorted(values)
    position = probability * (len(values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def trigger_label(flags):
    return "+".join(sorted(key for key, enabled in (flags or {}).items() if enabled)) or "unknown"


def read_events(path):
    events, bad_lines = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("event must be an object")
            events.append(value)
        except (ValueError, TypeError):
            bad_lines += 1
    return events, bad_lines


def summarize(log_dir):
    """Keep incomplete attempts visible and reconcile per-session cumulative cost."""
    path = log_dir / "real_execution_events.jsonl"
    events, bad_lines = read_events(path)
    headers = [e for e in events if e.get("event") == "run_start"]
    header = headers[0] if headers else {}
    trial = str(log_dir.parent)
    common = {k: header.get(k) for k in ("task", "seed", "method", "execution_mode")}
    common["trial"] = trial
    starts, ends, cleanup = {}, {}, defaultdict(float)
    cost_by_session = defaultdict(float)
    latest_cumulative = {}
    committed, selected, real_nodes, reviewed = set(), set(), set(), {}
    publications, skips, steps, review_s = [], defaultdict(int), [], 0.0
    duplicate_events = 0
    for event in events:
        kind, session = event.get("event"), event.get("session_id")
        key = (session, event.get("attempt_id"))
        if kind in {"attempt_start", "attempt_end"}:
            target = starts if kind == "attempt_start" else ends
            if key in target:
                duplicate_events += 1
            else:
                target[key] = event
                if kind == "attempt_end":
                    cost_by_session[session] += event.get("elapsed_s", 0.0)
        elif kind == "attempt_cleanup":
            cleanup[key] += event.get("elapsed_s", 0.0)
            cost_by_session[session] += event.get("elapsed_s", 0.0)
        elif kind == "node_selected":
            selected.add(event.get("node_id"))
        elif kind in {"node_committed", "validation_result"}:
            if event.get("real_won"):
                real_nodes.add(event.get("node_id"))
            if event.get("attempt_id"):
                reviewed[(session, event["attempt_id"])] = event
            if kind == "node_committed":
                committed.add(event.get("node_id"))
        elif kind == "publication":
            publications.append({**common, **event})
        elif kind == "execution_decision" and event.get("skip_reason"):
            skips[event["skip_reason"]] += 1
        elif kind == "step_end":
            steps.append(event.get("elapsed_s", 0.0))
        elif kind == "review_end":
            review_s += event.get("elapsed_s", 0.0)
        if isinstance(event.get("cumulative_real_s"), (int, float)):
            latest_cumulative[session] = event["cumulative_real_s"]
    attempts = []
    for key in starts.keys() | ends.keys() | cleanup.keys():
        start, end = starts.get(key, {}), ends.get(key, {})
        fields = {**start, **end}
        measured = end.get("elapsed_s")
        duration = measured + cleanup[key] if measured is not None else None
        node_result = reviewed.get(key, {})
        attempts.append({
            **common, "session_id": key[0], "attempt_id": key[1],
            "node_id": fields.get("node_id"), "code_sha256": fields.get("code_sha256"),
            "stage": fields.get("stage"), "policy": fields.get("policy"),
            "path": fields.get("path"), "trigger": trigger_label(fields.get("triggers")),
            "retry_index": fields.get("retry_index", 0), "start_timestamp": start.get("timestamp"),
            "end_timestamp": end.get("timestamp"), "real_s": duration,
            "measured_run_s": measured, "cleanup_s": cleanup[key],
            "reported_exec_s": end.get("reported_exec_s"),
            "status": end.get("status", "incomplete"), "exception_type": end.get("exception_type"),
            "missing_start": not bool(start), "enforced_timeout_s": start.get("enforced_timeout_s"),
            "configured_timeout_s": fields.get("configured_timeout_s"),
            "confidence": fields.get("confidence"), "predicted_exception": fields.get("predicted_exception"),
            "validation_status": node_result.get("validation_status"),
            "is_buggy": node_result.get("is_buggy"),
        })
    attempts.sort(key=lambda row: (row["start_timestamp"] or 0, row["attempt_id"] or ""))
    known_cost = sum(cost_by_session.values())
    durations = [a["real_s"] for a in attempts if a["real_s"] is not None]
    checked = {a["node_id"] for a in attempts if a["node_id"]} | real_nodes
    complete_sessions = {e.get("session_id") for e in events if e.get("event") == "run_end"}
    all_sessions = {e.get("session_id") for e in events}
    ended = bool(all_sessions) and all_sessions <= complete_sessions
    incomplete = sum(a["status"] == "incomplete" for a in attempts)
    missing_starts = sum(a["missing_start"] for a in attempts)
    reconciles = all(abs(cost_by_session[s] - value) < 1e-6 * max(1.0, value)
                     for s, value in latest_cumulative.items())
    total_wall = sum(max((e.get("trial_wall_s", 0.0) for e in events if e.get("session_id") == s), default=0.0)
                     for s in all_sessions)
    final = publications[-1] if publications else {}
    summary = {
        **common, "cost_source": "attempt_ledger", "known_real_s": known_cost,
        "generator_model": header.get("generator_model"), "feedback_model": header.get("feedback_model"),
        "wm_model": header.get("wm_model"),
        "known_real_hours": known_cost / 3600, "cost_complete": ended and not (incomplete or bad_lines or missing_starts or duplicate_events) and reconciles,
        "incomplete_attempts": incomplete, "missing_starts": missing_starts,
        "malformed_lines": bad_lines, "duplicate_events": duplicate_events, "cost_reconciles": reconciles,
        "real_attempts": len(attempts), "retry_attempts": sum(a["retry_index"] > 0 for a in attempts),
        "selected_nodes": len(selected), "committed_nodes": len(committed),
        "checked_committed_nodes": len(checked & committed),
        "real_check_coverage": len(checked & committed) / len(committed) if committed else None,
        "mean_attempt_s": statistics.mean(durations) if durations else None,
        "p50_attempt_s": quantile(durations, .5), "p90_attempt_s": quantile(durations, .9),
        "trial_wall_s": total_wall, "steps_per_hour": len(committed) * 3600 / total_wall if total_wall else None,
        "step_attempts": len(steps), "mean_step_s": statistics.mean(steps) if steps else None,
        "review_s": review_s, "publication_count": len(publications),
        "published_valid_submission": bool(final), "final_test_metric": final.get("test_metric"),
        "final_submission_sha256": final.get("submission_sha256"),
        "recorder_write_s": sum(e.get("recorder_write_s", 0.0) for e in events if e.get("event") == "run_end"),
        "skip_counts": dict(skips),
    }
    for name, predicate in {
        "timeout": lambda a: a["status"] == "timeout",
        "failed": lambda a: a["status"] in {"timeout", "execution_error", "interpreter_error"},
        "retry": lambda a: a["retry_index"] > 0,
    }.items():
        value = sum((a["real_s"] or 0.0) for a in attempts if predicate(a))
        summary[name + "_real_s"] = value
        summary[name + "_cost_share"] = value / known_cost if known_cost else None
    return summary, attempts, publications


def legacy_summary(log_dir):
    data = json.loads((log_dir / "journal.json").read_text())
    nodes = data.get("nodes", []) if isinstance(data, dict) else data
    real = [n for n in nodes if n.get("real_won") is True]
    cost = sum(n.get("exec_time") or 0.0 for n in real)
    return {
        "trial": str(log_dir.parent), "cost_source": "legacy_journal_proxy", "cost_complete": False,
        "known_real_s": cost, "known_real_hours": cost / 3600,
        "committed_nodes": len(nodes), "checked_committed_nodes": len(real),
        "real_check_coverage": len(real) / len(nodes) if nodes else None,
        "real_attempts": None, "retry_attempts": None,
    }


def model_usage(log_dir, summary):
    path = log_dir / "aide.log"
    usage = defaultdict(lambda: {"logged_operations": 0, "input_tokens": 0, "output_tokens": 0, "operation_s": 0.0})
    scoring_count, scoring_s = 0, 0.0
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            match = TOKEN_RE.search(line)
            if match:
                kind, tokens_in, tokens_out, seconds, model = match.groups()
                key = (kind, (model or "untagged").strip())
                row = usage[key]
                row["logged_operations"] += 1
                row["input_tokens"] += int(tokens_in)
                row["output_tokens"] += int(tokens_out)
                row["operation_s"] += float(seconds)
            scoring = SCORING_RE.search(line)
            if scoring:
                scoring_count += 1
                scoring_s += float(scoring[1])
    result = []
    for (kind, model), row in sorted(usage.items()):
        roles = [role for role in ("generator", "feedback") if summary.get(role + "_model") == model]
        role = "wm" if kind == "sim" else "+".join(roles) or "unknown"
        result.append({"trial": summary["trial"], "kind": kind, "model": model, "role": role, **row,
                       "coverage": "logged operations; underlying requests/retries and usage may be incomplete",
                       "operation_time_is_trial_wall_time": False})
    if scoring_count:
        result.append({"trial": summary["trial"], "kind": "exc_posterior", "model": summary.get("wm_model"),
                       "role": "wm", "logged_operations": scoring_count, "operation_s": scoring_s,
                       "input_tokens": None, "output_tokens": None,
                       "coverage": "scoring operation durations only; tokens and underlying request count unavailable",
                       "operation_time_is_trial_wall_time": False})
    return result


def paired_modes(trials):
    """Pair by task/seed, average seeds within task, bootstrap whole tasks."""
    cells = defaultdict(list)
    for row in trials:
        if row.get("cost_complete") and all(row.get(k) is not None for k in ("method", "task", "seed")):
            cells[(row["method"], row["task"], str(row["seed"]), row["execution_mode"])].append(row)
    pairs = defaultdict(lambda: defaultdict(list))
    for (method, task, seed, mode), rows in cells.items():
        if mode != "confidence" or len(rows) != 1:
            continue
        control = cells.get((method, task, seed, "best_only"), [])
        if len(control) == 1:
            pairs[method][task].append((rows[0]["known_real_s"], control[0]["known_real_s"]))
    result = []
    for method, tasks in sorted(pairs.items()):
        means = [(statistics.mean(a for a, _ in values), statistics.mean(b for _, b in values))
                 for values in tasks.values()]
        delta = statistics.mean(a - b for a, b in means)
        mean_control = statistics.mean(b for _, b in means)
        boot = []
        if len(means) > 1:
            rng = random.Random(0)
            boot = [statistics.mean(a - b for a, b in rng.choices(means, k=len(means))) for _ in range(2000)]
        result.append({"method": method, "contrast": "confidence-minus-best_only", "paired_tasks": len(tasks),
                       "paired_trials": sum(map(len, tasks.values())), "delta_real_s": delta,
                       "ratio_of_task_mean_costs": statistics.mean(a for a, _ in means) / mean_control if mean_control else None,
                       "ci95_low_s": quantile(boot, .025), "ci95_high_s": quantile(boot, .975),
                       "ci99_low_s": quantile(boot, .005), "ci99_high_s": quantile(boot, .995)})
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["trial"]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    log_dirs = set()
    for root in args.inputs:
        if root.is_file():
            log_dirs.add(root.parent.resolve())
        elif root.is_dir():
            for filename in ("real_execution_events.jsonl", "journal.json"):
                log_dirs.update(path.parent.resolve() for path in root.rglob(filename) if "agent_snapshot" not in path.parts)
        else:
            parser.error(f"input does not exist: {root}")
    if not log_dirs:
        parser.error("no execution ledgers or journals found")
    trials, attempts, publications, models = [], [], [], []
    for log_dir in sorted(log_dirs):
        if (log_dir / "real_execution_events.jsonl").exists():
            summary, trial_attempts, trial_publications = summarize(log_dir)
            attempts.extend(trial_attempts)
            publications.extend(trial_publications)
        else:
            summary = legacy_summary(log_dir)
        trials.append(summary)
        models.extend(model_usage(log_dir, summary))
    buckets = defaultdict(lambda: {"attempts": 0, "known_real_s": 0.0, "incomplete_attempts": 0})
    for row in attempts:
        for dimension in ("stage", "trigger", "status", "validation_status", "is_buggy"):
            key = (row["trial"], dimension, row[dimension])
            buckets[key]["attempts"] += 1
            buckets[key]["known_real_s"] += (row["measured_run_s"] or 0.0) + row["cleanup_s"]
            buckets[key]["incomplete_attempts"] += row["real_s"] is None
    breakdowns = [{"trial": trial, "dimension": dimension, "value": value, **values}
                  for (trial, dimension, value), values in sorted(buckets.items(), key=lambda item: str(item[0]))]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in {"trials": trials, "attempts": attempts, "breakdowns": breakdowns,
                       "publications": publications, "model_usage": models, "paired_modes": paired_modes(trials)}.items():
        write_csv(args.out_dir / f"{name}.csv", rows)
    print(f"Wrote {len(trials)} trial summaries and {len(attempts)} attempt records to {args.out_dir}")
    print("Incomplete ledgers and legacy journals are labelled; unknown attempt durations are not zero.")
    print("Publication metrics are raw task scores. No submission files were copied or graded.")


if __name__ == "__main__":
    main()
