"""Replay archived coder responses through the integrated checker without model calls.

Usage: python replay_validation_review.py --artifacts PATH --output RESULT.json
Optional: --public-data-root PATH to inspect prepared/public task metadata.
Use --continuations PATH for saved model reviews that the historical ID shortcut
skipped. The replay never substitutes an automatic ID verdict for missing calls.
Use --allow-decision-changes to measure deliberate component removals against
the archived decisions while retaining response/provenance checks.
"""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--public-data-root", type=Path)
parser.add_argument("--continuations", type=Path)
parser.add_argument("--allow-decision-changes", action="store_true")
args = parser.parse_args()
P = args.artifacts.resolve()
SOURCE = Path(__file__).resolve().parents[1] / "agents/aide/src"
sys.path.insert(0, str(SOURCE))
from aide import data_leakage as checker
from aide import validation_evidence as new
from aide.utils.validation_schema import PublicValidationSchema

assert Path(checker.__file__).is_relative_to(SOURCE)
sys.path.insert(0, str(P))
from source_projection import project as old_project
from source_evidence_v2 import numeric_selection_facts as old_numeric

saved = json.loads((P / "iterative_results.json").read_text())
schemas = json.loads((P / "final_comparison_inputs/public_schemas.json").read_text())
case_by_id = {c["case"]: c for c in saved["cases"]}
equivalence = []
# Dataset identity comes from manifests, not inferred labels.
original = []
for folder in ["replay_inputs", "final_comparison_inputs"]:
    original.extend(json.loads((P / folder / "cases.json").read_text()))
for c in saved["cases"]:
    code = (P / c["source"]).read_text()
    assert hashlib.sha256(code.encode()).hexdigest() == c["source_sha256"]
    item = next(a for a in original if a["id"] == c["case"])
    schema = schemas.get(item.get("task"), {})
    equivalence.append(
        {
            "case": c["case"],
            "projection_equal": new.project(code) == old_project(code),
            "numeric_facts_equal": new.numeric_selection_facts(code, schema)
            == old_numeric(code, schema),
        }
    )

rows = []
for row in saved["rows"]:
    if row["variant"] != "candidate":
        continue
    c = case_by_id[row["case"]]
    source = (P / c["source"]).read_text()
    original_case = next(a for a in original if a["id"] == c["case"])
    recorded = json.loads((P / row["artifact"]).read_text())
    phases = copy.deepcopy(recorded["phases"])
    reference_decision = row["decision"]
    continuation_path = None
    phase_errors = recorded["errors"]
    if phases.get("id_join_proofs"):
        if args.continuations is None:
            parser.error(
                "Historical ID rejection skipped model review; supply --continuations "
                "with saved export/confirmation responses for those evaluations."
            )
        continuation_path = args.continuations / f'{row["case"]}__r{row["repeat"]}.json'
        continuation = json.loads(continuation_path.read_text())
        assert continuation["case"] == row["case"]
        assert continuation["repeat"] == row["repeat"]
        assert continuation["source_sha256"] == c["source_sha256"]
        assert continuation["cached_upstream_sha256"] == hashlib.sha256(
            (P / row["artifact"]).read_bytes()
        ).hexdigest()
        phases.update(continuation["phases"])
        phase_errors = continuation["errors"]
        reference_decision = continuation["decision"]
    unexpected = []
    requested_phases = []

    def query(**kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        requested_phases.append(name)
        if name == "validation_inputs":
            record = phases.get("inputs")
        elif name.startswith("validation_input_verify_"):
            i = int(name.rsplit("_", 1)[1])
            records = phases.get("inputs", {}).get("verifications", [])
            record = records[i] if i < len(records) else None
        else:
            key = {
                "submission_export": "export",
                "submission_export_confirmation": "confirmation",
            }[name]
            record = phases.get(key)
        if record is None:
            # Some failed offline requests saved only an error rather than a response.
            if (
                not any(e["phase"] == "export" for e in phase_errors)
                or name != "submission_export"
            ):
                unexpected.append(name)
            raise RuntimeError("No successful archived response for " + name)
        response = record["response"]
        choice = response["choices"][0]
        kwargs["info_sink"].update(response.get("usage", {}))
        kwargs["info_sink"]["finish_reason"] = choice.get("finish_reason")
        return choice["message"]["content"]

    with (
        patch.object(checker, "check_data_leakage", return_value=copy.deepcopy(phases["leakage"])),
        patch.object(checker, "query", side_effect=query),
    ):
        result = checker.check_validation(
            task_desc=original_case["task_desc"],
            code=source,
            model="gpt-coderbf16-fb-vllm",
            eval_metric=original_case["metric"],
            public_schema=schemas.get(original_case.get("task"), {}),
            timeout_s=120,
        )
    rows.append(
        {
            "case": row["case"],
            "repeat": row["repeat"],
            "expected": row["expected"],
            "offline_decision": row["decision"],
            "integrated_decision": result["decision"],
            "reference_decision": reference_decision,
            "continuation_artifact": str(continuation_path) if continuation_path else None,
            "same_rejection": (reference_decision == "reject") == (result["decision"] == "reject"),
            "changed_from_original": (row["decision"] == "reject") != (result["decision"] == "reject"),
            "unexpected_requests": unexpected,
            "requested_phases": requested_phases,
            "errors": result["errors"],
        }
    )

public = []
for task in schemas if args.public_data_root else []:
    path = args.public_data_root / task / "prepared/public"
    start = time.monotonic()
    current = PublicValidationSchema(path).load(start + 120)
    public.append(
        {
            "task": task,
            "files": current["files"],
            "column_dtypes": current["column_dtypes"],
            "absent_fields": sorted(new.absent_fields(current)),
            "expected_absent_fields": sorted(new.absent_fields(schemas[task])),
            "designated_raw_target_fields": current["designated_raw_target_fields"],
            "diagnostics": current["diagnostics"],
            "elapsed_s": time.monotonic() - start,
        }
    )
result = {
    "mode": "saved-response adapter replay; saved continuations replace formerly skipped ID model phases; no new judge calls or candidate execution",
    "policy_version": checker.LEAKAGE_POLICY_VERSION,
    "source_directory": str(SOURCE),
    "mechanical_equivalence": equivalence,
    "replays": rows,
    "public_schema_smoke_checks": public,
    "all_mechanical_checks_equal": all(
        all(v for k, v in r.items() if k != "case") for r in equivalence
    ),
    "rejections_preserved": sum(r["same_rejection"] for r in rows),
    "changed_from_original": [
        {k: r[k] for k in ("case", "repeat", "offline_decision", "integrated_decision")}
        for r in rows if r["changed_from_original"]
    ],
    "continuation_replays": sum(r["continuation_artifact"] is not None for r in rows),
    "true_positives": sum(r["expected"] == "reject" and r["integrated_decision"] == "reject" for r in rows),
    "false_positives": sum(r["expected"] == "allow" and r["integrated_decision"] == "reject" for r in rows),
    "total": len(rows),
}
result["component_hashes"] = {
    name: hashlib.sha256((SOURCE / "aide" / name).read_bytes()).hexdigest()
    for name in [
        "data_leakage.py",
        "agent.py",
        "validation_review.py",
        "validation_evidence.py",
        "validation_prompts.py",
        "utils/validation_schema.py",
    ]
}
result["replay_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
output = args.output
output.write_text(json.dumps(result, indent=2) + "\n")
print("MECHANICAL EQUIVALENCE", result["all_mechanical_checks_equal"], len(equivalence))
print("REJECTIONS PRESERVED", result["rejections_preserved"], "/", len(rows))
print("CHANGED FROM ORIGINAL", result["changed_from_original"])
for r in rows:
    if not r["same_rejection"] or r["unexpected_requests"]:
        print(json.dumps(r))
for r in public:
    print(
        "PUBLIC SCHEMA",
        r["task"],
        len(r["files"]),
        r["absent_fields"] == r["expected_absent_fields"],
        r["elapsed_s"],
        r["diagnostics"],
    )
print(output)

assert result["all_mechanical_checks_equal"]
assert result["total"] == 129
if not args.allow_decision_changes:
    assert result["rejections_preserved"] == result["total"]
assert not any(r["unexpected_requests"] for r in rows)
assert all(r["absent_fields"] == r["expected_absent_fields"] for r in public)
