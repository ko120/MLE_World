"""Evidence-guided review using the configured AIDE backend and one deadline.

The prompts and mechanical recognizers come from the evaluated offline candidate.
Expected labels, scores, experiment paths, and model endpoints are not inputs.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from threading import Lock
import time

import jsonschema

from .backend import determine_provider
from . import validation_evidence as evidence
from . import validation_prompts as prompts


class EvidenceReview:
    def __init__(
        self,
        *,
        code,
        public_schema,
        model,
        query_fn,
        policy_version,
        timeout_s=120.0,
        max_tokens=3072,
        temperature=0.0,
        convert_system_to_user=False,
        reject_on_detection=True,
        eval_metric=None,
        task_desc=None,
        validated_submission=None,
        **legacy_options,
    ):
        self.code = code
        self.schema = public_schema
        self.model = model
        self.query = query_fn
        self.deadline = time.monotonic() + max(0.0, timeout_s)
        self.started = time.monotonic()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.convert_system_to_user = convert_system_to_user
        self.reject_on_detection = reject_on_detection
        self.eval_metric = eval_metric
        self.task_desc = (task_desc or "")[:12000]
        receipt = validated_submission or {}
        digest = receipt.get("sha256")
        # The caller supplies a receipt for this candidate's validated bytes,
        # never a simulated prediction of submission validity.
        self.validated_submission = (
            {"status": "valid", "sha256": digest,
             "actual_columns": list(receipt.get("columns") or [])}
            if receipt.get("status") == "valid" and isinstance(digest, str)
            and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
            else None
        )
        self.lock = Lock()
        self.uncertain = []
        self.result = {
            "policy_version": policy_version,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "model": model,
            "status": "checked",
            "decision": "allow",
            "has_leakage": False,
            "has_submission_mismatch": False,
            "reject_on_detection": reject_on_detection,
            "attempts": 0,
            "usage": [],
            "errors": [],
            "evidence": [],
            "phases": {},
            "schema_diagnostics": public_schema.get("diagnostics", []),
            "validated_submission": self.validated_submission,
        }

    def _error(self, phase, exc):
        with self.lock:
            self.result["errors"].append({"phase": phase, "reason": str(exc)[:500]})

    def _request(self, phase, policy, user, schema, token_cap=3072):
        """All calls, including parallel calls, consume the same wall-clock budget."""
        info = {}
        attempted = False
        try:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0 or self.max_tokens <= 0:
                raise TimeoutError("Validation review allowance exhausted")
            with self.lock:
                self.result["attempts"] += 1
            attempted = True
            response = self.query(
                system_message=policy,
                user_message=json.dumps(user) if isinstance(user, dict) else user,
                func_spec=None,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": phase,
                        "schema": schema,
                        "strict": True,
                    },
                },
                model=self.model,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, token_cap),
                convert_system_to_user=self.convert_system_to_user,
                info_sink=info,
                retry_max_time=remaining,
                retry_attempt_timeout=remaining,
                tool_call_max_attempts=1,
                **(
                    {"temperature_override": self.temperature}
                    if determine_provider(self.model) == "openai"
                    else {}
                ),
            )
            if info.get("finish_reason") not in (None, "stop"):
                raise ValueError("Incomplete response: " + str(info["finish_reason"]))
            parsed = json.loads(response) if isinstance(response, str) else response
            jsonschema.validate(parsed, schema)
            self._validate_lines(parsed)
            with self.lock:
                self.result["phases"][phase] = parsed
            return parsed
        except Exception as exc:
            self._error(phase, exc)
            return None
        finally:
            if attempted:
                with self.lock:
                    self.result["usage"].append({"phase": phase, **info})

    def _validate_lines(self, obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"source_lines", "lines", "raw_read_lines", "selected_feature_lines", "prediction_call_lines"}:
                    if any(
                        type(n) is not int or not 1 <= n <= len(self.code.splitlines())
                        for n in value
                    ):
                        raise ValueError("Citation outside the original source")
                else:
                    self._validate_lines(value)
        elif isinstance(obj, list):
            for value in obj:
                self._validate_lines(value)

    def _evidence_request(self, phase, question, schema, facts):
        user = (
            question
            + "\n\nPublic/mechanical evidence (not a verdict):\n"
            + json.dumps(facts, indent=2)
            + "\n\nHelper call sites:\n"
            + json.dumps(evidence.helpers(self.code), indent=2)
            + "\n\nPython source with original line numbers:\n"
            + evidence.project(self.code, True)
        )
        return self._request(phase, prompts.EVIDENCE_POLICY, user, schema)

    def _inputs(self):
        missing = evidence.absent_fields(self.schema)
        if not missing:
            return []
        user = {
            "metric_name": self.eval_metric,
            "public_input_schema": evidence.compact_schema(self.schema),
            "question": prompts.INPUT_FOCUS,
            "source": evidence.project(self.code, True),
            "output_contract": (
                "Return only the requested fact JSON. There is no verdict field. "
                "All booleans describe reached executable data flow. Schema: "
                + json.dumps(prompts.INPUT_SCHEMA)
            ),
            "mechanical_raw_schema_branch_facts": evidence.schema_branch_facts(
                self.code, self.schema
            ),
            "public_column_dtypes": evidence.input_column_dtypes(self.schema),
            "mechanical_column_filters": evidence.column_filter_facts(self.code, self.schema),
        }
        facts = self._request("validation_inputs", prompts.INPUT_POLICY, user, prompts.INPUT_SCHEMA)
        # Even if extraction fails, supported numeric selectors can supply a
        # proposal; only a subsequent successful verification can reject it.
        candidates = [
            f
            for f in (facts or {}).get("field_paths", [])
            if f["field"] in missing and f["validation_nondefault_value_reaches_prediction"]
        ]
        numeric = evidence.numeric_selection_facts(self.code, self.schema)
        if not candidates:
            for fact in numeric:
                columns = next(
                    (
                        v
                        for k, v in fact["raw_public_columns_satisfying_predicate"].items()
                        if "train" in k
                    ),
                    [],
                )
                for field in sorted(missing.intersection(columns))[:2]:
                    candidates.append(
                        {
                            "field": field,
                            "source_lines": [fact["line"]],
                            "selector_evidence": fact,
                            "validation_path": "This raw field satisfies the source numeric "
                            "selector on public train dtypes. Trace subsequent consumers and "
                            "filters; selection alone is not a defect verdict.",
                        }
                    )
        self.result["phases"]["input_proposals"] = candidates
        for index, proposal in enumerate(candidates[:3]):
            field = proposal["field"]
            literal_lines = evidence.field_literal_lines(self.code, field)
            selector_lines = {
                f["line"] for f in numeric
                if any(field in cols for cols in f["raw_public_columns_satisfying_predicate"].values())
            }
            if not (literal_lines or selector_lines):
                # An absent raw field with no identifiable read/selector cannot
                # support rejection. Dynamic/unsupported accesses remain open.
                self.result["phases"].setdefault("input_unproved_paths", []).append(
                    {"field": field, "reason": "No exact executable field reference or supported numeric selector."}
                )
                self.uncertain.append("unproved raw field access: " + field)
                continue
            checked = self._evidence_request(
                f"validation_input_verify_{index}",
                prompts.INPUT_VERIFY_QUESTION,
                prompts.VERIFY_SCHEMA,
                {
                    "untrusted_proposed_path": proposal,
                    "raw_field_is_present_in_labeled_data_and_absent_from_all_test_rows": proposal[
                        "field"
                    ],
                    "mechanical_numeric_selection": numeric,
                    "mechanical_raw_schema_branch_facts": evidence.schema_branch_facts(self.code, self.schema),
                    "exact_field_reference_lines": sorted(literal_lines),
                    "mechanical_column_filters": evidence.column_filter_facts(self.code, self.schema),
                },
            )
            if checked is None:
                continue
            if checked["role"] == "unresolved":
                self.uncertain.append("validation input: " + proposal["field"])
            if checked["role"] == "observed_predictor_input":
                if not (
                    checked["raw_field"] == field
                    and set(checked["raw_read_lines"]) & (literal_lines | selector_lines)
                ):
                    self.result["phases"].setdefault("input_unproved_paths", []).append(
                        {"field": field, "reason": "Verifier conclusion lacks a consistent, source-cited input witness.",
                         "verification": checked}
                    )
                    self.uncertain.append("inconsistent validation input proof: " + field)
                    continue
                return [
                    {
                        "mechanism": "submission_input_mismatch",
                        "data_flow": f"Validation consumes observed {proposal['field']!r}: "
                        + checked["field_value_at_prediction_input"],
                        "why_not_permitted": "The raw field is absent from public test "
                        "records, so validation does not measure the test-available predictor.",
                        "source_lines": checked["source_lines"],
                        "repair": "Evaluate validation using the same information available "
                        "at test prediction; exclude this field and its dependent features "
                        "or use the same fallback on both paths.",
                    }
                ]
        return []

    def _export(self):
        tree = ast.parse(self.code)
        operations = {
            "to_csv",
            "writer",
            "writerow",
            "writerows",
            "savetxt",
            "write_text",
            "write_bytes",
        }
        has_export = any(
            isinstance(n, ast.Call)
            and (
                isinstance(n.func, ast.Attribute)
                and n.func.attr in operations
                or isinstance(n.func, ast.Name)
                and n.func.id in operations
            )
            for n in ast.walk(tree)
        )
        has_path = any(
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and "submission" in n.value.lower()
            for n in ast.walk(tree)
        )
        if not (has_export or has_path):
            self.result["phases"]["export_scope"] = "No submission export/path shown."
            return []
        user = {
            "metric_name": self.eval_metric,
            "task_description": self.task_desc,
            "validated_submission": self.validated_submission,
            "public_input_schema": evidence.compact_schema(self.schema),
            "question": prompts.EXPORT_FOCUS,
            "source": evidence.project(self.code, True),
            "output_contract": "Return JSON matching this schema. trace: at most 350 "
            "words of concrete source operations. findings: at most two distinct "
            "actual defects; NEVER put permitted operations here. decision: reject "
            "means a real defect EXISTS, allow means no defect established and "
            "REQUIRES empty findings, uncertain means a necessary source fact is "
            "unresolved. A code fix that matches the existing reachable code is "
            "not a finding. No repeated findings. Schema: " + json.dumps(prompts.EXPORT_SCHEMA),
        }
        result = self._request(
            "submission_export", prompts.EXPORT_POLICY, user, prompts.EXPORT_SCHEMA
        )
        if result is None:
            return []
        if result["decision"] == "uncertain":
            self.uncertain.append("submission export")
        if (
            result["decision"] == "allow"
            and result["findings"]
            or result["decision"] == "reject"
            and not result["findings"]
        ):
            self._error("submission_export", "Verdict contradicts the findings list")
            return []
        export_mechanisms = {
            "export_value_mismatch",
            "inference_transform_or_state_mismatch",
        }
        # Selecting another record's feature row is a correspondence defect,
        # even when the proposer calls it an "input" mismatch. Keep raw-field
        # availability findings on their separate evidence path, but do not
        # discard an explicit row/class claim before independent confirmation.
        proposed = []
        for finding in result["findings"]:
            if (finding["mechanism"] == "input_information_mismatch"
                    and finding["basis"] in {"row_alignment", "class_alignment"}):
                self.result["phases"].setdefault("export_alignment_claims_routed", []).append(finding)
                finding = {**finding, "mechanism": "export_value_mismatch"}
            proposed.append(finding)
        value_bases = {"row_alignment", "class_alignment", "prediction_values", "inference_pipeline"}
        dropped = [f for f in proposed if f["basis"] in {
            "structural_format", "training_or_quality", "metric_only",
        }]
        if dropped:
            self.result["phases"]["export_nondefect_claims_dropped"] = dropped
        if any(f["basis"] == "unresolved" for f in proposed):
            self.uncertain.append("export basis unresolved")
        findings = [f for f in proposed
                    if f["mechanism"] in export_mechanisms and f["basis"] in value_bases]
        outside_scope = [f for f in proposed
                         if f["mechanism"] not in export_mechanisms and f not in dropped]
        if outside_scope:
            # Leakage and missing-input reviews have separate evidence paths.
            # A positive report outside this branch must not become a clean allow.
            self.result["phases"]["export_out_of_scope"] = outside_scope
            self.uncertain.append("export findings require leakage or input verification")
        if not findings:
            return []
        user = {
            "task_description": self.task_desc,
            "validated_submission": self.validated_submission,
            "actual_helper_call_sites": [
                {"line": c["line"], "expression": c["call"]} for c in evidence.helpers(self.code)
            ],
            "public_schema": evidence.compact_schema(self.schema),
            "source": evidence.project(self.code, True),
            "mechanical_schema_facts": evidence.schema_branch_facts(self.code, self.schema),
            "untrusted_claims": [
                {
                    "index": i,
                    "mechanism": f["mechanism"],
                    "basis": f["basis"],
                    "claim": f["actual_operations"],
                    "lines": f["lines"],
                }
                for i, f in enumerate(findings)
            ],
            "output_contract": "Address every claim exactly once, including refuted "
            "claims. Keep each analysis below 120 words. Schema: "
            + json.dumps(prompts.CONFIRM_SCHEMA),
        }
        confirmed = self._request(
            "submission_export_confirmation",
            prompts.CONFIRM_POLICY,
            user,
            prompts.CONFIRM_SCHEMA,
            token_cap=4096,
        )
        if confirmed is None:
            return []
        claims = confirmed["claims"]
        if len(claims) != len(findings) or {c["claim_index"] for c in claims} != set(
            range(len(findings))
        ):
            self._error("submission_export_confirmation", "Every claim must be addressed once")
            return []
        accepted = []
        for claim in claims:
            if claim["conclusion"] == "insufficient_evidence":
                self.uncertain.append("export confirmation")
            if claim["conclusion"] != "defect_proved":
                continue
            if claim["basis"] not in value_bases:
                self.result["phases"].setdefault("export_confirmation_claims_dropped", []).append(claim)
                if claim["basis"] == "unresolved":
                    self.uncertain.append("export confirmation basis unresolved")
                continue
            finding = findings[claim["claim_index"]]
            accepted.append(
                {
                    "mechanism": finding["mechanism"],
                    "basis": claim["basis"],
                    "data_flow": claim["actual_reached_operations"],
                    "why_not_permitted": finding["why_incorrect"],
                    "source_lines": claim["source_lines"] or finding["lines"],
                    "repair": finding["minimal_repair"],
                }
            )
        return accepted

    def _finish(self, findings, *, submission):
        flag = "has_submission_mismatch" if submission else "has_leakage"
        positive = "mismatch_detected" if submission else "leak_detected"
        negative = "no_mismatch_detected" if submission else "no_leak_detected"
        incomplete = bool(self.result["errors"] or self.uncertain)
        verdict = positive if findings else "uncertain" if incomplete else negative
        decision = (
            ("reject" if self.reject_on_detection else "advisory")
            if findings
            else ("advisory" if incomplete else "allow")
        )
        lines = self.code.splitlines()
        reason = "\n".join(f"{f['data_flow']} Repair: {f['repair']}" for f in findings)
        reason = reason or (
            "Review incomplete; inspect phase errors and uncertainty."
            if incomplete
            else "No source-supported defect established."
        )
        self.result.update(
            {
                flag: bool(findings),
                "decision": decision,
                "status": "unavailable" if self.result["errors"] and not findings else "checked",
                "reason": reason,
                "judge": {"verdict": verdict, "findings": findings, "reason": reason},
                "evidence": [
                    {
                        **f,
                        "evidence": [{"line": n, "code": lines[n - 1]} for n in f["source_lines"]],
                    }
                    for f in findings
                ],
                "uncertain_phases": self.uncertain,
                "elapsed_s": time.monotonic() - self.started,
            }
        )
        return self.result

    def submission(self):
        findings = []
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = {
                    "inputs": pool.submit(self._inputs),
                    "export": pool.submit(self._export),
                }
                for phase, future in jobs.items():
                    try:
                        findings.extend(future.result())
                    except Exception as exc:
                        self._error(phase, exc)
        except Exception as exc:
            self._error("submission_review", exc)
        return self._finish(findings, submission=True)
