"""Real-oracle BoN behavioural test.

Under `bon_oracle=real` every candidate actually executes in its own
workspace; the winner's finished ExecutionResult is REUSED (returned to the
step, which must then skip exec_callback) and its submission.csv is published
to the canonical workspace path. Runs real subprocesses via Interpreter.
"""

import json
import time
import types

from aide import agent as agent_mod
from aide.utils.submission_validation import SubmissionValidation, submission_sha256


class _Stub:
    """Minimal Agent stand-in: only what the real-oracle parallel path touches."""

    _parallel = agent_mod.Agent._parallel
    _draw_candidates = agent_mod.Agent._draw_candidates
    _persist_draw_axes = agent_mod.Agent._persist_draw_axes
    _observe = agent_mod.Agent._observe
    _review_candidates = agent_mod.Agent._review_candidates
    _observe_real = agent_mod.Agent._observe_real
    _process_real_submission = agent_mod.Agent._process_real_submission
    _validate_submission_checks = agent_mod.Agent._validate_submission_checks
    _select = agent_mod.Agent._select
    _commit = agent_mod.Agent._commit
    _new_step_log_dir = agent_mod.Agent._new_step_log_dir
    _write_bon_candidate_artifacts = agent_mod.Agent._write_bon_candidate_artifacts

    def __init__(self, tmp_path, draws):
        (tmp_path / "ws").mkdir()
        self.cfg = types.SimpleNamespace(
            log_dir=tmp_path / "logs",
            workspace_dir=tmp_path / "ws",
            exec=types.SimpleNamespace(
                timeout=60, format_tb_ipython=False, agent_file_name="runfile.py"
            ),
        )
        self.acfg = types.SimpleNamespace(
            best_of_n_size=2,
            use_best_of_n_listwise=False,
            # No `_review` on this stub; the guard defaults on without it.
            bon_candidate_review=False,
            use_line_frac_salvage=False,
        )
        self.current_step = 0
        self.task_desc = "t"
        self.oracle = "real"
        self.deadline = time.time() + 600
        self._draws = list(draws)

    def _generate_n_candidates(self, prompt, n, max_tokens=None):
        return [self._draws.pop(0) for _ in range(n)]

    def _validate_submission(self, csv_path, **kwargs):
        return SubmissionValidation("valid", "Submission is valid.", submission_sha256(csv_path))


GOOD = (
    "with open('submission/submission.csv', 'w') as f:\n"
    "    f.write('id,pred\\n1,0\\n')\n"
    "print('ok')"
)
BAD = "raise ValueError('boom')"


def test_real_oracle_executes_all_and_reuses_the_winners_result(tmp_path):
    stub = _Stub(tmp_path, draws=[("pb", BAD), ("pg", GOOD)])

    plan, code, exec_result = stub._parallel("prompt", "draft", None, None)

    assert code == GOOD
    # The winner already ran: its finished result comes back so the step can
    # skip exec_callback instead of executing twice.
    assert exec_result is not None and exec_result.exc_type is None
    assert "ok" in "".join(exec_result.term_out)
    # Winner CSV published to the canonical workspace path.
    canonical = stub.cfg.workspace_dir / "submission" / "submission.csv"
    assert canonical.read_text().startswith("id,pred")
    # Per-candidate artifacts: the loser keeps its real failure on record.
    step = stub.cfg.log_dir / "bon" / "step_0"
    r0 = json.loads((step / "cand_0" / "result.json").read_text())
    assert r0["accepted"] is False
    assert r0["reject_reason"] == "exception:ValueError"
    r1 = json.loads((step / "cand_1" / "result.json").read_text())
    assert r1["accepted"] is True and r1["reject_reason"] is None
    assert (step / "picked.txt").read_text() == "1"
    # Candidate workspaces are torn down once the winner is published.
    assert not (stub.cfg.workspace_dir / "bon").exists()


def test_width_four_real_metric_oracle_reuses_best_validation_submission(tmp_path):
    """Exercise four isolated executions, actual output reviews and winner reuse together."""
    import re

    values = [0.1, 0.6, 0.3, 0.9]
    draws = [("plan", GOOD.replace("1,0", f"1,{value}").replace("print('ok')", f"print('validation_metric={value}')"))
             for value in values]
    stub = _Stub(tmp_path, draws)
    stub.acfg.best_of_n_size = 4
    stub.acfg.bon_candidate_review = True
    stub.acfg.use_observed_metric_argmax_pick = True
    stub.acfg.eval_lower_is_better = False
    stub._select_observed_metric = types.MethodType(agent_mod.Agent._select_observed_metric, stub)

    def review(code, terminal_output, **kwargs):
        assert kwargs['predicted'] is False
        metric = float(re.search(r'validation_metric=([0-9.]+)', terminal_output).group(1))
        return {'metric': metric, 'metric_trustworthy': True, 'is_bug': False}

    stub._review = review
    _, code, result = stub._parallel("prompt", "draft", None, None)
    assert code == draws[3][1]
    assert result is not None and result.exc_type is None
    assert 'validation_metric=0.9' in ''.join(result.term_out)
    step = stub.cfg.log_dir / 'bon/step_0'
    assert len(list(step.glob('cand_*/result.json'))) == 4
    assert json.loads((step / 'listwise.json').read_text())['winner'] == 3
    assert '1,0.9' in (stub.cfg.workspace_dir / 'submission/submission.csv').read_text()
    assert not (stub.cfg.workspace_dir / 'bon').exists()
