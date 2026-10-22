"""Raw gate boundaries, missing scores, cached scoring, and real-result reuse."""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agents/aide/src'))
from aide.execution_gate import (ExecutionGateConfig, outcome_gate_confidence,
                                 outcome_gate_requires_real, validate_execution_gate)

Q2 = 0.9472301251634779
Q3 = 0.9982787678870029


@pytest.mark.parametrize('success,failure', [(Q3, Q3), (Q2, Q3), (Q3, Q2), (Q2, Q2)])
@pytest.mark.parametrize('exc', [None, 'ValueError'])
def test_exact_quintile_boundary_accepts_equality(success, failure, exc):
    cfg = SimpleNamespace(execution_gate=ExecutionGateConfig(True, success, failure))
    validate_execution_gate(cfg)
    threshold = success if exc is None else failure
    assert outcome_gate_requires_real(cfg, exc, math.nextafter(threshold, 0))
    assert not outcome_gate_requires_real(cfg, exc, threshold)
    assert not outcome_gate_requires_real(cfg, exc, math.nextafter(threshold, 1))
    assert outcome_gate_confidence(cfg, exc, threshold) == threshold


@pytest.mark.parametrize('bad', [None, True, float('nan'), float('inf'), -1., 1.01, '0.9'])
def test_missing_confidence_executes_except_timeout(bad):
    cfg = SimpleNamespace(execution_gate=ExecutionGateConfig(True, Q3, Q2))
    assert outcome_gate_confidence(cfg, None, bad) is None
    assert outcome_gate_requires_real(cfg, None, bad)
    assert outcome_gate_requires_real(cfg, 'ValueError', bad)
    assert not outcome_gate_requires_real(cfg, 'TimeoutError', bad)


@pytest.mark.parametrize('stage', ['draft', 'debug', 'improve'])
@pytest.mark.parametrize('exc,expected', [(None, True), ('ValueError', False), ('TimeoutError', False)])
def test_periodic_gate_reuses_raw_score_and_real_result(tmp_path, stage, exc, expected):
    from aide_simulator_acceptance_test import _agent, _Real
    from aide.interpreter import ExecutionResult
    from aide.journal import Node
    from aide.run import validate_on_best
    from aide.utils.metric import MetricValue
    from aide.utils.execution_cost import ExecutionCostRecorder
    agent = _agent(tmp_path)
    agent.acfg.execution_gate = ExecutionGateConfig(True, Q3, Q2)
    prior = agent.journal.nodes[0]
    prior.metric = MetricValue(.95, maximize=True)
    parent = None if stage == 'draft' else prior
    if stage == 'debug':
        parent = Node(code='broken', is_buggy=True)
        agent.journal.append(parent)
    agent.search_policy = lambda: parent
    agent._improve_prompt = lambda _: {}
    agent._build_debug_prompt = lambda *args: {}
    agent._exc_scores = Mock(return_value={'max_p': .99, 'top_exc': 'ValueError', 'p_top_exc': .99})
    agent.step(lambda *args: ExecutionResult(['Validation score: 0.9'], 0, exc,
                 predicted_submission_exists='true' if exc is None else 'false'))
    node = agent.journal.nodes[-1]
    assert node.stage_name == stage
    recorder = ExecutionCostRecorder(tmp_path / 'cost')
    agent.execution_cost_recorder = recorder
    real = _Real(tmp_path, sim_only=True)
    assert validate_on_best(agent, agent.journal, real, tmp_path, periodic=True) is expected
    assert real.calls == int(expected)
    assert agent._exc_scores.call_count == int(exc != 'TimeoutError')
    decisions = [json.loads(s) for s in recorder.path.read_text().splitlines()
                 if json.loads(s)['event'] == 'execution_decision']
    assert decisions[0]['gate_score_type'] == 'raw_p_max'
    if exc != 'TimeoutError':
        assert node.sim_p_max == decisions[0]['confidence'] == decisions[0]['gate_confidence'] == .99
    if exc in (None, 'TimeoutError'):
        assert not validate_on_best(agent, agent.journal, real, tmp_path, periodic=True, force=True)
        assert real.calls == int(expected)
