# AIDE in MLE-World

This is the MLE-World fork of [WecoAI AIDE](https://github.com/WecoAI/aideml).
It retains AIDE's draft/debug/improve search and adds world-model execution,
candidate selection, gated real checks, and submission validation. Upstream's
MIT license and attribution are retained in [LICENSE](LICENSE). The Python
package and console command remain `aideml` and `aide` for compatibility.

For competition setup, container execution, and the paper suite, see
[the agents guide](../../README.md). Installing the upstream PyPI release does
not install this fork.

## Installation

Use Python 3.11 or later for the combined MLE-bench harness and agent. The host
harness, model servers, and competition execution container are separate
environments. The pinned competition packages are intended for the execution
image; do not combine them with the harness's different dependency constraints.

From the repository root, install the full agent execution environment with:

```bash
python -m pip install -e agents/aide/src
```

The dependency files have distinct roles:

| File | Contents |
| --- | --- |
| `requirements-agent.txt` | Agent orchestration, providers, data preview, and code formatting |
| `requirements-execution.txt` | Scientific/ML packages available to generated programs |
| `requirements-dev.txt` | Test runner |
| `requirements.txt` | Complete environment, including all three groups above |

The default package installation retains the complete dependency set. For
agent-only development in an isolated virtual environment, install the smaller
set and register this checkout without pulling in competition libraries:

```bash
python -m pip install -r agents/aide/src/requirements-agent.txt \
  -r agents/aide/src/requirements-dev.txt
python -m pip install --no-deps -e agents/aide/src
```

This smaller environment runs mocked agent tests; real generated programs need
their execution libraries. Harness tests also need the MLE-bench dependencies.
The complete test suite additionally needs PyTorch for the DataLoader and
multiprocessing regressions. Install the harness and the CPU test dependency in this development environment:

```bash
python -m pip install -e '.[dev]'
python -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```
Provider credentials and endpoint URLs must be supplied separately.

## Reading the implementation

The runtime follows this flow:

```text
run.py: configure interpreters and budgets
  -> Agent.step: choose draft/debug/improve and a parent
  -> generation_prompts.py: render the task and evidence
  -> generate candidates and observe real or predicted results
  -> candidate_selection.py: rank, compare, or fuse candidates
  -> execute/reuse the selected result and review it
  -> submission_review.py: validate artifacts and record diagnoses
  -> real_validation.py: gate real checks and publish strict improvements
  -> journal.py / utils.artifacts: persist the search and published solution
```

`Agent` owns the shared state. Its prompt, selection, and submission mixins group
methods by responsibility and operate on that same instance. Each mixin's
docstring lists the state it reads or updates. Generation and candidate-judging
calls use the agent's budget-aware `_query` callback; submission review has its
own bounded validation budget.

| Module | What to inspect |
| --- | --- |
| `aide/agent.py` | Parent choice, stage routing, candidate generation, node review, and step lifecycle |
| `aide/generation_prompts.py` | Draft, improve, and diagnosis-specific debug prompts |
| `aide/candidate_selection.py` | Metric/UCB selection, pairwise comparison, and fusion |
| `aide/interpreter.py` | Real subprocess execution, output capture, timeouts, and process cleanup |
| `aide/parallel_interpreter.py` | Race between real execution and a delayed simulator |
| `aide/simulator.py` | Simulator protocol, prediction parsing, and interpreter interface |
| `aide/simulator_scoring.py` | HTTP/tokenizer operations for exception-posterior scoring |
| `aide/real_validation.py` | Execution gates, real-check reserves, and publication decisions |
| `aide/submission_review.py` | CSV checks, leakage review, and evaluation-only private grading |
| `aide/validation_*`, `aide/data_leakage.py` | Evidence, prompts, and validation review contracts |
| `aide/backend/` | Provider routing, structured responses, and retries |
| `aide/utils/config.py`, `config.yaml` | Typed configuration and runtime defaults |

## Policies and evidence

Each stage (`draft`, `debug`, `improve`) has a generation policy: `single`,
`sequential`, `parallel`, `argmax`, `ucb`, `pairwise`, or `fuse`. The separate
`bon_oracle` setting controls evidence: `sim` predicts execution, `race` uses a
real result before its deadline and otherwise a prediction, and `real` executes
every candidate. The resolver in `Agent` validates prerequisites and logs any
downgrade to `single`.

Predictions do not establish that a submission exists. Real candidate results
can be reused while their workspace survives; race workspaces may already have
been removed and require re-execution. A real validated score must strictly
improve the incumbent to replace the published solution. Ties keep the incumbent.
Private test scores are recorded for evaluation and never guide search or enter
prompts. See `real_validation.py` for gate precedence and timeout vetoes.

Configuration retains historical aliases for reproducing older runs. Current
paper policy/gate combinations are listed in `configs/paper.yaml`;
`--dry-run` on `run_paper.py` shows the actual resolved launch commands.

## Reproducibility contracts

- Prompt strings, ordering, and version identifiers affect experiments.
  `fusion_prompts/typed_paco_v13.py` contains the prompt used by the current
  configurations and its required helpers. Older prompt families are removed.
  Its rendered wording is unchanged; wording changes require a new version.
- Offline tools extract standalone helper definitions and protocol constants
  from `agent.py` and `simulator.py`. These source locations remain stable.
- Journal fields, artifact names, and node-lifecycle log messages are consumed
  by corpus builders and evaluation scripts. Preserve their formats in refactors.
- Snapshots freeze the whole agent package, start scripts, and graders. New modules
  must remain inside that package so every task uses its launch-time code.

## Development checks

From the repository root, run both harness and agent suites:

```bash
python -m pytest
```

Pytest configuration includes both `tests/` and `agents/aide/src/aide/tests/`, and
adds the local agent source to the import path. For a focused selection check:

```bash
python -m pytest agents/aide/src/aide/tests/ucb_stage_policy_test.py
```

Most tests use model stubs and temporary workspaces. Interpreter tests execute
small local subprocesses. Enroot integration tests require the corresponding
Linux runtime and namespace support; they do not require competition datasets.
Use the repository's Black/isort configuration when formatting changed code.

## Standalone usage

With execution dependencies and provider access configured:

```bash
aide data_dir="/path/to/public/data" \
  desc_file="/path/to/task-description.txt" \
  agent.code.model="gpt-model-name" agent.feedback.model="gpt-model-name"
```

Model names determine provider routing in `aide/backend/__init__.py`. Task goals
may also be passed as `goal="..."`. The default configuration controls search,
timeouts, and whether simulation is enabled; override it for the intended mode.
Outputs include `journal.json`, `best_solution.py`, and `tree_plot.html` under
the run's log directory, plus generated files under its workspace.
