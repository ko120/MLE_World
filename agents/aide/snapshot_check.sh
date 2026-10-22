#!/bin/bash
# Agent-code handshake for enroot runs, delivered to the container via the
# per-run-group snapshot binds (agents/run.py:run_in_enroot; snapshot created
# by run_agent.py via agents/run.py:snapshot_agent_code).
#
# Modes:
#   snapshot_check.sh hash DIR   Print the content hash of DIR. The host uses
#                                this at snapshot time; verify mode recomputes
#                                it in-container -> one implementation, no
#                                algorithm drift between the two sides.
#   snapshot_check.sh            Verify mode, run by start.sh inside the
#                                container when AIDE_SNAPSHOT_SHA is set:
#                                1) recompute the hash of $AGENT_DIR/src/aide
#                                   and compare to $AIDE_SNAPSHOT_SHA. Catches
#                                   a missing/misconfigured snapshot bind,
#                                   where the container would otherwise
#                                   silently run stale image code.
#                                2) fail on AIDE_* env vars this agent code
#                                   does not know - the AIDE_ENV_EXTRA_NOTE
#                                   silent-no-op class (config ships a knob,
#                                   stale code never reads it).
#                                   AIDE_KNOB_CHECK=warn downgrades to a
#                                   warning for emergencies.
set -uo pipefail

content_hash() {
  # Path-independent (cd + relative find) variant of the docker path's
# src-hash (scripts/legacy/run_mlebench_aide.sh AIDE_SRC_HASH): same exclusions,
  # locale-pinned order, name+content sensitive.
  (cd "$1" && find . -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
    | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1)
}

if [ "${1:-}" = "hash" ]; then
  content_hash "$2"
  exit 0
fi

: "${AGENT_DIR:?}"
: "${AIDE_SNAPSHOT_SHA:?}"

actual="$(content_hash "$AGENT_DIR/src/aide")"
if [ "$actual" != "$AIDE_SNAPSHOT_SHA" ]; then
  echo "SNAPSHOT_CHECK FATAL: src/aide hash $actual != expected $AIDE_SNAPSHOT_SHA" >&2
  echo "SNAPSHOT_CHECK FATAL: snapshot bind missing/misconfigured; refusing to run stale image code" >&2
  exit 1
fi
echo "SNAPSHOT_OK src/aide sha256=$actual"
if [ -f "$AGENT_DIR/PROVENANCE.json" ]; then
  cat "$AGENT_DIR/PROVENANCE.json"
fi

# --- unknown-knob guard -----------------------------------------------------
# Every AIDE_* env var the current agent code or harness legitimately uses.
# Adding a knob to src/aide without listing it here fails the next launch
# loudly - that is the point. Keep in sync: grep os.getenv src/aide.
KNOWN_AIDE_KNOBS=(
  AIDE_BON_CSV_MAX_BYTES        # agent.py: BoN candidate CSV copy cap
  AIDE_EXEC_GRACEFUL_SHUTDOWN_S # interpreter.py: cooperative-exit grace
  AIDE_EXEC_INTERRUPT_GRACE_S   # interpreter.py: process-group SIGINT grace
  AIDE_EXEC_TERM_GRACE_S        # interpreter.py: process-group SIGTERM grace
  AIDE_EXEC_KILL_GRACE_S        # interpreter.py: process-group SIGKILL wait
  AIDE_EXEC_SPAWN_TIMEOUT_S     # interpreter.py: fresh-worker startup bound
  AIDE_MEM_PER_TASK_GB          # interpreter.py: per-exec-process RLIMIT_DATA cap in GiB (0 = off)
  AIDE_DEBUG_USE_ANALYSIS       # backend: gemini feedback path
  AIDE_OPENAI_TIMEOUT           # backend_openai.py: local vLLM client read-timeout (secs)
  AIDE_VERTEX_REASONING_EFFORT  # backend_openai.py: Vertex thinking level
  AIDE_VERTEX_RETRY_MAX_TIME    # backend_openai.py: Vertex retry deadline (secs)
  AIDE_VERTEX_TIMEOUT           # backend_openai.py: Vertex client read-timeout (secs)
  AIDE_DRAFT_MAX_TOKENS         # backend: per-stage token cap
  AIDE_ENV_EXTRA_NOTE           # agent.py: extra Installed-Packages sentence
  AIDE_MODEL_BASE_URLS          # backend_openai.py: per-model endpoint routing JSON
  AIDE_MODEL_SAMPLING           # backend_openai.py: per-model sampling JSON (REPLACES AIDE_SAMPLING)
  AIDE_SAMPLING                 # backend: sampling overrides JSON
  AIDE_SIMULATOR_API_KEY        # simulator client auth
  AIDE_SIMULATOR_BASE_URL       # simulator vLLM endpoint
  AIDE_SIMULATOR_PROMPT_FORMAT  # simulator.py: env_v1_reorder_excfirst_token (default) | env_v1_reorder_excfirst
  AIDE_SIMULATOR_SAMPLING       # simulator sampling overrides
  AIDE_EXC_SCORING             # simulator.py: auto (default) | single_token | forced_suffix | echo
  AIDE_EXC_SCORE_WORKERS       # simulator.py: concurrent suffix requests per candidate
  AIDE_PCLEAN_CHUNK             # simulator.py: classes per prompt-logprob request (OOM guard)
  AIDE_PCLEAN_SCORE             # simulator.py: p_clean() scalar (p_clean | max_p)
  AIDE_GRADING_SERVER_URL       # agents/run.py: per-container grading server
  AIDE_VALIDATE_SERVER_URL      # agents/run.py: per-container validate endpoint
  AIDE_RUN_TOKEN                # agents/run.py: host descendant ownership token
  AIDE_RUN_SEED                 # run.py: matched Python and NumPy runtime seed
  AIDE_JOB_TOKEN                # enroot launcher: whole-job ownership token
  AIDE_SHUTDOWN_RESERVE_S       # run.py/agent.py: clean pre-timeout reserve
  AIDE_SNAPSHOT_SHA             # agents/run.py: this handshake
  AIDE_KNOB_CHECK               # this script: fail (default) | warn
  AIDE_SRC_BIND                 # legacy launcher flag; ignored (snapshots always on)
  AIDE_CPU_AFFINITY             # run_agent.py: opt-in per-container taskset pinning (host-side)
  AIDE_ENROOT_CREATE_CONCURRENCY # agents/run.py: concurrent `enroot create` cap (host-side; fs.file-max)
  AIDE_CPU_OVERCOMMIT           # run_agent.py: affinity slice width factor (host-side)
  AIDE_CPU_PER_TASK             # run_agent.py: affinity slice width in cores (host-side; wins over the factor)
)
unknown=""
while IFS= read -r var; do
  known=0
  for k in "${KNOWN_AIDE_KNOBS[@]}"; do
    if [ "$var" = "$k" ]; then known=1; break; fi
  done
  if [ "$known" != 1 ]; then unknown="$unknown $var"; fi
done < <(env | sed -n 's/^\(AIDE_[A-Za-z0-9_]*\)=.*/\1/p' | sort -u)
if [ -n "$unknown" ]; then
  echo "SNAPSHOT_CHECK: unknown AIDE_* env var(s):$unknown" >&2
  echo "SNAPSHOT_CHECK: a config/launcher ships a knob this agent code does not list (KNOWN_AIDE_KNOBS above) - version skew?" >&2
  if [ "${AIDE_KNOB_CHECK:-fail}" = "warn" ]; then
    echo "SNAPSHOT_CHECK: continuing (AIDE_KNOB_CHECK=warn)" >&2
  else
    exit 1
  fi
fi
echo "SNAPSHOT_CHECK passed"
