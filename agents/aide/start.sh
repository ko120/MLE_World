#!/bin/bash
set -x # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

# Per-run snapshot handshake (enroot path only: agents/run.py sets
# AIDE_SNAPSHOT_SHA and binds snapshot_check.sh + this start.sh from the run
# group's agent_snapshot). Fails the run at t=0 on stale/mis-bound agent code
# or unknown AIDE_* knobs. The docker path bakes current code at image build
# (run_mlebench_aide.sh src-hash rebuild) and never sets the var.
if [ -n "${AIDE_SNAPSHOT_SHA:-}" ]; then
  bash ${AGENT_DIR}/snapshot_check.sh || exit 1
fi

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE
# check that we can use the GPU in PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"
# check that we can use the GPU in TensorFlow
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))"

# convert $TIME_LIMIT_SECS to more readable format for prompt
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# overwrite instructions.txt with instructions_obfuscated.txt if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/instructions_obfuscated.txt /home/instructions.txt
fi

# start a new file to store the full instructions, starting with general instructions
cp /home/instructions.txt ${AGENT_DIR}/full_instructions.txt

# Update instructions for agent-specific details: replace `/home/` paths to make paths relative
# (since the agent will have its own copies of these files in its workspace).
# e.g. /home/submission/submission.csv -> submission/submission.csv
sed -i 's|/home/||g' ${AGENT_DIR}/full_instructions.txt
# we'll take care of moving things to home/submission/ ourselves

# move on to agent-specific instructions, with a linebreak in between
# substitute env variables into additional_notes.txt and append result to full_instructions.txt
echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt
# finally, append the comp instructions, with a linebreak in between
printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt

# overwrite description.md with description_obfuscated.md if $OBFUSCATE is set
if [ "$OBFUSCATE" = "true" ]; then
  if [ ! -w /home/data/ ]; then
    echo "Obfuscation not implemented for read-only mounts"
    exit 1
  fi
  mv /home/data/description_obfuscated.md /home/data/description.md
fi
cat /home/data/description.md >> ${AGENT_DIR}/full_instructions.txt

# symbolic linking
# agent will write to AGENT_DIR/workspaces/exp/ and AGENT_DIR/logs/exp
# we will mirror the contents of these to CODE_DIR, LOGS_DIR, and SUBMISSION_DIR

# these need to pre-exist for the symbolic links to work
mkdir -p ${AGENT_DIR}/workspaces/exp
mkdir -p ${AGENT_DIR}/logs
# symbolic linking
ln -s ${LOGS_DIR} ${AGENT_DIR}/logs/exp
ln -s ${CODE_DIR} ${AGENT_DIR}/workspaces/exp/best_solution
ln -s ${SUBMISSION_DIR} ${AGENT_DIR}/workspaces/exp/best_submission

# Per-competition eval metric + direction injected by the harness (run.py).
# Forward as OmegaConf overrides only when present; empty => the agent keeps
# its null defaults and falls back to the LLM-judged metric/direction.
METRIC_ARGS=()
if [ -n "${METRIC_NAME:-}" ]; then
  METRIC_ARGS+=("agent.eval_metric=${METRIC_NAME}")
fi
if [ -n "${METRIC_LOWER_IS_BETTER:-}" ]; then
  METRIC_ARGS+=("agent.eval_lower_is_better=${METRIC_LOWER_IS_BETTER}")
fi

# run with timeout, and print if timeout occurs. -k 120: escalate to SIGKILL
# 2 min after the deadline SIGTERM. GPU/NFS I/O can delay process shutdown;
# escalation bounds timeout's wait, and the harness also sweeps descendants
# through agents/runtime_enroot.py:run_in_enroot.
timeout -k 120 $TIME_LIMIT_SECS aide data_dir="/home/data/" desc_file="${AGENT_DIR}/full_instructions.txt" \
  exp_name="exp" \
  "${METRIC_ARGS[@]}" \
  $@ # forward the bash arguments to aide
rc=$?
if [ $rc -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
elif [ $rc -eq 137 ]; then
  echo "Timed out after $TIME_LIMIT (agent survived SIGTERM; SIGKILLed after the 120s -k grace)"
fi
# Preserve the agent/timeout status for the harness. In particular, the final
# if statement otherwise turns an agent crash into a successful shell exit.
exit "$rc"
