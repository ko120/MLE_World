# MLE-World

[Website](https://mle-world.github.io/) | [Dataset](https://huggingface.co/datasets/ko120/MLE-World) | [Model](https://huggingface.co/ko120/MLE-World-Qwen3.5-4B)

Run the paper's gated and ungated experiments with `run_paper.py`.

## Install

Use Linux, Conda, [Enroot](https://github.com/NVIDIA/enroot/blob/main/doc/installation.md), and an NVIDIA GPU allocation.
Run the commands below in Bash from this repository's root directory.

```bash
conda create -n mle_world python=3.11 -y
conda activate mle_world
python -m pip install -e .
```

## Task data

Create a **Legacy API Key** in [Kaggle settings](https://www.kaggle.com/settings/api).
Save the downloaded `kaggle.json` file as `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`.
Accept each competition's rules on Kaggle before downloading its data.

Set the data path and download all 17 tasks:

```bash
DATA_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/mle-bench/data"
mlebench prepare --list configs/tasks.txt --data-dir "$DATA_DIR"
```

If the tasks are already prepared, set `DATA_DIR` to that directory and skip these downloads.

## Execution image

Set `ENROOT_IMAGE` to an AIDE image for your node's CPU architecture:

```bash
ENROOT_IMAGE="$HOME/aide.$(uname -m).sqsh"
```

If you do not have an image, run the builder on an allocated GPU node.
It downloads Ubuntu 22.04 and installs the agent dependencies.
Install `rsync` first. Use local storage with space for the unpacked image plus 110 GiB for export.
The builder replaces the output image if it exists.

```bash
BUILD_DIR="${TMPDIR:-/tmp}/mle-world-build-$USER"
mkdir -p "$BUILD_DIR/rootfs"
ENROOT_DATA_PATH="$BUILD_DIR/rootfs" EXPORT_DIR="$BUILD_DIR" \
  OUT_SQSH="$ENROOT_IMAGE" bash scripts/image/build_aide_enroot.sh
```

Reuse an existing compatible image to keep its package versions.
See the [runtime requirements](agents/README.md#execution-image).

## Model servers

Use the [server commands](agents/README.md#model-servers) to start these models:

| Function | Paper model | API model name |
| --- | --- | --- |
| Code | Qwen3.5-9B | `gpt-qwen3-vllm` |
| Feedback | Qwen3-Coder-Next BF16 | `gpt-coderbf16-fb-vllm` |
| Execution prediction | MLE-World Qwen3.5-4B, checkpoint-6685 | `qwen3_5-4b-exc-token-sim` |

Set the server addresses:

```bash
CODER_URL=http://coder-host:8000/v1
FEEDBACK_URL=http://feedback-host:8002/v1
WORLD_URL=http://world-host:8001/v1
```

## Paper experiments

Each regime runs seven configurations on 17 tasks with seeds 34, 35, and 36: 357 trials with a 12-hour limit each.
The [paper configuration](configs/paper.yaml) defines the tasks, budgets, and [gate settings](configs/gates/).

`RUN_OPTIONS` is a Bash array that stores the options shared by both commands.
`"${RUN_OPTIONS[@]}"` passes each array item as a separate command argument.
Run this block in the same Bash session as the setup commands above.

```bash
RUN_OPTIONS=(
  --grade
  --coder-url "$CODER_URL"
  --feedback-url "$FEEDBACK_URL"
  --world-url "$WORLD_URL"
  --enroot-image "$ENROOT_IMAGE"
  --data-dir "$DATA_DIR"
  --n-workers 1
  --gpu-device 0
)

python run_paper.py "${RUN_OPTIONS[@]}" \
  --regime ungated --output-dir runs/paper-ungated

python run_paper.py "${RUN_OPTIONS[@]}" \
  --regime gated --output-dir runs/paper-gated
```

Use a new output directory for each run. Add `--dry-run` to inspect commands without execution.
Set `--n-workers` and `--gpu-device` for your allocation.

## Results

Within each output directory, check:

- `runs/<run-group>/*_grading_report.json` for submission validity and scores.
- `costs/*.csv` for execution costs and model usage.

The runner also saves the settings in `agent-config.yaml` and `launch.json`.
