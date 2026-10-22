# Runtime setup

Use commands from the repository root.
See the [main README](../README.md) for data preparation, experiment commands, and results.

## Execution image

Use Linux, Enroot, an NVIDIA GPU allocation, and working user and PID namespaces.
The runtime requires `unshare --mount-proc --kill-child`.
Use local storage that supports reflinks for `ENROOT_DATA_PATH`.
The runtime creates separate writable roots for the agent and grader.
Only the grader receives private labels.

Reuse an existing AIDE `.sqsh` image on nodes with the same CPU architecture.
Pass its path with `--enroot-image`.
Use the existing prepared competition data with `--data-dir`.
The image must contain no competition data.

If no compatible image is available, `scripts/image/build_aide_enroot.sh` can build one.
The builder requires Enroot, `rsync`, internet access, and a visible NVIDIA GPU.
Set `OUT_SQSH` to the output file and `EXPORT_DIR` to local storage with at least 110 GiB free.
It replaces the output file if it exists.
The build uses Ubuntu 22.04, Python 3.11, and CUDA 12.8 PyTorch packages.
Some dependencies are not pinned. Reuse the original image for its exact package environment.

`run_paper.py` selects local Enroot storage by default.
Set `ENROOT_DATA_PATH` to override that path.
Each worker needs space for its agent and grader roots.
Set `--n-workers` and `--gpu-device` for the available allocation.
On Slurm, run the commands inside your site's GPU allocation.

## Model servers

Reuse idle model servers when available.
To start a server, use a serving environment with vLLM 0.27.1 and PyYAML.
Keep serving dependencies separate from the `mle_world` host environment.

Replace each model path below with a local checkpoint path.
Use a separate terminal or job for each server.
The code and feedback profiles each target one B200 GPU.
Keep server GPUs separate from the agent's execution GPUs.
Set `CUDA_VISIBLE_DEVICES` for each allocation.

Start the code server:

```bash
python scripts/serve/start_vllm_server.py \
  --config configs/serve/coder.yaml --model /path/to/Qwen3.5-9B \
  --served-model-name gpt-qwen3-vllm --port 8000
```

Start the feedback server:

```bash
python scripts/serve/start_vllm_server.py \
  --config configs/serve/feedback.yaml --model /path/to/Qwen3-Coder-Next \
  --served-model-name gpt-coderbf16-fb-vllm --port 8002
```

Start the world-model server:

```bash
python scripts/serve/start_vllm_server.py \
  --config configs/serve/world.yaml --model /path/to/checkpoint-6685 \
  --served-model-name qwen3_5-4b-exc-token-sim --port 8001
```

Use the trained MLE-World checkpoint with all 82 `EXC_*` tokens.
The base Qwen3.5-4B checkpoint does not contain this training.
Model weights are supplied separately.
The world-model API must support `/tokenize` and `logprob_token_ids` with raw log probabilities.

Add `--dry-run` to inspect a server command without starting it.
The launcher monitors generation and restarts a failed server.
It writes token counters to `runs/vllm_metrics/`.

Check code and feedback generation after startup:

```bash
bash scripts/serve/validate_serve.sh http://coder-host:8000/v1 gpt-qwen3-vllm
bash scripts/serve/validate_serve.sh http://feedback-host:8002/v1 gpt-coderbf16-fb-vllm
```

Return to the host environment before you run experiments:

```bash
conda activate mle_world
```

Pass the server URLs to `run_paper.py` as shown in the main README.

## Code and configuration

| Path | Purpose |
| --- | --- |
| `../run_paper.py` | Compile the paper profiles, run trials, grade submissions, and report costs |
| `../configs/paper.yaml` | Select tasks, seeds, policies, gates, and time limits |
| `aide/src/aide/` | Run the search agent and execution simulator |
| `runtime_enroot.py` | Start isolated agent and grader containers |
| `snapshots.py` | Save and verify the agent and grader source for each run group |

The [AIDE guide](aide/src/README.md) describes the code and development checks.
