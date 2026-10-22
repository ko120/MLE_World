#!/usr/bin/env python3
"""Launch a vLLM OpenAI-compatible server from a yaml config.

Consumes the `vllm_server:` block of a yaml config and launches
`python -m vllm.entrypoints.openai.api_server` with translated CLI args.

Example:
    python scripts/serve/start_vllm_server.py \\
        --config configs/serve/coder.yaml \\
        --model /path/to/Qwen3.5-9B \\
        --served-model-name gpt-qwen3-vllm \\
        --port 8000

Unknown args are forwarded to vLLM (e.g. ``--distributed-executor-backend ray``).

Supervision:
    The launched vLLM process is supervised by this script. If vLLM exits
    with a non-zero code (e.g. the v1 EngineCore RPC-timeout fatal error
    that took down the 2026-05-17 run), the supervisor relaunches it.
    A generation healthcheck also covers the *hung-but-alive* failure (the
    2026-07-01 GPU wedge: compute stalled while the api_server kept serving
    /health, so exit-based supervision never fired) — it probes a 1-token
    completion periodically and, after HEALTH_MAX_FAILS consecutive failures,
    SIGKILLs the process tree to force a relaunch. A (re)spawned child that
    never answers its first probe within STARTUP_GRACE_SEC (hung model load,
    e.g. an NFS stall) is killed the same way — launch_vllm's readiness curl
    only guards the first launch, not mid-run restarts.
    Every RE-spawn first waits (bounded) for the previous engine's VRAM to
    drain before launching, so the relaunch doesn't fail vLLM's startup
    free-memory check while the GPU is still releasing memory.
    Anti-flap policy: at most DEFAULT_MAX_RESTARTS crashes within a
    DEFAULT_RESTART_WINDOW_SEC rolling window, then give up. Additionally,
    DEFAULT_MAX_UNREADY_STARTS consecutive (re)starts that die without ever
    serving a probe also give up — those cycles are paced by STARTUP_GRACE_SEC
    so they can never trip the rolling window. SIGTERM / SIGINT received by
    this script is propagated to vLLM and the supervisor exits cleanly (no
    restart). A clean child exit (rc=0) is NOT restarted — the supervisor
    follows the child's intent.

    Exit contract (consumed by run_mlebench_aide*_enroot.sh): the supervisor
    process stays alive as long as relaunching remains viable. It EXITS only
    when (a) it was told to shut down (rc=0) or (b) relaunching is exhausted
    (rc=1, "FATAL" in the log). The launcher watches this PID and turns a
    fatal exit into job policy: abort-for-resubmit early in the run, or ride
    to the agent deadline late to keep banked results.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from typing import Any, Callable

import yaml


# Fields handled by the launcher itself, never forwarded to vLLM as flags.
_OUR_FIELDS = {
    "num_replicas",
    "time_limit",
    "endpoint_json_path",
    "model_path",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "type",
    "max_output_tokens",
    "healthcheck_interval",
    "vllm_local",
    "model",
}

_FIELD_RENAMES = {"model_path": "model"}

_BOOLEAN_FLAGS = {
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "enable_auto_tool_choice",
    "trust_remote_code",
    "disable_log_requests",
    "enable_reasoning",
}

_ENV_VAR_FIELDS = {
    "enable_expert_parallel": "VLLM_ENABLE_EXPERT_PARALLEL",
    "use_deep_gemm": "VLLM_USE_DEEP_GEMM",
}

# Environment-backed switches where false must be propagated explicitly. This
# differs from _ENV_VAR_FIELDS, whose false value means "leave the env unset".
_ENV_BOOL_FIELDS = {
    "use_flashinfer_sampler": "VLLM_USE_FLASHINFER_SAMPLER",
}


# --- supervisor defaults -----------------------------------------------------
# More than DEFAULT_MAX_RESTARTS crashes inside DEFAULT_RESTART_WINDOW_SEC
# seconds → the supervisor gives up and exits non-zero. After
# DEFAULT_RESTART_WINDOW_SEC of healthy uptime the rolling counter resets.
# Chosen so that one isolated crash every few hours is absorbed silently,
# but a fundamentally broken config (e.g. bad model path, OOM at startup
# every time) caps out within minutes instead of looping forever.
DEFAULT_MAX_RESTARTS = 5
DEFAULT_RESTART_WINDOW_SEC = 600

# Consecutive (re)starts that die without EVER serving a single generation
# probe → relaunching is exhausted (dead weights path, NFS outage, GPU wedge
# that survives process death) → give up and exit 1. This is a separate axis
# from the rolling-window cap above: never-ready cycles are paced by
# STARTUP_GRACE_SEC (2400s), so a permanently load-hung model produces crashes
# 40 min apart — the 600s window never accumulates two of them and the old
# supervisor relaunched forever while the job idled to walltime as a zombie.
# A start that serves at least one probe is a real recovery and resets the
# count. Only armed when the generation healthcheck is on (it defines "ready").
DEFAULT_MAX_UNREADY_STARTS = 3

# --- respawn free-VRAM gate ---------------------------------------------------
# A SIGKILLed engine's VRAM drains over MINUTES, not instantly (2026-07-13 sim
# replica: free memory rose 11 → 128 → 158 GiB across three immediate respawns,
# each of which failed vLLM's startup free-memory check and burned a never-ready
# strike — the supervisor gave up about a minute before the drain would have
# completed). So before every RE-spawn, wait until each GPU the child will use
# has free ≥ gpu_memory_utilization × total + margin (vLLM's own startup
# requirement), up to DRAIN_TIMEOUT_SEC — then launch anyway, so the never-ready
# give-up path still exists for genuinely leaked GPUs.
DRAIN_TIMEOUT_SEC = 600.0
DRAIN_POLL_SEC = 15.0
DRAIN_MARGIN_MIB = 2048.0

# --- generation healthcheck defaults -----------------------------------------
# Covers the "hung-but-alive" failure that exit-based supervision misses (the
# 2026-07-01 half2 run: GPU compute stalled while the api_server kept answering,
# so the process never exited). We actively generate 1 token every
# HEALTH_INTERVAL_SEC; once the server has served at least once, HEALTH_MAX_FAILS
# consecutive failures (each capped at HEALTH_TIMEOUT_SEC) means generation is
# wedged → kill the process tree so the restart loop relaunches it. That wedge
# was confirmed restart-recoverable (soft CUDA-context hang; GPU cleared on
# process death, no reboot / ECC fault).
HEALTH_INTERVAL_SEC = 60.0
# 90 not 30: a 1-token probe on a HEALTHY server can queue for tens of seconds
# behind a waiting queue of 32k-token prefills at 30+ concurrent agents (the
# per-request latency blowup seen on the 54-agent runs), and a false kill drops
# every in-flight generation plus a model reload. 90s stays above saturation
# queueing while keeping wedge detection at HEALTH_MAX_FAILS x (interval +
# timeout) ~ 7.5 min — small vs a 5h budget.
HEALTH_TIMEOUT_SEC = 90.0
HEALTH_MAX_FAILS = 3
# A child that has NEVER answered a probe since its (re)spawn gets this long to
# finish model load before it's presumed load-hung and killed (same SIGKILL ->
# relaunch path). Mid-run restarts have no other readiness watchdog —
# launch_vllm's one-shot curl only guards the first launch — so without this a
# relaunch that hangs loading (e.g. an NFS stall on the 75G coder read) would be
# hosted forever while every agent retries against a dead endpoint. Sized to the
# slowest legitimate cold load (coder VLLM_READY_TIMEOUT=1800) + 10 min slack.
STARTUP_GRACE_SEC = 2400.0

# --- forward-progress watchdog (metrics-based) -------------------------------
# The 1-token probe above can pass on a slow/partially-wedged engine while real
# 32k-prefill serving makes zero progress (the 2026-07-07 base-coder wedge: 80
# requests queued, 0 tok/s, /metrics token counters frozen, yet a trivial ping
# might still squeak through). So additionally scrape /metrics and kill when
# there is pending work but the cumulative token counters do not advance across
# PROGRESS_STALL_MAX consecutive intervals. This never arms when the queue is
# empty (pending == 0), so it cannot false-fire on an agents-are-executing idle
# lull — the exact ambiguity that makes log-staleness alone unsafe. It also
# won't kill a saturated-but-healthy server: if the counter keeps advancing the
# stall counter resets, so it forgives 1-token-probe timeouts under load.
PROGRESS_STALL_MAX = 3
METRICS_TIMEOUT_SEC = 10.0


def _read_progress(
    port: int, timeout: float, csv_path: str | None = None
) -> tuple[int, float] | None:
    """Scrape vLLM's Prometheus /metrics → (pending_requests, cumulative_tokens).

    pending = num_requests_running + num_requests_waiting; cumulative_tokens =
    prompt_tokens_total + generation_tokens_total (monotonic). Returns None if
    /metrics is unreachable or unparseable — the caller treats that as
    inconclusive (the 1-token probe covers total unresponsiveness). Metric names
    verified against vLLM 0.23.0; both the `_total` and bare counter-sample
    spellings are accepted for version resilience.

    With `csv_path`, each successful scrape also appends a row
    (epoch,running,waiting,prompt_tokens,generation_tokens,prefix_queries,
    prefix_hits) — the per-server token record that per-arm compute accounting
    (scripts/sample/flops_report.py) consumes. Counters restart from 0 whenever the
    supervisor respawns vLLM, so consumers must sum POSITIVE deltas, never read
    last-minus-first. Telemetry failures never propagate: the watchdog verdict
    is returned regardless."""
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/metrics", timeout=timeout
        ) as resp:
            if resp.status != 200:
                return None
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    targets = {
        "running": ("vllm:num_requests_running",),
        "waiting": ("vllm:num_requests_waiting",),
        "gen": ("vllm:generation_tokens_total", "vllm:generation_tokens"),
        "prompt": ("vllm:prompt_tokens_total", "vllm:prompt_tokens"),
        # Prefix-cache hit tokens cost ~0 prefill FLOPs; the report subtracts
        # them. Name candidates cover the V1 gpu_-prefixed and bare spellings;
        # absent on --no-enable-prefix-caching servers (row records 0).
        "pfx_q": (
            "vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries",
            "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries",
        ),
        "pfx_h": (
            "vllm:gpu_prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits",
            "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits",
        ),
    }
    vals = {k: 0.0 for k in targets}
    seen = {k: False for k in targets}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for key, names in targets.items():
            if name in names:
                try:
                    vals[key] += float(line.rsplit(" ", 1)[1])
                    seen[key] = True
                except (ValueError, IndexError):
                    pass
                break
    # Require the request gauges + generation counter; otherwise version skew
    # (or metrics disabled) makes the scrape unreliable → inconclusive.
    if not (seen["running"] and seen["waiting"] and seen["gen"]):
        return None
    if csv_path is not None:
        try:
            write_header = not (
                os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            )
            with open(csv_path, "a") as f:
                if write_header:
                    f.write(
                        "epoch,running,waiting,prompt_tokens,"
                        "generation_tokens,prefix_queries,prefix_hits\n"
                    )
                f.write(
                    f"{int(time.time())},{int(round(vals['running']))},"
                    f"{int(round(vals['waiting']))},{int(vals['prompt'])},"
                    f"{int(vals['gen'])},{int(vals['pfx_q'])},{int(vals['pfx_h'])}\n"
                )
        except OSError:
            pass
    pending = int(round(vals["running"] + vals["waiting"]))
    return pending, vals["gen"] + vals["prompt"]


def _probe_generation(port: int, model: str, timeout: float) -> bool:
    """True iff the server returns a 1-token completion within `timeout` seconds.
    Unlike /health (which stays 200 during a GPU wedge) this exercises the GPU."""
    body = json.dumps(
        {"model": model, "prompt": "ping", "max_tokens": 1, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _sigkill_tree(child: subprocess.Popen | None) -> None:
    """SIGKILL the child's whole process group (vLLM api_server + EngineCore
    workers) so a restart isn't blocked by an orphan still holding the GPU."""
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            child.kill()
        except ProcessLookupError:
            pass


def _visible_gpu_indices(env: dict[str, str]) -> list[int] | None:
    """Physical GPU indices from the child's CUDA_VISIBLE_DEVICES, or None for
    "all GPUs" (unset, or UUID/MIG forms we don't parse)."""
    raw = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw:
        return None
    try:
        return [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        return None


def _gpu_mem_mib() -> list[tuple[int, float, float]] | None:
    """[(index, free_mib, total_mib)] per GPU via nvidia-smi; None if the query
    fails (no GPUs / driver hiccup) — callers must not block on that."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=20,
        )
    except Exception:
        return None
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
        except (ValueError, IndexError):
            continue
    return rows or None


def _wait_gpu_drain(
    env: dict[str, str],
    mem_frac: float,
    log: Callable[[str], None],
    abort: Callable[[], bool] | None = None,
    timeout: float = DRAIN_TIMEOUT_SEC,
    poll: float = DRAIN_POLL_SEC,
) -> bool:
    """Block until every GPU the child will use has enough free memory for
    vLLM's startup check, `timeout` elapses, or `abort()` turns true. Returns
    True iff the bar was met. On nvidia-smi failure returns True immediately:
    a broken probe must never block recovery."""
    deadline = time.monotonic() + timeout
    while True:
        if abort is not None and abort():
            return False
        rows = _gpu_mem_mib()
        if rows is None:
            log("[supervisor] nvidia-smi unavailable; skipping free-VRAM gate")
            return True
        vis = _visible_gpu_indices(env)
        if vis is not None:
            picked = [r for r in rows if r[0] in vis]
            rows = picked or rows
        # Cap the bar below the physical total: driver/context overhead means a
        # fully drained GPU never reports 100% free, and an unreachable bar
        # would turn every respawn into a dead full-timeout wait.
        short = []
        for i, free, total in rows:
            need = min(mem_frac * total + DRAIN_MARGIN_MIB, total - 1024.0)
            if free < need:
                short.append((i, free, need))
        if not short:
            return True
        status = ", ".join(f"gpu{i} {free / 1024:.1f}/{need / 1024:.1f} GiB"
                           for i, free, need in short)
        if time.monotonic() >= deadline:
            log(f"[supervisor] free-VRAM gate timed out after {timeout:.0f}s "
                f"({status}); launching anyway")
            return False
        log(f"[supervisor] waiting for GPU memory to drain before relaunch: {status}")
        time.sleep(poll)


def supervise(
    cmd: list[str],
    env: dict[str, str] | None = None,
    *,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    restart_window_sec: float = DEFAULT_RESTART_WINDOW_SEC,
    max_unready_starts: int = DEFAULT_MAX_UNREADY_STARTS,
    mem_frac: float | None = None,
    health_check: Callable[[], bool] | None = None,
    health_interval: float = HEALTH_INTERVAL_SEC,
    health_max_fails: int = HEALTH_MAX_FAILS,
    startup_grace: float = STARTUP_GRACE_SEC,
    progress_probe: Callable[[], "tuple[int, float] | None"] | None = None,
    progress_max_stalls: int = PROGRESS_STALL_MAX,
    log: Callable[[str], None] = print,
) -> int:
    """Run cmd as a subprocess; restart on ANY non-shutdown exit; cap restarts; propagate signals.

    Returns:
        0 if we received SIGTERM/SIGINT and propagated it to the child.
        1 if the child exited more than max_restarts times within
          restart_window_sec without an external shutdown, if
          max_unready_starts consecutive (re)starts died without ever
          serving a probe, or if the child could not be spawned.

    Signal contract:
        SIGTERM / SIGINT sent to this process is forwarded to the child,
        the child is reaped, and the supervisor returns 0 without
        restarting. This matches run_mlebench.sh's `trap cleanup EXIT`
        behaviour: `kill -TERM $VLLM_PID` -> graceful shutdown.

    Why we restart on rc=0 too:
        vLLM 0.21 exits rc=0 even on EngineDeadError — its API server
        runs the normal shutdown handlers after engine death, so the
        process returns 0 despite a fatal crash. The 2026-05-19 runs
        died twice this way (~2h and ~80min apart) with rc=0; the old
        "rc=0 -> don't restart" rule mistook those for intentional
        shutdowns and left the agent containers hammering a dead
        endpoint. Now: only an explicit SIGTERM/SIGINT to the supervisor
        counts as "intentional", everything else is a crash.

    Respawn gate:
        With mem_frac set (vLLM's gpu_memory_utilization), every respawn
        first waits — up to DRAIN_TIMEOUT_SEC — for the previous engine's
        VRAM to drain (free ≥ mem_frac·total + margin on the child's GPUs).
        A SIGKILLed engine frees memory over minutes; respawning immediately
        fails vLLM's startup free-memory check and burns the never-ready
        strikes on a GPU that is seconds from ready (the 2026-07-13 sim
        replica died this way). On timeout it launches anyway, preserving
        the give-up path for genuinely leaked GPUs.
    """
    env = dict(env) if env is not None else os.environ.copy()
    state: dict[str, Any] = {"child": None, "shutting_down": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        state["shutting_down"] = True
        log(f"[supervisor] received signal {signum}; terminating child")
        child = state["child"]
        if child is not None and child.poll() is None:
            try:
                child.terminate()
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    def _health_loop() -> None:
        last_child = None
        ready = False
        fails = 0
        stall = 0
        last_tokens: float | None = None
        started = time.monotonic()
        while not state["shutting_down"]:
            time.sleep(health_interval)
            if state["shutting_down"]:
                return
            child = state["child"]
            if child is not last_child:        # a (re)start → wait for ready again
                last_child, ready, fails = child, False, 0
                stall, last_tokens = 0, None
                started = time.monotonic()
            if child is None or child.poll() is not None:
                continue
            if health_check():
                ready, fails = True, 0
                # Mark THIS incarnation as having served (identity-keyed so a
                # restart can't inherit its predecessor's credit); the restart
                # loop uses it to reset the never-ready strike counter.
                state["ready_child"] = child
            elif ready:                        # was serving, now unresponsive
                fails += 1
                log(f"[supervisor] generation probe failed {fails}/{health_max_fails}")
                # ⚠️ A PROBE TIMEOUT UNDER SATURATION IS NOT A WEDGE. At 128
                # concurrent with a 300s+ e2e latency, a 1-token probe cannot
                # return inside HEALTH_TIMEOUT_SEC however healthy the engine is,
                # so this branch fires forever and each kill costs a full model
                # reload PLUS every in-flight generation. Measured 2026-08-18 on
                # the 4-arm 122b campaign: four kills in 46 minutes, accelerating
                # (23 -> 12 -> 11.5 min apart), ~39% of serve time lost to
                # reloads — while the engine was sustaining 1352 tok/s and
                # answering /v1/models in 5 ms.
                # So consult the token counters before killing: if they are
                # advancing, the engine is BUSY and the probe result says nothing
                # about its health. This is exactly the forgiveness the
                # forward-progress watchdog below was documented to provide but
                # could never deliver — this branch kills and `continue`s before
                # that code runs. A REAL wedge is still caught, by that watchdog:
                # pending > 0 with FROZEN counters is the signature, and it is
                # unaffected by how long a probe takes.
                if progress_probe is not None:
                    prog = progress_probe()
                    if prog is not None:
                        pending, tokens = prog
                        advancing = last_tokens is not None and tokens > last_tokens
                        last_tokens = tokens
                        if advancing:
                            log("[supervisor] probe timed out with "
                                f"{pending} request(s) pending but token counters "
                                "are still advancing — SATURATED, not wedged; "
                                "not killing")
                            fails = 0
                            continue
                if fails >= health_max_fails:
                    log("[supervisor] vLLM generation wedged (process alive, GPU "
                        "stalled); killing process tree to force restart")
                    _sigkill_tree(child)
                    ready, fails, stall, last_tokens = False, 0, 0, None
                    continue
            elif time.monotonic() - started > startup_grace:
                # Never served since (re)spawn: model load is hung. Kill so the
                # restart loop gets another attempt; without this the wedge
                # detection above never arms (`ready` requires one success).
                log(f"[supervisor] vLLM never became ready within "
                    f"{startup_grace:.0f}s of spawn (load hung); killing "
                    "process tree to force restart")
                _sigkill_tree(child)
                started = time.monotonic()     # re-arm pending the respawn
                continue

            # Independent forward-progress watchdog: a wedge can leave real
            # requests queued while the trivial 1-token probe still passes. If
            # there is pending work but the cumulative token counters have not
            # advanced for progress_max_stalls consecutive intervals, the engine
            # is stuck (not merely idle — idle means pending == 0). A None scrape
            # is inconclusive; advancing tokens reset the stall counter.
            if progress_probe is not None and ready and child.poll() is None:
                prog = progress_probe()
                if prog is not None:
                    pending, tokens = prog
                    if last_tokens is not None and pending > 0 and tokens <= last_tokens:
                        stall += 1
                        log(f"[supervisor] no token progress with {pending} "
                            f"request(s) pending {stall}/{progress_max_stalls}")
                        if stall >= progress_max_stalls:
                            log("[supervisor] vLLM engine wedged (requests "
                                "pending, zero token progress); killing process "
                                "tree to force restart")
                            _sigkill_tree(child)
                            ready, fails, stall, last_tokens = False, 0, 0, None
                            continue
                    else:
                        stall = 0
                    last_tokens = tokens

    if health_check is not None:
        threading.Thread(target=_health_loop, name="vllm-health", daemon=True).start()

    crash_times: deque[float] = deque()
    unready_starts = 0
    attempt = 0
    while True:
        attempt += 1
        if attempt > 1 and mem_frac is not None:
            # Post-crash respawn: wait out the previous engine's VRAM drain so
            # the relaunch doesn't fail vLLM's startup free-memory check and
            # burn a never-ready strike on a GPU that was seconds from ready.
            _wait_gpu_drain(env, mem_frac, log,
                            abort=lambda: state["shutting_down"])
            if state["shutting_down"]:
                log("[supervisor] shutdown requested during drain wait; "
                    "not respawning")
                return 0
        log(f"[supervisor] starting child (attempt {attempt})")
        try:
            state["child"] = subprocess.Popen(cmd, env=env, start_new_session=True)
        except Exception as exc:
            log(f"[supervisor] FATAL: failed to spawn child: {exc!r}")
            return 1
        rc = state["child"].wait()

        if state["shutting_down"]:
            log(f"[supervisor] child exited rc={rc} after shutdown signal")
            return 0

        now = time.monotonic()
        crash_times.append(now)
        # Evict crashes outside the rolling window.
        while crash_times and crash_times[0] < now - restart_window_sec:
            crash_times.popleft()
        recent = len(crash_times)
        log(
            f"[supervisor] child exited rc={rc}; "
            f"{recent}/{max_restarts} crashes in last {restart_window_sec:.0f}s"
        )
        if recent >= max_restarts:
            log("[supervisor] FATAL: crash cap reached; giving up")
            return 1
        if health_check is not None:
            if state.get("ready_child") is state["child"]:
                unready_starts = 0
            else:
                unready_starts += 1
                log(f"[supervisor] child died without ever serving a probe "
                    f"({unready_starts}/{max_unready_starts} consecutive)")
                if unready_starts >= max_unready_starts:
                    log("[supervisor] FATAL: relaunching is not recovering "
                        f"({unready_starts} consecutive never-ready starts); "
                        "giving up")
                    return 1


def build_cli_args(server_config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Translate a vllm_server config dict to (cli_args, env_vars)."""
    cli_args: list[str] = []
    env_vars: dict[str, str] = {}
    for key, value in server_config.items():
        if key in _OUR_FIELDS:
            continue
        if value is None or value == "":
            continue
        if key == "extra_args":
            if isinstance(value, list):
                cli_args.extend(str(v) for v in value)
            continue
        if key in _ENV_BOOL_FIELDS:
            env_vars[_ENV_BOOL_FIELDS[key]] = "1" if bool(value) else "0"
            continue
        if key in _ENV_VAR_FIELDS:
            if value:
                env_vars[_ENV_VAR_FIELDS[key]] = "1"
            continue
        arg_name = _FIELD_RENAMES.get(key, key).replace("_", "-")
        if key in _BOOLEAN_FLAGS:
            if value:
                cli_args.append(f"--{arg_name}")
            continue
        if isinstance(value, bool):
            cli_args.extend([f"--{arg_name}", str(value).lower()])
        elif isinstance(value, dict):
            cli_args.extend([f"--{arg_name}", json.dumps(value)])
        else:
            cli_args.extend([f"--{arg_name}", str(value)])
    return cli_args, env_vars


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Launch vLLM API server from a yaml config. "
                    "Unknown args are passed through to vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", required=True, help="Path to yaml config containing a vllm_server block")
    p.add_argument("--model", default=None,
                   help="Model path or HF id (overrides yaml's vllm_server.model_path)")
    p.add_argument("--served-model-name", default=None,
                   help="Name exposed via the OpenAI API (e.g. qwen3_5-4b)")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the server command without starting vLLM or writing files")
    return p.parse_known_args()


def main() -> None:
    # Line-buffer stdout so the supervisor's own [supervisor] lines flush to
    # $VLLM_LOG immediately. Redirected to a file, stdout is block-buffered by
    # default, so the healthcheck/kill messages never reached disk until 8 KB
    # filled or the process exited — on 2026-07-07 that hid whether the wedge
    # detector fired at all. vLLM's own logger flushes independently.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    args, passthrough = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    vs = cfg.get("vllm_server") or {}

    model = args.model or vs.get("model_path")
    if not model:
        raise SystemExit(
            "ERROR: model path required — pass --model or set vllm_server.model_path in the yaml"
        )

    tp = int(vs.get("tensor_parallel_size", 1))
    pp = int(vs.get("pipeline_parallel_size", 1))
    dp = int(vs.get("data_parallel_size", 1))

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(tp),
    ]
    if pp > 1:
        cmd += ["--pipeline-parallel-size", str(pp)]
    if dp > 1:
        cmd += ["--data-parallel-size", str(dp)]
    if args.served_model_name:
        # Whitespace-separated => multiple aliases for ONE engine. Lets a single
        # serve answer to both the base and feedback model names when they are
        # the SAME checkpoint, instead of paying a second serve for a duplicate.
        cmd += ["--served-model-name", *args.served_model_name.split()]

    extra_cli, env_vars = build_cli_args(vs)
    cmd.extend(extra_cli)
    cmd.extend(passthrough)

    env = os.environ.copy()
    env.update(env_vars)
    # Local server extensions must be importable in both the API process and
    # spawned engine workers, independently of the launcher's working directory.
    serve_modules = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (serve_modules, env.get("PYTHONPATH", "")) if p
    )
    env.setdefault("VLLM_MODEL_PATH", model)
    # vLLM's default compile cache lives under ~/.cache on NFS. Concurrent
    # jobs compiling the same model have raced while creating AOTAutograd
    # entries there (FileExistsError), so give each job/server a node-local
    # cache. Keep an explicit caller override intact.
    slurm_job_id = env.get("SLURM_JOB_ID")
    cache_scope = (
        slurm_job_id
        if slurm_job_id and slurm_job_id.isdecimal()
        else str(os.getpid())
    )
    env.setdefault(
        "VLLM_CACHE_ROOT",
        f"/tmp/vllm_uid{os.getuid()}_job{cache_scope}_port{args.port}",
    )

    # First alias is canonical: the health probe generates against one name.
    served = (
        args.served_model_name.split()[0] if args.served_model_name else model
    )
    health_interval = float(vs.get("healthcheck_interval") or HEALTH_INTERVAL_SEC)
    # gpu_memory_utilization from the yaml (vLLM's 0.9 default when absent)
    # feeds the respawn free-VRAM gate. A CLI passthrough override isn't
    # parsed — worst case the gate waits to a slightly wrong bar and then
    # launches anyway.
    mem_frac = float(vs.get("gpu_memory_utilization") or 0.9)

    print("Launching vLLM (supervised):")
    print("  " + " ".join(shlex.quote(c) for c in cmd))
    print(f"  compile cache: {env['VLLM_CACHE_ROOT']}")
    if env_vars:
        print(f"  env additions: {env_vars}")
    print(f"  healthcheck: 1-token gen every {health_interval:.0f}s, "
          f"restart after {HEALTH_MAX_FAILS} consecutive failures")
    if args.dry_run:
        return
    # Persist each watchdog scrape's token counters for compute accounting.
    # Same naming convention as gpu_task_sampler.sh so flops_report.py can
    # glob both by job id. VLLM_METRICS_DIR= overrides; empty disables.
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metrics_dir = os.environ.get(
        "VLLM_METRICS_DIR", os.path.join(repo, "runs", "vllm_metrics")
    )
    metrics_csv = None
    if metrics_dir:
        try:
            os.makedirs(metrics_dir, exist_ok=True)
            job = os.environ.get("SLURM_JOB_ID") or f"manual{os.getpid()}"
            host = os.uname().nodename.split(".")[0]
            metrics_csv = os.path.join(
                metrics_dir, f"{job}_{host}_port{args.port}_metrics.csv"
            )
        except OSError:
            metrics_csv = None

    print(f"  progress watchdog: /metrics every {health_interval:.0f}s, restart "
          f"after {PROGRESS_STALL_MAX} intervals with requests pending and no "
          "token progress")
    print(f"  token-counter log: {metrics_csv or 'DISABLED'}")
    print(f"  respawn gate: wait <={DRAIN_TIMEOUT_SEC:.0f}s for free VRAM >= "
          f"{mem_frac:.2f}*total + {DRAIN_MARGIN_MIB / 1024:.0f} GiB before any relaunch")
    sys.stdout.flush()

    sys.exit(supervise(
        cmd, env=env,
        mem_frac=mem_frac,
        health_check=lambda: _probe_generation(args.port, served, HEALTH_TIMEOUT_SEC),
        health_interval=health_interval,
        progress_probe=lambda: _read_progress(
            args.port, METRICS_TIMEOUT_SEC, csv_path=metrics_csv
        ),
    ))


if __name__ == "__main__":
    main()
