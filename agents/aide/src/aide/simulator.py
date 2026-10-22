"""LLM-backed simulator interpreter for AIDE.

Drop-in replacement for `Interpreter` that asks a fine-tuned LLM (served via
vLLM) to predict the stdout/stderr a piece of Python code would produce. Used
by `ParallelInterpreter` as the slow-code fallback in a race against the real
subprocess. Does not spawn a child process; does not write files.

The vLLM call is issued with `stream=True` so it can be cancelled mid-flight
when the real interpreter wins the race (see `ParallelInterpreter`). The
client is configured with `max_retries=0` — a cancellation must not trigger
a retry loop.
"""

import builtins
import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path

import httpx
import humanize
import openai

from . import metric_posterior as metric_sampling
from . import simulator_scoring
from .interpreter import ExecutionResult

logger = logging.getLogger("aide")


# OpenAI chat.completions.create top-level params. Everything else from
# AIDE_SIMULATOR_SAMPLING is routed via `extra_body` so vLLM honors top_k,
# min_p, repetition_penalty, etc. Mirrors backend_openai._OPENAI_TOP_LEVEL.
_OPENAI_TOP_LEVEL = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "stop",
    "logit_bias",
    "response_format",
    "logprobs",
    "top_logprobs",
    "stream",
    "user",
}

# Deterministic defaults: greedy decoding with no nucleus/top-k filtering.
# `temperature=0` makes vLLM ignore top_p/top_k anyway, but we pin them
# explicitly so callers can read the call site and know what's going on.
# `seed=0` is belt-and-suspenders — irrelevant under greedy, load-bearing
# if a caller overrides temperature.
_DETERMINISTIC_SIM_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "seed": 0,
}


def _load_sim_sampling_overrides() -> dict:
    """Parse `AIDE_SIMULATOR_SAMPLING` (JSON object). Empty → no overrides."""
    raw = os.environ.get("AIDE_SIMULATOR_SAMPLING", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"AIDE_SIMULATOR_SAMPLING is not valid JSON, ignoring: {e}")
        return {}
    if not isinstance(parsed, dict):
        logger.warning(
            f"AIDE_SIMULATOR_SAMPLING must be a JSON object, got {type(parsed).__name__}"
        )
        return {}
    return parsed


def _build_sim_create_kwargs(max_tokens: int) -> tuple[dict, dict]:
    """Build (top_level_kwargs, extra_body) for the simulator's create() call.

    Starts from `_DETERMINISTIC_SIM_SAMPLING`, then layers the user-supplied
    `AIDE_SIMULATOR_SAMPLING` JSON on top. Splits known OpenAI top-level keys
    from vLLM-extension keys (top_k, min_p, repetition_penalty, …).
    """
    merged = dict(_DETERMINISTIC_SIM_SAMPLING)
    merged.update(_load_sim_sampling_overrides())
    # max_tokens from the SimulatorInterpreter ctor wins unless the env
    # explicitly sets it — gives a sane upper bound without surprise OOMs.
    merged.setdefault("max_tokens", max_tokens)

    top_level: dict = {}
    extra_body: dict = {
        # Skip qwen3 thinking even if the server didn't bake this in via
        # --default-chat-template-kwargs. SFT corpus has no <think> in the
        # target, so enabling it puts inference out of distribution.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for k, v in merged.items():
        if k in _OPENAI_TOP_LEVEL:
            top_level[k] = v
        else:
            extra_body[k] = v
    return top_level, extra_body


# Canonical built-in exception names, used to recognize a bare `ClassName`
# traceback summary line (a `raise X` with no message, hence no colon).
_BUILTIN_EXC_NAMES = frozenset(
    name
    for name, obj in vars(builtins).items()
    if isinstance(obj, type) and issubclass(obj, BaseException)
)


def _parse_exc_type_from_term_out(term_out: str) -> str | None:
    """Extract an exception class name from a Python traceback in `term_out`.

    Matches `scripts/datagen/extract_aide_logs.py:_parse_exc_type` (the
    gold-standard copy); kept inline so this module stays import-light.
    `automind/interpreter.py` keeps its own, now-divergent, simpler copy.

    Per CPython's traceback layout the exception summary line is UNINDENTED
    while frame info is indented; we use that to find the final exception and
    to accept a bare `ClassName` (no `: message`) such as `KeyboardInterrupt`.
    Dotted names are returned as their bare leaf ('lightgbm.basic.LightGBMError'
    -> 'LightGBMError'), matching the canonical spelling the corpus trains on.
    """
    if not term_out:
        return None
    if "TimeoutError: Execution exceeded the time limit of" in term_out:
        return "TimeoutError"
    if "Traceback (most recent call last):" not in term_out:
        return None
    lines = term_out.rstrip().split("\n")
    while lines and (
        not lines[-1].strip()
        or lines[-1].startswith("Execution time:")
        or "TimeoutError: Execution exceeded" in lines[-1]
    ):
        lines.pop()
    for line in reversed(lines):
        if not line or line[0].isspace():
            continue  # frame info, indented continuation, or blank line
        head, sep, _ = line.partition(":")
        head = head.strip()
        if sep:
            if head and all(p.isidentifier() for p in head.split(".")):
                return head.rsplit(".", 1)[-1]
        elif head in _BUILTIN_EXC_NAMES or (
            "." in head and all(p.isidentifier() for p in head.split("."))
        ):
            return head.rsplit(".", 1)[-1]
    return None


# ExecutionResult.exc_type values produced by SimulatorInterpreter.run when the
# simulator itself failed (backend error, cancellation, empty response) — NOT
# when the code was predicted to crash.
_SIM_INTERNAL_FAILURES = {"SimulatorError", "SimulatorEmpty", "SimulatorCancelled"}

# Defense in depth against parser garbage: _parse_exc_type_from_term_out (above)
# can occasionally pick non-exception identifiers (e.g. "model.safetensors") out
# of a stripped TimeoutError tail. Real exception class names end in one of these
# suffixes, equal the lowercase "error" some C extensions expose (cv2.error,
# re.error, struct.error, _csv.error), or match the well-known exact set;
# everything else is "not a crash".
_EXC_SUFFIXES = ("Error", "Exception", "Warning", "Interrupt", "Exit")
_EXC_EXACT = {
    "StopIteration",
    "KeyboardInterrupt",
    "SystemExit",
    "GeneratorExit",
    "StopAsyncIteration",
    "ExceptionGroup",
    "BaseExceptionGroup",
}


def is_predicted_crash(sim_res: ExecutionResult) -> bool:
    """True iff the simulator predicted the code would raise a real exception.

    False for simulator-internal failures (the sentinels above) and for legacy
    parser garbage that doesn't look like a real exception class name. Used by the BoN
    / sim-debug selection hooks in agent.py, where being lenient about parse
    noise is correct (real execution is the ground truth).
    """
    exc = sim_res.exc_type
    if exc is None or exc in _SIM_INTERNAL_FAILURES:
        return False
    # Structured parsers carry an explicit outcome; custom classes such as
    # ArrowInvalid are failures even though their names do not end in Error.
    if sim_res.predicted_submission_exists is not None:
        return True
    leaf = exc.rsplit(".", 1)[-1]
    return leaf in _EXC_EXACT or leaf == "error" or leaf.endswith(_EXC_SUFFIXES)


def is_predicted_candidate_usable(sim_res: ExecutionResult) -> bool:
    """Whether a simulator prediction may enter the BoN accepted set.

    Require explicit predictions of clean execution and a produced submission.
    Missing or malformed evidence, including legacy parser garbage, does not
    establish usability even when it is not a recognized code exception.
    """
    submission = getattr(sim_res, "predicted_submission_exists", None)
    return sim_res.exc_type is None and submission == "true"


def is_bad_prediction(sim_res: ExecutionResult) -> bool:
    """True iff a sim result should make its node buggy: a predicted real crash
    OR a simulator-internal failure. Parser garbage is neither (treated as a
    benign/successful prediction).

    This is the right test wherever a *sim* result drives a buggy/keep decision
    — parse_exec_result's buggy flag and the race's on-timeout CSV reconcile —
    as opposed to is_predicted_crash, which is only about a predicted crash and
    deliberately distinguishes internal failures from predicted code crashes.
    """
    return is_predicted_crash(sim_res) or sim_res.exc_type in _SIM_INTERNAL_FAILURES


def is_sim_internal_failure(sim_res: ExecutionResult) -> bool:
    """Identify unavailable predictions, including backend errors and cancellation.

    ParallelInterpreter then waits for real execution. A failed prediction
    request provides no evidence that the generated program itself is buggy.
    """
    return sim_res.exc_type in _SIM_INTERNAL_FAILURES


# Byte-identical to build_llamafactory_dataset.py's
# ENV_V1_REORDER_EXCFIRST_SYSTEM_PROMPT (env_v1_reorder_excfirst).
_ENV_V1_REORDER_EXCFIRST_SYSTEM_PROMPT = """Predict the outcome and terminal output (stdout+stderr) of a Python ML script.

Input, in order:
- <code>, the script, run as runfile.py.
- <environment>, the importable packages. Authoritative: a library not listed is not importable, and the listed version is the one whose API applies.
- <data_overview>, truth for files/columns under ./input/.
- <runtime_budget>, the wall-clock limit. The script is killed when it is exceeded.

Output, in order and nothing else:
- <exc_type>…</exc_type>, the exception class the script raises, or None if it runs clean; TimeoutError if its wall-clock time would exceed <runtime_budget>.
- <term_out>…</term_out>, the script's stdout+stderr. Write it before the two fields below — they are read off it.
- <val_metric>…</val_metric>, the validation score reported in the term_out, or None if it reports none, the validation is invalid (e.g. scoring training data or computing the wrong metric), or the script raises.
- <submission_exists>true|false</submission_exists>, whether ./submission/submission.csv exists after the run.

Missing files or columns → predict the right Python exception, don't invent data.
A script can finish without raising and still write no submission — decide <submission_exists> from what the code actually writes to ./submission/, not from whether it crashed.
"""


_PLAIN_PROMPT_FORMAT = "env_v1_reorder_excfirst"
_TOKEN_PROMPT_FORMAT = "env_v1_reorder_excfirst_token"
# Keep identical to scripts/datagen/exception_tokens.py:token_system_prompt.
_ENV_V1_REORDER_EXCFIRST_TOKEN_SYSTEM_PROMPT = _ENV_V1_REORDER_EXCFIRST_SYSTEM_PROMPT.replace(
    "- <exc_type>…</exc_type>, the exception class the script raises, or None if it runs clean; TimeoutError if its wall-clock time would exceed <runtime_budget>.",
    "- <exc_type>…</exc_type>, one EXC_<ClassName> token: EXC_None if it runs clean, otherwise its exception class prefixed with EXC_ (e.g. EXC_ValueError); EXC_TimeoutError if its wall-clock time would exceed <runtime_budget>.",
)
_SYSTEM_PROMPTS = {
    _PLAIN_PROMPT_FORMAT: _ENV_V1_REORDER_EXCFIRST_SYSTEM_PROMPT,
    _TOKEN_PROMPT_FORMAT: _ENV_V1_REORDER_EXCFIRST_TOKEN_SYSTEM_PROMPT,
}


def _resolve_system_prompt() -> tuple[str, str]:
    """Use EXC_* targets by default; retain explicit plain-label world_9_3 runs."""
    name = os.environ.get("AIDE_SIMULATOR_PROMPT_FORMAT", "").strip() or _TOKEN_PROMPT_FORMAT
    if name not in _SYSTEM_PROMPTS:
        raise ValueError(
            f"Unsupported AIDE_SIMULATOR_PROMPT_FORMAT={name!r}; "
            f"expected one of {tuple(_SYSTEM_PROMPTS)}"
        )
    return name, _SYSTEM_PROMPTS[name]


_DATA_OVERVIEW_FENCE_OPEN_RE = re.compile(r"^```[A-Za-z0-9_+\-]*\s*$")


def _normalize_to_aide_data_overview(text: str) -> str:
    """Normalize AutoMind headers and outer fences to match the training format."""
    if not text:
        return ""
    s = text
    s = s.replace("Here is the directory structure:\n\n", "", 1)
    s = s.replace("\n\nHere is the data analysis:\n\n", "\n\n", 1)
    # Match scripts/datagen/extract_aide_logs.py:_strip_code_fence.
    # Internal fences, such as the tree's closer before CSV previews, are content.
    lines = s.strip().split("\n")
    while lines and (not lines[0].strip() or _DATA_OVERVIEW_FENCE_OPEN_RE.match(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or lines[-1].strip() == "```"):
        lines.pop()
    return "\n".join(lines)


# Avoid duplicating a harness timeout footer already emitted by the model.
_TIMEOUT_FOOTER_PREFIX = "TimeoutError: Execution exceeded the time limit of"


def _ends_with_timeout_footer(lines):
    """True iff the predicted log already ends in the harness timeout footer.

    Membership, not startswith: the harness itself sometimes writes the footer
    with no separating newline, glueing it onto the last content line (4 of
    49,791 rows in the D-tail corpus, e.g. `...best gain: -infTimeoutError:
    Execution exceeded...`). A startswith test misses those and deploy would
    render a second footer. The phrase is specific enough that a log containing
    it anywhere in its final line has one.
    """
    for ln in reversed(lines):
        if ln.strip():
            return _TIMEOUT_FOOTER_PREFIX in ln
    return False


_ENV_RESULT_RE = re.compile(
    r"\A\s*<exc_type>\s*(?P<exc>[A-Za-z_][\w.]*)\s*</exc_type>\s*"
    r"<term_out>(?P<output>.*?)</term_out>\s*"
    r"<val_metric>\s*(?P<metric>.*?)\s*</val_metric>\s*"
    r"<submission_exists>\s*(?P<submission>true|false|unknown)\s*"
    # vLLM omits the stop string from returned content.
    r"(?:</submission_exists>)?\s*\Z",
    re.S,
)


def _parse_env_result(text: str, *, prompt_format: str | None = None) -> dict | None:
    """Decode outcome labels to ExecutionResult's canonical Python class names."""
    match = _ENV_RESULT_RE.fullmatch(text or "")
    if match is None:
        return None
    prompt_format = prompt_format or _resolve_system_prompt()[0]
    exc_type = match.group("exc")
    if prompt_format == _TOKEN_PROMPT_FORMAT:
        if not exc_type.startswith("EXC_"):
            return None
        exc_type = exc_type.removeprefix("EXC_")
        if exc_type not in _EXC_TOKEN_SUPPORT:
            return None
    elif prompt_format != _PLAIN_PROMPT_FORMAT:
        raise ValueError(f"unsupported simulator prompt format: {prompt_format}")
    elif exc_type.startswith("EXC_"):
        return None  # A mismatched checkpoint/prompt must abstain, not mark clean as buggy.
    metric = match.group("metric").strip()
    if metric != "None":
        try:
            if exc_type != "None" or not math.isfinite(float(metric)):
                return None
        except ValueError:
            return None
    return {
        "exc_type": None if exc_type == "None" else exc_type,
        "submission_exists": match.group("submission"),
        "reported_metric": metric,
        # These targets contain raw terminal text, not XML-escaped text.
        "observed_output": match.group("output").removeprefix("\n").removesuffix("\n"),
    }


# Retry incomplete structured predictions before abstaining.
_MALFORMED_RETRIES = 3


_sim_client = openai.OpenAI(
    base_url=os.environ.get("AIDE_SIMULATOR_BASE_URL", "http://localhost:8001/v1"),
    api_key=os.environ.get("AIDE_SIMULATOR_API_KEY", "dummy"),
    max_retries=0,
    timeout=300,
    # Echo comparisons and long generations can outlive vLLM's idle HTTP
    # timeout. Avoid reusing a stale socket for a subsequent scoring branch;
    # keep retries disabled so cancellation never resubmits work.
    http_client=httpx.Client(
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=0),
        follow_redirects=True,
    ),
)

# Flipped to False the first time the serving stack rejects `stream_options`
# (pre-include_usage vLLM or an old SDK) so every later call skips the retry.
_STREAM_USAGE_OK = True


# Legacy conditional posterior over this fixed set of complete exception strings.
# Classes outside the support are not assigned probability. This is not a
# calibrated probability that the candidate produces a valid submission.
_EXC_SUPPORT = (
    "None",
    "ValueError",
    "TypeError",
    "NameError",
    "SyntaxError",
    "AttributeError",
    "KeyError",
    "TimeoutError",
    "RuntimeError",
    "PicklingError",
    "IndexError",
    "ImportError",
    "FileNotFoundError",
    "ParserError",
    "ModuleNotFoundError",
    "UnboundLocalError",
    "LightGBMError",
    "SystemExit",
    "AcceleratorError",
    "NotFittedError",
    "IndentationError",
    "AssertionError",
    "OutOfMemoryError",
    "XGBoostError",
)


# Keep ordered support identical to scripts/datagen/exception_tokens.py.
# The agent snapshot cannot import the host's datagen scripts. Only class names
# are shared here: actual token IDs are validated through the served tokenizer
# on every request, so a reload cannot leave a stale checkpoint-specific map.
_EXC_TOKEN_SUPPORT = (
    "None",
    "AcceleratorError",
    "ArrowInvalid",
    "ArrowTypeError",
    "AssertionError",
    "AttributeError",
    "AxisError",
    "BadZipFile",
    "BdbQuit",
    "BracketError",
    "CalledProcessError",
    "CatBoostError",
    "CompressionError",
    "DTypePromotionError",
    "DateParseError",
    "DeferredCudaCallError",
    "EOFError",
    "EmptyDataError",
    "Error",
    "Exception",
    "FileExistsError",
    "FileNotFoundError",
    "FloatingPointError",
    "HTTPError",
    "ImportError",
    "IndentationError",
    "IndexError",
    "IndexingError",
    "IntCastingNaNError",
    "InvalidIndexError",
    "InvalidParameterError",
    "InvalidURL",
    "InvalidVersion",
    "IsADirectoryError",
    "JSONDecodeError",
    "KeyError",
    "LightGBMError",
    "LinAlgError",
    "LookupError",
    "MemoryError",
    "MergeError",
    "MissingCorpusError",
    "ModuleNotFoundError",
    "NameError",
    "NotADirectoryError",
    "NotFittedError",
    "NotImplementedError",
    "OSError",
    "OptionError",
    "OutOfBoundsDatetime",
    "OutOfMemoryError",
    "OverflowError",
    "ParserError",
    "PicklingError",
    "ReadError",
    "RecursionError",
    "RepositoryNotFoundError",
    "RuntimeError",
    "SameFileError",
    "SpecificationError",
    "StopIteration",
    "SyntaxError",
    "SystemExit",
    "TabError",
    "TerminatedWorkerError",
    "TimeoutError",
    "TorchRuntimeError",
    "TqdmKeyError",
    "TypeError",
    "UFuncTypeError",
    "URLError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnidentifiedImageError",
    "UnpicklingError",
    "UnsupportedOperation",
    "ValueError",
    "VersionError",
    "XGBoostError",
    "ZeroDivisionError",
    "error",
)


def _exc_scoring_method(prompt_format: str | None = None, method: str | None = None) -> str:
    prompt_format = prompt_format or _resolve_system_prompt()[0]
    method = method or os.environ.get("AIDE_EXC_SCORING", "").strip() or "auto"
    if prompt_format not in _SYSTEM_PROMPTS:
        raise ValueError(f"unsupported simulator prompt format: {prompt_format}")
    token_format = prompt_format == _TOKEN_PROMPT_FORMAT
    if method == "auto":
        return "single_token" if token_format else "forced_suffix"
    allowed = {"single_token"} if token_format else {"forced_suffix", "echo"}
    if method not in allowed:
        raise ValueError(
            f"AIDE_EXC_SCORING={method!r} is incompatible with {prompt_format}; "
            f"use auto or one of {sorted(allowed)}"
        )
    return method


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


# A registered special token is a hard tokenizer boundary. Repeating this
# assistant header lets one /tokenize request encode the long input once and
# all class suffixes, without a local tokenizer or cached checkpoint token map.
_SIM_ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _pack_exc_prompts(prefix: str, support=_EXC_SUPPORT, label_prefix: str = "") -> str:
    if not prefix.endswith(_SIM_ASSISTANT_PREFIX):
        raise ValueError("exception scorer requires the Qwen no-thinking chat template")
    return prefix[: -len(_SIM_ASSISTANT_PREFIX)] + "".join(
        f"{_SIM_ASSISTANT_PREFIX}<exc_type>{label_prefix}{exc}</exc_type>" for exc in support
    )


def _unpack_exc_tokens(
    response: dict, support=_EXC_SUPPORT, label_prefix: str = ""
) -> tuple[list[int], list[list[int]]]:
    ids, token_strs = response["tokens"], response["token_strs"]
    if not ids or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("invalid token IDs from /tokenize")
    if token_strs is None or len(token_strs) != len(ids):
        raise ValueError("/tokenize must return token_strs")
    # The last len(support) <|im_start|> tokens are our assistant headers. The input may
    # itself contain this special token, so earlier occurrences belong to it.
    starts = [i for i, token in enumerate(token_strs) if token == "<|im_start|>"]
    if len(starts) < len(support):
        raise ValueError("served tokenizer must register <|im_start|> as a special token")
    starts = starts[-len(support) :]
    sequences = [ids[: starts[0]] + ids[a:b] for a, b in zip(starts, [*starts[1:], len(ids)])]
    common = 0
    for column in zip(*sequences):
        if len(set(column)) != 1:
            break
        common += 1
    if common == 0 or common == min(map(len, sequences)):
        raise ValueError("exception strings must have a shared prefix and distinct suffixes")
    if label_prefix:
        for exc, start in zip(support, starts):
            if token_strs[start + common - starts[0]] != f"{label_prefix}{exc}":
                raise ValueError(f"served tokenizer must encode {label_prefix}{exc} as one token")
        if len({seq[common] for seq in sequences}) != len(support):
            raise ValueError("exception labels must have distinct token IDs")
    # The packed transport is CPU-only. For EXC_* scoring only the common
    # prompt plus one outcome token consumes context, not the closing tag.
    needed = common + 1 if label_prefix else max(map(len, sequences))
    if needed > response["max_model_len"]:
        raise ValueError("exception scoring prompt exceeds the server context limit")
    return sequences[0][:common], [seq[common:] for seq in sequences]


def _tokenize_exc_prompts(
    prefix: str, model: str, *, support=_EXC_SUPPORT, label_prefix: str = ""
) -> tuple[list[int], list[list[int]]]:
    """Tokenize with the shared client and the deployment protocol's vocabulary."""
    return simulator_scoring.tokenize_exc_prompts(
        _sim_client,
        prefix,
        model,
        support=support,
        label_prefix=label_prefix,
        pack=_pack_exc_prompts,
        unpack=_unpack_exc_tokens,
    )


def _exc_logps_single_token(prefix: str, model: str) -> list[float]:
    """Score the registered outcome tokens at one next-token position."""
    return simulator_scoring.single_token_logps(
        _sim_client,
        prefix,
        model,
        tokenize=_tokenize_exc_prompts,
        support=_EXC_TOKEN_SUPPORT,
    )


def _exc_logps_forced_suffix(prefix: str, model: str) -> list[float]:
    """Score legacy complete labels using the server's forced-token processor."""
    return simulator_scoring.forced_suffix_logps(
        _sim_client,
        prefix,
        model,
        tokenize=_tokenize_exc_prompts,
        workers=min(len(_EXC_SUPPORT), _positive_env_int("AIDE_EXC_SCORE_WORKERS", 8)),
    )


def _exc_logps_echo(prefix: str, model: str) -> list[float]:
    """Retain the explicit full-prompt reference scorer for legacy experiments."""
    return simulator_scoring.echo_logps(
        _sim_client,
        prefix,
        model,
        support=_EXC_SUPPORT,
        chunk=_positive_env_int("AIDE_PCLEAN_CHUNK", 8),
    )


def _render_sim_prefix(system_prompt: str, user_msg: str) -> str:
    """Chat prompt for /v1/completions, byte-identical to the chat endpoint's.

    Rendered here rather than through AutoTokenizer.apply_chat_template because
    the checkpoint stores its template in chat_template.jinja (transformers>=5
    layout) while this container pins transformers 4.44.2, which looks for
    `chat_template` inside tokenizer_config.json and would silently fall back to
    a different template.

    The `.strip()` calls are load-bearing: the real template pipes every message
    content through `|trim` (chat_template.jinja:55,63,71,82), so omitting them
    leaves the system prompt's trailing newline in and puts the call off
    distribution with no error. Verified byte-exact against apply_chat_template
    by scripts/eval/wm_entropy/test_render_prefix.py on the world_9_3 holdout.
    """
    return (
        "<|im_start|>system\n" + system_prompt.strip() + "<|im_end|>\n"
        "<|im_start|>user\n" + user_msg.strip() + "<|im_end|>\n" + _SIM_ASSISTANT_PREFIX
    )


def predict_exc_posterior(
    data_overview: str,
    code: str,
    model: str,
    system_prompt: str,
    *,
    user_msg: str,
    method: str | None = None,
) -> dict | None:
    """Return the outcome posterior for the same input used by run().

    EXC_* checkpoints score all 82 outcome tokens at one output position.
    Explicit plain-label runs retain their 24 complete-string scorer. Both
    posteriors are conditional on their enumerated support. Scoring failures
    return None so UCB can record missing confidence and use metric-greedy;
    they never trigger a different scoring method automatically.
    """
    prompt_format = next(
        (name for name, prompt in _SYSTEM_PROMPTS.items() if prompt == system_prompt), None
    )
    if prompt_format is None:
        raise ValueError("exception scoring requires a supported world-model system prompt")
    selected_method = _exc_scoring_method(prompt_format, method)
    support = _EXC_TOKEN_SUPPORT if selected_method == "single_token" else _EXC_SUPPORT
    prefix = _render_sim_prefix(system_prompt, user_msg)
    start = time.monotonic()
    try:
        scorer = {
            "single_token": _exc_logps_single_token,
            "forced_suffix": _exc_logps_forced_suffix,
            "echo": _exc_logps_echo,
        }[selected_method]
        logps = scorer(prefix, model)
        if len(logps) != len(support) or any(not math.isfinite(v) for v in logps):
            raise ValueError("missing or nonfinite exception log probabilities")
        top = max(logps)
        weights = [math.exp(v - top) for v in logps]
        total = math.fsum(weights)
        scores = _finalize_exc_posterior([v / total for v in weights], support)
        if selected_method == "single_token":
            log_mass = top + math.log(total)
            # Allow small roundoff above one, but reject processed/forced or
            # otherwise invalid probabilities that no longer form raw mass.
            if log_mass > 1e-5:
                raise ValueError("EXC_* probability mass exceeds one; use raw_logprobs")
            scores["raw_label_mass"] = math.exp(log_mass)
            scores["raw_other_mass"] = max(0.0, 1.0 - scores["raw_label_mass"])
    except Exception as e:
        logger.warning(
            f"predict_exc_posterior ({selected_method}) failed ({type(e).__name__}: {e}); "
            "confidence is unavailable for this candidate"
        )
        return None
    scores["scoring_method"] = selected_method
    scores["scoring_seconds"] = time.monotonic() - start
    return scores


def _finalize_exc_posterior(p: list[float], support=_EXC_SUPPORT) -> dict:
    """Keep the same UCB/judge fields across the 82-token and legacy scorers."""
    order = sorted(range(len(p)), key=lambda j: -p[j])
    i_clean = support.index("None")
    # Leading failure hypothesis: top NON-clean class + its mass. Not
    # argmax_class, which is "None" whenever clean leads even narrowly --
    # exactly the middling-p_clean rows where naming the runner-up failure
    # mode (typically the TimeoutError hesitation) is informative.
    i_exc = next(j for j in order if j != i_clean)
    return {
        "p_clean": p[i_clean],
        "p_exception": math.fsum(q for i, q in enumerate(p) if i != i_clean),
        "probabilities": dict(zip(support, p)),
        "max_p": p[order[0]],
        "margin": p[order[0]] - p[order[1]],
        "entropy": -sum(q * math.log(q) for q in p if q > 0),
        "support_size": len(p),  # entropy normalization for UCB selection
        "argmax_class": support[order[0]],
        "top_exc": support[i_exc],
        "p_top_exc": p[i_exc],
    }


# Convenience scalar returned by exc_scores()/p_clean(). Selection policies can
# still inspect the complete posterior; max_p need not describe a clean outcome.
_PCLEAN_SCORES = ("p_clean", "max_p")


def _resolve_pclean_score() -> str:
    """AIDE_PCLEAN_SCORE (unset → "p_clean"). Unknown values raise -- a typo'd
    knob must fail the launch, not silently score with the default."""
    name = os.environ.get("AIDE_PCLEAN_SCORE", "").strip() or "p_clean"
    if name not in _PCLEAN_SCORES:
        raise ValueError(f"Unknown AIDE_PCLEAN_SCORE {name!r}; expected one of {_PCLEAN_SCORES}")
    return name


def _streaming_query(
    messages: list[dict],
    model: str,
    max_tokens: int,
    cancel_event: threading.Event,
) -> str | None:
    """Streaming call to the simulator. Returns None if `cancel_event` fires
    or the backend returns nothing.

    The full response, including validation metric and submission flag, is
    needed even when the predicted exception is a crash.
    """
    top_level, extra_body = _build_sim_create_kwargs(max_tokens)
    # A hedged simulator Future may sit queued behind the previous request. If
    # the real interpreter wins meanwhile, never send this stale request.
    if cancel_event.is_set():
        return None
    global _STREAM_USAGE_OK
    t0 = time.time()
    create_kwargs = dict(
        model=model,
        messages=messages,
        stream=True,
        stop=["</submission_exists>"],
        extra_body=extra_body,
        **top_level,
    )
    if _STREAM_USAGE_OK:
        try:
            stream = _sim_client.chat.completions.create(
                stream_options={"include_usage": True}, **create_kwargs
            )
        except (TypeError, openai.BadRequestError):
            # Older SDK/server that rejects stream_options: usage accounting is
            # a best-effort cross-check (the :8001 /metrics counters are the
            # authoritative source), never worth failing a sim call over.
            _STREAM_USAGE_OK = False
            stream = _sim_client.chat.completions.create(**create_kwargs)
    else:
        stream = _sim_client.chat.completions.create(**create_kwargs)
    parts: list[str] = []
    usage = None
    try:
        for chunk in stream:
            if cancel_event.is_set():
                try:
                    stream.response.close()
                except Exception:
                    pass
                return None
            # include_usage delivers a final choices-less chunk carrying token
            # counts; an early break/cancel below simply forgoes it.
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # qwen3 reasoning-parser may emit `delta.reasoning` before
            # `delta.content`; only the content goes into the term_out target.
            content = getattr(delta, "content", None)
            if content:
                parts.append(content)
    finally:
        try:
            stream.response.close()
        except Exception:
            pass
    if usage is not None:
        # Same field order as backend_openai's "openai call:" line so one
        # parser handles both; this is the per-competition WM token record
        # (server /metrics only sees the node-level aggregate).
        logger.info(
            f"sim call: in_tokens={usage.prompt_tokens} "
            f"out_tokens={usage.completion_tokens} "
            f"req_time={time.time() - t0:.2f}s model={model}"
        )
    return "".join(parts) if parts else None


class SimulatorInterpreter:
    """LLM-backed drop-in for `Interpreter`. Predicts stdout/stderr instead of running."""

    def __init__(
        self,
        working_dir: Path | str,
        model: str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        # Default generation ceiling; AIDE_SIMULATOR_SAMPLING can override it.
        max_tokens: int = 8192,
        environment: str | None = None,
    ):
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working directory {self.working_dir} does not exist"
        if not model:
            raise ValueError(
                "SimulatorInterpreter requires `model` (set cfg.exec.simulator_model)."
            )
        self.model = model
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython  # unused, kept for sig parity
        self.agent_file_name = agent_file_name  # unused, kept for sig parity
        self.max_tokens = max_tokens
        self.prompt_format, self.system_prompt = _resolve_system_prompt()
        logger.info(f"SimulatorInterpreter prompt format: {self.prompt_format}")
        self.environment = environment
        if self.timeout is None or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError(f"{self.prompt_format} requires a finite positive timeout")
        if self.environment is None:
            from .simulator_environment import resolve_environment

            # Default is the training-corpus text for this arch: a live probe
            # matches no training row (see simulator_environment).
            self.environment, env_meta = resolve_environment()
            logger.info(
                "SimulatorInterpreter environment: source={source} arch={arch} "
                "lines={lines} sha256={sha256}".format(**env_meta)
            )
        if not self.environment.strip():
            raise ValueError(f"{self.prompt_format} requires an environment manifest")
        self.exc_scoring_method = _exc_scoring_method(self.prompt_format)
        logger.info(f"SimulatorInterpreter exception scoring: {self.exc_scoring_method}")
        _positive_env_int("AIDE_EXC_SCORE_WORKERS", 8)
        _positive_env_int("AIDE_PCLEAN_CHUNK", 8)
        self.pclean_score = _resolve_pclean_score()
        if self.pclean_score != "p_clean":
            logger.info(
                f"AIDE_PCLEAN_SCORE={self.pclean_score}: judge runs the "
                "outcome-confidence arm (header 'Predicted outcome: ... "
                "(confidence = x)'), not P(runs clean)"
            )
        self._data_overview: str = ""

    def set_data_overview(self, text: str | None) -> None:
        """Cache the per-task data overview to embed in every simulator call."""
        self._data_overview = _normalize_to_aide_data_overview(text or "")

    def _user_message(self, code: str) -> str:
        overview = self._data_overview or "(no data overview available)"
        return (
            f"<code>\n{code}\n</code>\n\n"
            f"<environment>\n{self.environment}\n</environment>\n\n"
            f"<data_overview>\n{overview}\n</data_overview>\n\n"
            f"<runtime_budget>{int(round(self.timeout))}s</runtime_budget>"
        )

    def exc_scores(self, code: str) -> dict | None:
        """Return the outcome posterior and AIDE_PCLEAN_SCORE's selected scalar.

        Uses the same data overview, prompt, and model as run(). None means
        confidence is unavailable, not that execution is predicted to fail.
        """
        scores = predict_exc_posterior(
            self._data_overview,
            code,
            self.model,
            self.system_prompt,
            user_msg=self._user_message(code),
            method=self.exc_scoring_method,
        )
        if scores is None:
            return None
        # Offline analysis parses this line. Append new fields at the end only.
        logger.info(
            "exc_posterior: p_clean={p_clean:.4f} max_p={max_p:.4f} "
            "margin={margin:.4f} entropy={entropy:.4f} "
            "argmax={argmax_class} top_exc={top_exc} "
            "p_top_exc={p_top_exc:.4f} method={scoring_method} "
            "seconds={scoring_seconds:.3f}".format(**scores)
        )
        scores["score"] = scores[self.pclean_score]
        # Record the scalar's meaning without asking consumers to reread the env.
        scores["score_name"] = self.pclean_score
        return scores

    def p_clean(self, code: str) -> float | None:
        """Return the configured scalar, or None when scoring is unavailable."""
        scores = self.exc_scores(code)
        return None if scores is None else scores["score"]

    def metric_posterior(
        self, code: str, *, config: metric_sampling.MetricPosteriorConfig, deadline: float
    ) -> dict:
        """Resample from the same original input as run(), through </val_metric>.

        The deadline is monotonic inference time; the runtime budget in the WM
        input remains this interpreter's execution timeout.
        """
        return metric_sampling.sample(
            _sim_client,
            model=self.model,
            prefix=_render_sim_prefix(self.system_prompt, self._user_message(code)),
            config=config,
            deadline=deadline,
        )

    def create_process(self) -> None:
        return None

    def cleanup_session(self) -> None:
        return None

    def run(
        self,
        code: str,
        reset_session: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> ExecutionResult:
        cancel_event = cancel_event or threading.Event()

        # Empty-code short-circuit: predicting from an empty <code> block is
        # hallucination — return a zero-time success (no fabricated
        # "Execution time:" footer; empty output is the honest prediction).
        if not (code or "").strip():
            return ExecutionResult(
                term_out=[],
                exec_time=0.0,
                exc_type=None,
                exc_info=None,
                exc_stack=None,
                predicted_submission_exists="false",
            )

        # Avoid opening an HTTP stream for work cancelled before its executor
        # thread began. Mid-stream cancellation remains handled per chunk.
        if cancel_event.is_set():
            return ExecutionResult(
                term_out=["SimulatorCancelled"],
                exec_time=0.0,
                exc_type="SimulatorCancelled",
                exc_info=None,
                exc_stack=None,
            )

        logger.info("SimulatorInterpreter is predicting execution output")
        start_time = time.time()
        predicted_submission_exists = None
        predicted_metric = None
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._user_message(code),
            },
        ]
        attempts = 1 + _MALFORMED_RETRIES
        predicted = None
        malformed_responses = set()
        for attempt in range(1, attempts + 1):
            try:
                predicted = _streaming_query(
                    messages,
                    self.model,
                    self.max_tokens,
                    cancel_event,
                )
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Simulator backend error: {e}")
                return ExecutionResult(
                    term_out=[f"SimulatorError: {e}"],
                    exec_time=elapsed,
                    exc_type="SimulatorError",
                    exc_info={"args": [str(e)]},
                    exc_stack=[],
                )
            # An empty/cancelled request keeps its sentinel; malformed output
            # must never be mistaken for a clean run.
            if (
                predicted is None
                or _parse_env_result(predicted, prompt_format=self.prompt_format)
                or cancel_event.is_set()
            ):
                break
            # The usual temperature=0/seed=0 request can reproduce the same
            # truncated response exactly. One retry can recover a transient
            # truncation; repeating an already-seen malformed answer provides
            # no new evidence and needlessly delays the real-execution fallback.
            if predicted in malformed_responses:
                logger.warning(
                    "Simulator repeated an identical malformed response; " "stopping retries"
                )
                break
            malformed_responses.add(predicted)
            if attempt < attempts and not cancel_event.is_set():
                logger.warning(
                    "Simulator response is missing its required result fields "
                    f"({len(predicted)} chars); re-asking "
                    f"(attempt {attempt + 1}/{attempts})"
                )

        elapsed = time.time() - start_time

        if predicted is None or cancel_event.is_set():
            # Either cancelled or backend returned no content.
            reason = "SimulatorCancelled" if cancel_event.is_set() else "SimulatorEmpty"
            return ExecutionResult(
                term_out=[reason],
                exec_time=elapsed,
                exc_type=reason,
                exc_info=None,
                exc_stack=None,
            )

        result = _parse_env_result(str(predicted), prompt_format=self.prompt_format)
        if result is None:
            logger.warning(
                f"Simulator {self.prompt_format} response remained malformed "
                f"after {attempt} attempt(s); abstaining"
            )
            return ExecutionResult(["SimulatorEmpty"], elapsed, "SimulatorEmpty")
        lines = result["observed_output"].splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        exc_type = result["exc_type"]
        # Check raw terminal output before adding simulator metadata below.
        if exc_type == "TimeoutError" and not _ends_with_timeout_footer(lines):
            # Unlike real child output, the predicted log is rstripped (stop-
            # string path), so pad it or the footer merges into its last line
            # under "".join.
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines[-1] += "\n"
            lines.append(
                f"TimeoutError: Execution exceeded the time limit of "
                f"{humanize.naturaldelta(self.timeout)}"
            )
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        predicted_submission_exists = result["submission_exists"]
        lines.append(f"[simulator result] submission_exists={predicted_submission_exists}\n")
        lines.append(f"[simulator result] reported_metric={result['reported_metric']}\n")
        if result["reported_metric"] != "None":
            predicted_metric = float(result["reported_metric"])
        # Runtime is not predicted by this schema; 0.0 means unavailable.
        return ExecutionResult(
            term_out=lines,
            exec_time=0.0,
            exc_type=exc_type,
            exc_info=None,
            exc_stack=None,
            predicted_submission_exists=predicted_submission_exists,
            predicted_metric=predicted_metric,
        )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        sim = SimulatorInterpreter(td, model="qwen3_5-4b-sim")
        sim.set_data_overview("train.csv: rows=10 cols=[a,b]")
        res = sim.run("print('hello from simulator')")
        print("--- result ---")
        print(f"exc_type={res.exc_type!r} exec_time={res.exec_time:.2f}s")
        print("\n".join(res.term_out))

        print("\n--- pre-cancelled ---")
        ev = threading.Event()
        ev.set()
        res2 = sim.run("print('should not run')", cancel_event=ev)
        print(f"exc_type={res2.exc_type!r} exec_time={res2.exec_time:.2f}s")
        print("\n".join(res2.term_out))
