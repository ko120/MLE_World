"""OpenAI-compatible backend for local vLLM and Vertex models."""

import json
import logging
import os
import re
import time

from aide.backend.utils import FunctionSpec, OutputType, backoff_create, opt_messages_to_list
from funcy import notnone, once, select_values
import openai

logger = logging.getLogger("aide")

_client: openai.OpenAI = None  # type: ignore
_vertex_creds = None  # google.oauth2.credentials.Credentials, lazily built
_model_clients: dict = {}  # base_url -> openai.OpenAI, for AIDE_MODEL_BASE_URLS routing


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# Standard API parameters; vLLM sampling extensions go in extra_body.
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
_OUTPUT_CAP_KEYS = ("max_completion_tokens", "max_tokens")
_CONTEXT_MAX_ATTEMPTS = 3
_TOOL_CALL_MAX_ATTEMPTS = 3
_TEXT_RETRY_MAX_TOKENS = 32768
# Context recovery may shorten only optional evidence.
_COMPACTABLE_SECTIONS = {
    "Memory",
    "Execution output",
    "Predicted execution output",
    "Diagnosis of the failure (from an expert reviewer)",
    "Data Overview",
}


def _load_json_dict_env(var: str) -> dict:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"{var} is not valid JSON, ignoring: {e}")
        return {}
    if not isinstance(parsed, dict):
        logger.warning(f"{var} must be a JSON object, got {type(parsed).__name__}")
        return {}
    return parsed


def _positive_env_seconds(name: str, default: str) -> float:
    value = float(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _openai_timeout_seconds() -> float:
    return _positive_env_seconds("AIDE_OPENAI_TIMEOUT", "600")


def _vertex_timeout_seconds() -> float:
    return _positive_env_seconds("AIDE_VERTEX_TIMEOUT", "600")


def _vertex_retry_max_time_seconds() -> float:
    return _positive_env_seconds("AIDE_VERTEX_RETRY_MAX_TIME", "1200")


def _is_vertex_model(model: str) -> bool:
    return model.startswith("google/")


def _is_gemma4_model(model: str) -> bool:
    return re.search(r"gemma[-_]?4(?:\D|$)", model.lower()) is not None


@once
def _setup_openai_client():
    global _client
    _client = openai.OpenAI(max_retries=0, timeout=_openai_timeout_seconds())


def _get_client_for_model(model_name: str) -> openai.OpenAI:
    """Cache clients by AIDE_MODEL_BASE_URLS endpoint; unmapped models use the default."""
    base_url = _load_json_dict_env("AIDE_MODEL_BASE_URLS").get(model_name)
    if not base_url or not isinstance(base_url, str):
        if base_url is not None and not isinstance(base_url, str):
            logger.warning(
                f"AIDE_MODEL_BASE_URLS[{model_name!r}] must be a string URL, got "
                f"{type(base_url).__name__}; using the default client"
            )
        _setup_openai_client()
        return _client
    client = _model_clients.get(base_url)
    if client is None:
        client = openai.OpenAI(base_url=base_url, max_retries=0, timeout=_openai_timeout_seconds())
        _model_clients[base_url] = client
    return client


def _get_vertex_client(timeout: float | None = None) -> openai.OpenAI:
    """Build a Vertex client, refreshing GOOGLE_AUTH_USER_JSON credentials if expired."""
    global _vertex_creds
    from google.oauth2.credentials import Credentials
    import google.auth.transport.requests

    raw = os.environ.get("GOOGLE_AUTH_USER_JSON")
    if not raw:
        raise RuntimeError(
            "GOOGLE_AUTH_USER_JSON env var not set; required for Vertex AI / Gemini models"
        )
    info = json.loads(raw)

    if _vertex_creds is None:
        _vertex_creds = Credentials.from_authorized_user_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    if not _vertex_creds.valid or _vertex_creds.expired:
        _vertex_creds.refresh(google.auth.transport.requests.Request())

    project_id = os.environ.get("GCP_PROJECT_ID") or info.get("quota_project_id")
    if not project_id:
        raise RuntimeError("GCP project id not set (env GCP_PROJECT_ID or quota_project_id in key)")
    location = os.environ.get("GCP_LOCATION", "global")
    base_url = (
        f"https://aiplatform.googleapis.com/v1/projects/{project_id}"
        f"/locations/{location}/endpoints/openapi"
    )
    if timeout is None:
        timeout = _vertex_timeout_seconds()
    return openai.OpenAI(
        api_key=_vertex_creds.token,
        base_url=base_url,
        max_retries=0,
        timeout=timeout,
    )


def _load_sampling_overrides() -> dict:
    return _load_json_dict_env("AIDE_SAMPLING")


def _load_model_sampling(model_name: str):
    """A valid per-model entry replaces global sampling, including an empty dict."""
    entry = _load_json_dict_env("AIDE_MODEL_SAMPLING").get(model_name)
    if entry is None:
        return None
    if not isinstance(entry, dict):
        logger.warning(
            f"AIDE_MODEL_SAMPLING[{model_name!r}] must be a JSON object, got "
            f"{type(entry).__name__}; falling back to AIDE_SAMPLING"
        )
        return None
    return entry


def _apply_sampling_overrides(model_kwargs: dict) -> None:
    overrides = _load_model_sampling(model_kwargs.get("model") or "")
    if overrides is None:
        overrides = _load_sampling_overrides()
    extra_body = dict(model_kwargs.get("extra_body") or {})
    explicit_cap = any(model_kwargs.get(k) is not None for k in _OUTPUT_CAP_KEYS)
    for k, v in (overrides or {}).items():
        # Explicit per-call output caps take precedence over sampling defaults.
        if k in _OUTPUT_CAP_KEYS and explicit_cap:
            continue
        if k in _OPENAI_TOP_LEVEL:
            model_kwargs[k] = v
        else:
            extra_body[k] = v
    if extra_body:
        model_kwargs["extra_body"] = extra_body
    # Match vLLM's precedence when both cap aliases are supplied.
    if model_kwargs.get("max_completion_tokens") is not None:
        model_kwargs.pop("max_tokens", None)


def _prepare_request(model_kwargs, func_spec, n, retry_max_time, retry_attempt_timeout):
    """Resolve provider defaults, client routing, and the API request parameters."""
    model_name = model_kwargs.get("model", "")
    is_vertex = _is_vertex_model(model_name)
    if is_vertex:
        # Vertex uses provider sampling/output defaults and medium reasoning.
        reasoning_effort = os.environ.get("AIDE_VERTEX_REASONING_EFFORT", "medium").strip().lower()
        if reasoning_effort not in {"minimal", "low", "medium", "high"}:
            raise ValueError(
                "AIDE_VERTEX_REASONING_EFFORT must be one of minimal, low, medium, high"
            )
        vertex_timeout = _vertex_timeout_seconds()
        retry_max_time = (
            min(retry_max_time, _vertex_retry_max_time_seconds())
            if retry_max_time is not None
            else _vertex_retry_max_time_seconds()
        )
        retry_attempt_timeout = (
            min(retry_attempt_timeout, vertex_timeout)
            if retry_attempt_timeout is not None
            else vertex_timeout
        )

        # Rebuild on every attempt so expired OAuth tokens can be refreshed.
        def create_fn(**kw):
            return _get_vertex_client(vertex_timeout).chat.completions.create(**kw)

        # extra_body supports reasoning_effort with the pinned OpenAI SDK.
        filtered_kwargs: dict = {
            "model": model_name,
            "extra_body": {"reasoning_effort": reasoning_effort},
        }
    else:
        if retry_attempt_timeout is not None or retry_max_time is not None:
            # Bound each request even when only a total allowance was supplied.
            local_timeout = _openai_timeout_seconds()
            retry_attempt_timeout = (
                min(retry_attempt_timeout, local_timeout)
                if retry_attempt_timeout is not None
                else local_timeout
            )
            explicit_timeout = model_kwargs.get("timeout")
            if isinstance(explicit_timeout, (int, float)):
                retry_attempt_timeout = min(retry_attempt_timeout, explicit_timeout)
        create_fn = _get_client_for_model(model_name).chat.completions.create
        filtered_kwargs = select_values(notnone, model_kwargs)  # type: ignore

    if func_spec is not None:
        if is_vertex or _is_gemma4_model(model_name):
            # Gemma can render forced calls as Python text with no tool_calls.
            # Constrain the arguments themselves instead of that tool syntax.
            filtered_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": func_spec.name, "schema": func_spec.json_schema},
            }
        else:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    if not is_vertex:
        _apply_sampling_overrides(filtered_kwargs)

    if n > 1:
        assert func_spec is None, "n>1 not supported with func_spec (no batched tool calls)"
        filtered_kwargs["n"] = n
    return create_fn, filtered_kwargs, retry_max_time, retry_attempt_timeout, is_vertex


def _choice_reasoning(choice) -> str | None:
    message = choice.message
    extra = getattr(message, "model_extra", None) or {}
    return (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or extra.get("reasoning")
        or extra.get("reasoning_content")
    )


def _truncated_without_complete_answer(choice) -> bool:
    """Recover empty answers and code cut off inside a Markdown fence.

    A syntactically valid prefix can still omit training or submission output.
    Preserve completed fenced answers when only trailing prose was truncated.
    """
    if choice is None or choice.finish_reason != "length":
        return False
    content = choice.message.content or ""
    if not content.strip():
        return True
    fence = None
    for line in content.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if not match:
            continue
        marker, suffix = match.groups()
        if fence is None:
            fence = marker
        elif marker[0] == fence[0] and len(marker) >= len(fence) and not suffix.strip():
            fence = None
    return fence is not None


def _extract_choice_text(choice) -> str:
    """Return answer text; never mistake unfinished reasoning for a solution."""
    if choice is None:
        logger.warning("openai backend: response has no choice; returning empty text")
        return ""
    if _truncated_without_complete_answer(choice):
        logger.warning(
            "openai backend: output limit left an incomplete answer; returning "
            "empty text for caller recovery (finish_reason=length)"
        )
        return ""
    output = choice.message.content or ""
    if output.strip():
        return output
    if choice.finish_reason == "length":
        logger.warning(
            "openai backend: no answer before output limit; returning empty text "
            "for caller recovery (finish_reason=length)"
        )
        return ""
    reasoning = _choice_reasoning(choice)
    if reasoning:
        logger.warning(
            f"openai backend: empty content, falling back to completed reasoning "
            f"({len(reasoning)} chars). finish_reason={choice.finish_reason}"
        )
        return reasoning
    logger.warning(
        f"openai backend: empty content and no reasoning field. "
        f"finish_reason={choice.finish_reason}"
    )
    return ""


def _extract_tool_call(choice, func_spec: FunctionSpec) -> dict | None:
    """Return forced tool-call arguments, or None to request another sample."""
    msg = choice.message
    if not msg.tool_calls:
        logger.error(f"function_call is empty, it is not a function call: {msg}")
        return None
    call = msg.tool_calls[0].function
    if call.name != func_spec.name:
        logger.warning(
            f"tool call name mismatch: got {call.name!r}, expected "
            f"{func_spec.name!r}; requesting a matching tool response"
        )
        return None
    try:
        parsed = json.loads(call.arguments)
    except json.JSONDecodeError:
        logger.error(f"Error decoding the function arguments: {call.arguments}")
        return None
    if isinstance(parsed, dict):
        return parsed
    logger.error(f"function arguments are not a JSON object: {call.arguments[:200]!r}")
    return None


def _parse_structured_content(choice, func_spec: FunctionSpec) -> dict | None:
    """Parse constrained JSON content using the tool-call return interface."""
    content = choice.message.content or ""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    logger.error(f"structured output not usable JSON: {content[:200]!r}")
    return None


def _compact_context(messages: list[dict]) -> bool:
    """Trim oversized AIDE evidence sections, retaining their beginning and end."""
    changed = False
    for message in messages:
        content = message["content"]
        # Recognize top-level sections outside fenced code blocks.
        headers = []
        fence = ""
        for match in re.finditer(r"(?m)^(?:# [^\n]*| {0,3}(?:`{3,}|~{3,})[^\n]*)(?:\n|$)", content):
            line = match[0].strip()
            if fence:
                if line.startswith(fence) and not line.strip(fence[0]):
                    fence = ""
            elif line.startswith(("```", "~~~")):
                fence = re.match(r"[`~]+", line)[0]
            else:
                headers.append(match)
        if fence:
            # Ambiguous boundaries could absorb instructions into a trimmed body.
            logger.warning("context recovery skipped a prompt with an unclosed code fence")
            continue
        pieces = []
        end = 0
        for i, header in enumerate(headers):
            start = header.end()
            stop = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            title = header[0][2:].strip()
            body = content[start:stop]
            if title in _COMPACTABLE_SECTIONS and len(body) > 16384:
                marker = "\n[... context shortened after server rejection ...]\n"
                keep = (16384 - len(marker)) // 2
                shortened = body[:keep] + marker + body[-keep:]
                pieces.extend((content[end:start], shortened))
                end = stop
                logger.warning(
                    "context-window overflow: shortened %s from %d to %d characters",
                    title,
                    len(body),
                    len(shortened),
                )
                changed = True
        if pieces:
            message["content"] = "".join([*pieces, content[end:]])
    return changed


def _recover_context_overflow(messages: list[dict], kwargs: dict, error: str) -> bool:
    """Repair known context failures without guessing tokens from character counts."""
    limit = re.search(r"maximum context length is (\d+) tokens", error)
    prompt = re.search(r"prompt contains (\d+) input tokens", error)
    if limit and prompt:
        # Only exact counts can size the output cap; 'at least N' cannot.
        available = int(limit[1]) - int(prompt[1])
        for key in _OUTPUT_CAP_KEYS:
            cap = kwargs.get(key)
            if isinstance(cap, int) and 0 < available < cap:
                kwargs[key] = available
                logger.warning("context-window overflow: reducing %s to %d", key, available)
                return True
    if _compact_context(messages):
        return True
    lower_bound = re.search(r"prompt contains (?:at least )?(\d+) input tokens", error)
    prompt_too_large = limit and lower_bound and int(lower_bound[1]) >= int(limit[1])
    # Without an exact count, let the server size output once if input may fit.
    if not prompt_too_large and any(key in kwargs for key in _OUTPUT_CAP_KEYS):
        for key in _OUTPUT_CAP_KEYS:
            kwargs.pop(key, None)
        logger.warning("context-window overflow: retrying once without an explicit output cap")
        return True
    logger.error("context-window overflow: required prompt content still exceeds the model window")
    return False


def _truncation_retry_kwargs(kwargs: dict, completion, *, n: int) -> tuple[dict, bool]:
    """Give a failed reply more output room without changing its prompt."""
    retry_kwargs = dict(kwargs)
    key = next((k for k in _OUTPUT_CAP_KEYS if kwargs.get(k) is not None), "max_tokens")
    # Usage is summed over the batch, so it is only a capped estimate when
    # the caller left the per-choice limit to the server.
    cap = kwargs.get(key) or max(
        1, min(completion.usage.completion_tokens, _TEXT_RETRY_MAX_TOKENS // 2)
    )
    # Never lower an explicitly larger cap; limit automatic expansion to 32K.
    retry_kwargs[key] = max(cap, min(2 * cap, _TEXT_RETRY_MAX_TOKENS))
    retry_kwargs["n"] = n
    extra = dict(kwargs.get("extra_body") or {})
    automatic_thinking_budget = False
    if (
        "vllm" in kwargs.get("model", "").lower()
        and any(_choice_reasoning(c) for c in completion.choices)
        and "thinking_token_budget" not in extra
    ):
        # Reserve answer space for confirmed vLLM reasoning models only.
        extra["thinking_token_budget"] = min(cap, retry_kwargs[key] // 2)
        automatic_thinking_budget = True
    if extra:
        retry_kwargs["extra_body"] = extra
    return retry_kwargs, automatic_thinking_budget


def _complete_text(create, completion, kwargs, *, is_vertex, n):
    """Retry truncated slots once, retaining completed batch siblings."""
    choices = list(completion.choices)
    failed = [i for i, c in enumerate(choices) if _truncated_without_complete_answer(c)]
    if failed and not is_vertex:
        logger.warning(
            "openai backend: %d/%d replies exhausted output tokens without a complete answer; "
            "retrying only those replies once",
            len(failed),
            len(choices),
        )
        retry_kwargs, auto_budget = _truncation_retry_kwargs(kwargs, completion, n=len(failed))
        try:
            retried = create(retry_kwargs, automatic_thinking_budget=auto_budget)
        except (openai.APIError, TimeoutError) as exc:
            logger.warning("openai backend: truncated-reply recovery failed: %s", exc)
        else:
            for index, replacement in zip(failed, retried.choices):
                choices[index] = replacement
    texts = [_extract_choice_text(c) for c in choices]
    return (texts if n > 1 else (texts[0] if texts else "")), choices


def _complete_tool_call(create, completion, kwargs, func_spec, *, is_vertex, attempt_limit,
                        structured_content=False):
    """Retry malformed tool/JSON responses, then let the caller recover with {}."""
    extract = _parse_structured_content if is_vertex or structured_content else _extract_tool_call
    for attempt in range(1, attempt_limit + 1):
        choice = completion.choices[0] if completion.choices else None
        # Valid JSON alone is not completion evidence: a truncated object can
        # omit optional fields or be one prefix of an unfinished tool response.
        output = (
            extract(choice, func_spec)
            if choice is not None and choice.finish_reason != "length"
            else None
        )
        if output is not None:
            return output, completion
        if attempt == attempt_limit:
            break
        logger.warning(
            f"no usable {func_spec.name} response; "
            f"re-requesting (attempt {attempt + 1}/{attempt_limit})"
        )
        auto_budget = False
        if choice is not None and choice.finish_reason == "length" and not is_vertex:
            kwargs, auto_budget = _truncation_retry_kwargs(kwargs, completion, n=1)
        completion = create(kwargs, automatic_thinking_budget=auto_budget)
    logger.error(
        f"giving up on {func_spec.name} tool call after "
        f"{attempt_limit} attempts; returning empty dict"
    )
    return {}, completion


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    n: int = 1,
    retry_max_time: float | None = None,
    retry_attempt_timeout: float | None = None,
    tool_call_max_attempts: int | None = None,
    temperature_override: float | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Run one query with a shared retry budget and complete token accounting."""
    tool_attempt_limit = (
        _TOOL_CALL_MAX_ATTEMPTS if tool_call_max_attempts is None else tool_call_max_attempts
    )
    if tool_attempt_limit < 1:
        raise ValueError("tool_call_max_attempts must be positive")
    create_fn, filtered_kwargs, retry_max_time, retry_attempt_timeout, is_vertex = _prepare_request(
        model_kwargs, func_spec, n, retry_max_time, retry_attempt_timeout
    )
    # Source review may need deterministic decisions even when generation and
    # general feedback use an environment-wide sampling recipe. This explicit
    # control never reaches the provider as an unknown API parameter.
    if temperature_override is not None and not is_vertex:
        filtered_kwargs["temperature"] = temperature_override
    initial_thinking_budget = False
    model_name = model_kwargs.get("model", "")
    if not is_vertex and _is_gemma4_model(model_name) and "vllm" in model_name.lower():
        cap = next((filtered_kwargs[k] for k in _OUTPUT_CAP_KEYS
                    if filtered_kwargs.get(k) is not None), None)
        extra = dict(filtered_kwargs.get("extra_body") or {})
        if isinstance(cap, int) and 1 < cap <= 4096 and "thinking_token_budget" not in extra:
            # Short audit/judge calls have tight token and wall-time limits.
            # Reserve answer space before the first attempt, not after the
            # reasoning has consumed their shared review deadline.
            extra["thinking_token_budget"] = cap // 2
            filtered_kwargs["extra_body"] = extra
            initial_thinking_budget = True
    messages = opt_messages_to_list(
        system_message, user_message, convert_system_to_user=convert_system_to_user
    )
    gemma_function = func_spec is not None and _is_gemma4_model(model_kwargs.get("model", ""))
    if gemma_function:
        # vLLM's decoder enforces the schema but does not show field meanings
        # to the model. Preserve that contract in the prompt as well.
        messages.append({
            "role": "user",
            "content": (
                f"Return the arguments for {func_spec.name} as one JSON object. "
                "Do not emit a Python function call or Markdown fences.\n"
                f"{func_spec.description}\nJSON schema, including field meanings:\n"
                + json.dumps(func_spec.json_schema)
            ),
        })

    t0 = time.time()
    retry_started = time.monotonic()
    completions = []
    context_retries = 0

    def bounded_create(request_kwargs, *, automatic_thinking_budget=False):
        nonlocal context_retries
        # All recovery paths and transport retries share this query's allowance.
        for attempt in range(_CONTEXT_MAX_ATTEMPTS):
            remaining = retry_max_time
            if remaining is not None:
                remaining -= time.monotonic() - retry_started
                if remaining <= 0:
                    raise TimeoutError("LLM query time allowance exhausted")
            if automatic_thinking_budget:
                # Preserve answer space if context recovery reduced the cap.
                for key in _OUTPUT_CAP_KEYS:
                    if request_kwargs.get(key) is not None:
                        extra = request_kwargs["extra_body"]
                        extra["thinking_token_budget"] = min(
                            extra["thinking_token_budget"], request_kwargs[key] // 2
                        )
                        break
            try:
                response = backoff_create(
                    create_fn,
                    OPENAI_TIMEOUT_EXCEPTIONS,
                    retry_max_time=remaining,
                    retry_attempt_timeout=retry_attempt_timeout,
                    messages=[dict(m) for m in messages],
                    **request_kwargs,
                )
            except openai.BadRequestError as exc:
                if attempt == _CONTEXT_MAX_ATTEMPTS - 1:
                    raise
                error = str(exc)
                if "maximum context length" in error:
                    if not _recover_context_overflow(messages, request_kwargs, error):
                        raise
                    context_retries += 1
                elif automatic_thinking_budget and "thinking_token_budget" in error:
                    # Older vLLM servers can reject the automatic extension.
                    request_kwargs["extra_body"] = dict(request_kwargs["extra_body"])
                    request_kwargs["extra_body"].pop("thinking_token_budget", None)
                    automatic_thinking_budget = False
                    logger.warning(
                        "vLLM rejected automatic thinking_token_budget; retrying without it"
                    )
                else:
                    raise
            else:
                completions.append(response)
                return response

    completion = bounded_create(filtered_kwargs, automatic_thinking_budget=initial_thinking_budget)
    if func_spec is None:
        output, choices = _complete_text(
            bounded_create, completion, filtered_kwargs, is_vertex=is_vertex, n=n
        )
    else:
        output, completion = _complete_tool_call(
            bounded_create,
            completion,
            filtered_kwargs,
            func_spec,
            is_vertex=is_vertex,
            attempt_limit=tool_attempt_limit,
            structured_content=gemma_function,
        )
        choices = list(completion.choices)
    req_time = time.time() - t0
    # Include discarded reasoning/tool samples in usage accounting.
    in_tokens = sum(c.usage.prompt_tokens for c in completions)
    out_tokens = sum(c.usage.completion_tokens for c in completions)
    # Preserve field order for existing log/accounting parsers.
    logger.info(
        f"openai call: in_tokens={in_tokens} out_tokens={out_tokens} "
        f"req_time={req_time:.2f}s model={completion.model}"
    )

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
        "response_attempts": len(completions),
        "context_retries": context_retries,
        "finish_reasons": [c.finish_reason for c in choices],
    }
    # Retain reasoning and finish metadata for callers persisting artifacts.
    first_choice = choices[0] if choices else None
    if first_choice is not None:
        info["reasoning"] = _choice_reasoning(first_choice)
        info["finish_reason"] = first_choice.finish_reason

    return output, req_time, in_tokens, out_tokens, info
