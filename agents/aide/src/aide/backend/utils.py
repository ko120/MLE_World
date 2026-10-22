import logging
import random
import time
from dataclasses import dataclass
from typing import Callable

import jsonschema
from dataclasses_json import DataClassJsonMixin

PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

logger = logging.getLogger("aide")


def backoff_create(
    create_fn: Callable,
    retry_exceptions: tuple[type[BaseException], ...],
    *args,
    retry_max_time: float | None = None,
    retry_attempt_timeout: float | None = None,
    **kwargs,
):
    """Call ``create_fn`` with exponential backoff on transient exceptions.

    ``retry_max_time`` is a wall-clock deadline for this retry sequence. The
    backoff package's ``max_time`` samples elapsed time *before* an attempt, so
    a long request can cross the deadline and still trigger one more full
    request. This loop checks after every attempt and, for clients that accept
    a per-request ``timeout`` (Vertex/OpenAI), caps that timeout to the remaining
    deadline. Thus a 600s attempt limit plus a 1200s retry limit cannot turn
    into a third 600s request.

    With ``retry_max_time=None``, behavior remains retry-forever. This is
    intentional for local vLLM, where the supervisor may restart the server
    mid-call. Provider backends choose their own deadline per call, allowing a
    mixed local-vLLM + Vertex process to use both policies simultaneously.
    """
    if retry_max_time is not None and retry_max_time <= 0:
        raise ValueError("retry_max_time must be greater than zero")
    if retry_attempt_timeout is not None and retry_attempt_timeout <= 0:
        raise ValueError("retry_attempt_timeout must be greater than zero")

    started = time.monotonic()
    failure_count = 0
    last_exc: BaseException | None = None

    while True:
        remaining: float | None = None
        if retry_max_time is not None:
            remaining = retry_max_time - (time.monotonic() - started)
            if remaining <= 0:
                if last_exc is not None:
                    raise last_exc
                raise TimeoutError("LLM retry deadline exhausted before first attempt")

        call_kwargs = kwargs
        if retry_attempt_timeout is not None:
            timeout = retry_attempt_timeout
            if remaining is not None:
                timeout = min(timeout, remaining)
            call_kwargs = {**kwargs, "timeout": timeout}

        try:
            return create_fn(*args, **call_kwargs)
        except retry_exceptions as exc:
            last_exc = exc
            failure_count += 1
            logger.info(f"Backoff exception: {exc}")

            if retry_max_time is not None:
                remaining = retry_max_time - (time.monotonic() - started)
                if remaining <= 0:
                    logger.warning(
                        f"LLM retry deadline exhausted after {failure_count} "
                        f"attempt(s) and {time.monotonic() - started:.1f}s"
                    )
                    raise

            # Same schedule as the old backoff.expo(factor=1.5, max_value=60)
            # with its default full jitter, without the pre-attempt max_time bug.
            exponent = min(failure_count - 1, 6)
            wait_ceiling = min(60.0, 1.5 * (2**exponent))
            wait_seconds = random.uniform(0.0, wait_ceiling)
            if remaining is not None:
                wait_seconds = min(wait_seconds, remaining)
            if wait_seconds > 0:
                time.sleep(wait_seconds)


def opt_messages_to_list(
    system_message: str | None,
    user_message: str | None,
    convert_system_to_user: bool = False,
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        if convert_system_to_user:
            messages.append({"role": "user", "content": system_message})
        else:
            messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
            "strict": True,
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }
