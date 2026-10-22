"""HTTP operations for the simulator's exception-posterior scorers.

The caller supplies its existing client and tokenizer callbacks. This keeps
connection/cancellation policy in simulator.py and permits mocked transport tests
without model servers. The protocol vocabulary and source-extracted rendering
helpers remain in simulator.py for offline evaluation compatibility.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def tokenize_exc_prompts(client, prefix, model, *, support, label_prefix, pack, unpack):
    """Tokenize all complete class strings in one CPU-only request.

    Qwen's registered <|im_start|> token prevents merges across packed segments.
    Reconstructing each segment with the shared input gives exactly the IDs of
    independently tokenizing each complete prompt. This retains BPE merges such
    as >Name. No token IDs are hard-coded or cached across model reloads.
    """
    path = client.base_url.path.rstrip("/")
    if not path.endswith("/v1"):
        raise ValueError("AIDE_SIMULATOR_BASE_URL must end in /v1")
    url = str(client.base_url.copy_with(path=path[:-3] + "/tokenize"))
    response = client.post(
        url,
        cast_to=dict[str, Any],
        body={
            "model": model,
            "prompt": pack(prefix, support, label_prefix),
            "add_special_tokens": False,
            "return_token_strs": True,
        },
    )
    return unpack(response, support, label_prefix)


def single_token_logps(client, prefix, model, *, tokenize, support):
    """Return all 82 raw label log probabilities from one next-token request.

    /tokenize validates ordinary EXC_* tokens in the complete tag context;
    it performs no model forward. Only the prefix ending after <exc_type>
    enters /completions. No label branch, forced suffix, or trace is decoded.
    """
    base, suffixes = tokenize(prefix, model, support=support, label_prefix="EXC_")
    label_ids = [suffix[0] for suffix in suffixes]
    response = client.completions.create(
        model=model,
        prompt=base,
        max_tokens=1,
        temperature=0.0,
        logprobs=len(label_ids),
        extra_body={
            "logprob_token_ids": label_ids,
            "return_tokens_as_token_ids": True,
            "add_special_tokens": False,
        },
    )
    if len(response.choices) != 1:
        raise ValueError("single-token scorer requires exactly one completion")
    if response.usage is None or response.usage.completion_tokens != 1:
        raise ValueError("single-token scorer requires exactly one generated token")
    lp = response.choices[0].logprobs
    if (
        lp is None
        or len(lp.tokens) != 1
        or len(lp.token_logprobs) != 1
        or lp.top_logprobs is None
        or len(lp.top_logprobs) != 1
        or lp.top_logprobs[0] is None
    ):
        raise ValueError("single-token scorer requires log probabilities at one position")
    values = [lp.top_logprobs[0].get(f"token_id:{t}") for t in label_ids]
    if any(v is None or not math.isfinite(v) or v > 0 for v in values):
        raise ValueError("missing or invalid EXC_* log probabilities; request all 82 token IDs")
    return values


def forced_suffix_logps(client, prefix, model, *, tokenize, workers):
    """Decode only forced suffixes, reporting pre-processor raw logprobs.

    The server must load aide_wm_logits:ForceTokens with raw_logprobs and
    prefix caching enabled. Missing/ignored processor requests fail the exact
    output-token check; there is no silent fallback to the expensive echo path.
    """
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exc-suffix") as pool:
        base, suffixes = tokenize(prefix, model)

        def score(suffix):
            response = client.completions.create(
                model=model,
                prompt=base,
                max_tokens=len(suffix),
                temperature=0.0,
                logprobs=0,
                extra_body={
                    "vllm_xargs": {"forced_token_ids": suffix},
                    "return_tokens_as_token_ids": True,
                    "ignore_eos": True,
                },
            )
            if len(response.choices) != 1:
                raise ValueError("forced scorer requires exactly one completion")
            lp = response.choices[0].logprobs
            if lp is None or lp.tokens != [f"token_id:{t}" for t in suffix]:
                raise ValueError("forced suffix was not honored; load aide_wm_logits:ForceTokens")
            values = lp.token_logprobs
            if len(values) != len(suffix) or any(v is None or not math.isfinite(v) for v in values):
                raise ValueError("missing or nonfinite forced-token log probabilities")
            return sum(values)

        # Finish one branch first so the remaining branches can reuse its KV
        # cache. All branches include the complete class AND closing tag.
        first = score(suffixes[0])
        return [first, *pool.map(score, suffixes[1:])]


def echo_logps(client, prefix, model, *, support, chunk):
    """Original full-prompt scorer, retained for explicit reference comparisons.

    Prompt logprobs bypass vLLM's prefix-cache reads. Keep batches small: their
    full-vocabulary intermediate tensors can exhaust memory on a shared server.
    """
    prompts = [f"{prefix}<exc_type>{c}</exc_type>" for c in support]
    logps = []
    for lo in range(0, len(prompts), chunk):
        batch = prompts[lo : lo + chunk]
        response = client.completions.create(
            model=model,
            prompt=batch,
            max_tokens=0,
            echo=True,
            logprobs=0,
            temperature=0.0,
        )
        by_idx = {ch.index: ch for ch in response.choices}
        if set(by_idx) != set(range(len(batch))):
            raise ValueError("missing or duplicate echo choices")
        for i in range(len(batch)):
            lp = by_idx[i].logprobs
            if lp is None or len(lp.text_offset) != len(lp.token_logprobs):
                raise ValueError("missing echo log probabilities")
            values = [v for off, v in zip(lp.text_offset, lp.token_logprobs) if off >= len(prefix)]
            if not values or any(v is None or not math.isfinite(v) for v in values):
                raise ValueError("missing or nonfinite echo suffix log probabilities")
            logps.append(sum(values))
    return logps
