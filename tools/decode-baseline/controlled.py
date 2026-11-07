"""Controlled synchronous generation loop for the Phase 8.2 decode baseline.

Frozen contract (req/plans/phase8-2-reference-findings.md section 6):

- one full-prompt forward through the loaded model's backbone with
  last-position-only output projection through the model's own output head,
  then exactly (emitted - 1) cached forwards for the emitted tokens;
- no next-token lookahead, no async_eval pipeline, no periodic cache clearing;
- the required logits and cache state are evaluated before each step timer
  stops;
- greedy argmax with first-maximum (lowest index) tie-break, matching the
  stock loop's greedy sampler and the Java adapter;
- no stop/EOS termination: all emitted tokens are produced, including stop
  set members (manifest controlled_generation.stop_termination = false).

Own-head rule: tied models (no stored lm_head) project through the embedding's
as_linear; untied models use the loaded lm_head module, preserving any
quantized implementation. No dense head is ever substituted and no module is
reconstructed from copied weights.
"""

import time

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

EMITTED_TOKENS = 128
CONTROLLED_FINISH_REASON = "controlled-max-tokens"


class ControlledContractError(Exception):
    """Raised when the loaded model does not expose the expected structure."""


def own_head(model):
    """Return the loaded model's own output projection (see module docstring)."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.as_linear
    raise ControlledContractError(
        "loaded model has neither an lm_head module nor a model.embed_tokens; "
        "the own-head rule cannot be applied"
    )


def cache_states(cache):
    states = []
    for entry in cache:
        states.extend(entry.state)
    return states


def greedy_token(logits):
    """First-maximum argmax of the last-position logits (shape [1, 1, V]) as a
    Python int. argmax over the flattened row keeps the lowest index on ties,
    matching the stock greedy sampler."""
    return int(mx.argmax(logits).item())


def top2_margin(row):
    """Margin between the two largest logits of a 1-D [V] row (float).

    mx.sort in the pinned mlx 0.32.4 sorts ascending with no order keyword,
    so the two largest values are the last two of the sorted row. Callers
    pass the flattened last-position logits row, e.g. logits[0, 0] of a
    [1, 1, V] projection."""
    s = mx.sort(mx.reshape(row, (-1,))).tolist()
    return float(s[-1] - s[-2])


def run(model, prompt_ids, emitted=EMITTED_TOKENS):
    """Run the controlled loop over a fresh per-request cache.

    Returns (tokens, prefill_s, steps, token_rel_s):

    - tokens: the emitted token ids (token 1 from the prefill projection, then
      one per cached step);
    - prefill_s: monotonic seconds from t_entry to evaluated prefill logits;
    - steps: [{"position": 1-based position of the fed token (the cache
      length after the step), "s": seconds}] for the (emitted - 1) cached
      steps;
    - token_rel_s: host-visible relative time (seconds from t_entry) of each
      emitted token; every entry, token 1 included, is taken after that
      token's argmax.
    """
    if emitted < 1:
        raise ControlledContractError("emitted must be >= 1")
    head = own_head(model)
    cache = make_prompt_cache(model)
    t0 = time.monotonic()

    hidden = model.model(mx.array(prompt_ids, dtype=mx.int32)[None], cache)
    logits = head(hidden[:, -1:, :])
    mx.eval(logits, *cache_states(cache))
    prefill_s = time.monotonic() - t0

    # Token 1's event time is taken after its argmax, like every later
    # token's; prefill_s above stops at the evaluated logits.
    tokens = [greedy_token(logits)]
    token_rel_s = [time.monotonic() - t0]
    steps = []
    for k in range(emitted - 1):
        step_start = time.monotonic()
        tok = mx.array([tokens[-1]], dtype=mx.int32)
        hidden = model.model(tok[None], cache)
        logits = head(hidden[:, -1:, :])
        mx.eval(logits, *cache_states(cache))
        steps.append({"position": len(prompt_ids) + k + 1, "s": time.monotonic() - step_start})
        tokens.append(greedy_token(logits))
        token_rel_s.append(time.monotonic() - t0)
    return tokens, prefill_s, steps, token_rel_s


def run_fixed_continuation(model, prompt_ids, continuation_ids):
    """Step the fixed-continuation ids through the model (no sampling).

    Each step feeds exactly one given id (identical context growth on both
    runtimes) and records the evaluated last-position projection:
    position (1-based position of the fed token, i.e. the cache length after
    the step), seconds, greedy argmax id and top-2 margin. Used by the
    qualification workload (plan section 3.3); it is not a generation run and
    emits no tokens.

    Returns (prefill_s, steps, step_region_s): step_region_s is the
    monotonic wall time over the step region only (from the prefill
    evaluation to the end of the final step's argmax/margin host work); it
    includes that per-step host work, so it is not decode time (the
    per-step s values are the decode comparison; findings section 13.3),
    and it is defined the same way on the stock path. prefill_s is
    per-adapter and not comparable across adapters: the stock shipped call
    projects the output head at every prompt position, while the
    controlled prefill does so only at the last position.
    """
    head = own_head(model)
    cache = make_prompt_cache(model)
    t0 = time.monotonic()

    hidden = model.model(mx.array(prompt_ids, dtype=mx.int32)[None], cache)
    logits = head(hidden[:, -1:, :])
    mx.eval(logits, *cache_states(cache))
    t_steps = time.monotonic()
    prefill_s = t_steps - t0

    steps = []
    for k, fed in enumerate(continuation_ids):
        step_start = time.monotonic()
        tok = mx.array([int(fed)], dtype=mx.int32)
        hidden = model.model(tok[None], cache)
        logits = head(hidden[:, -1:, :])
        mx.eval(logits, *cache_states(cache))
        steps.append(
            {
                "position": len(prompt_ids) + k + 1,
                "s": time.monotonic() - step_start,
                "argmax_id": greedy_token(logits),
                "top2_margin": top2_margin(logits),
            }
        )
    step_region_s = time.monotonic() - t_steps
    return prefill_s, steps, step_region_s
