# CPU usage at high concurrency — profiling findings

This is the result of profiling the user-reported scenario where
~300 concurrent model connections drive CPU usage high. The
benchmark and profile-analysis tools live alongside this file
(`cpu_concurrency.py`, `analyze_profile.py`).

## Methodology

- `bench/cpu_concurrency.py` runs the full `inspect_ai.eval` pipeline against
  the `mockllm/` provider with a callable `custom_outputs` that returns
  `ModelOutput` with pre-populated `usage` (so mockllm itself isn't doing
  per-call `count_tokens`).
- The mock model is registered via `set_model_info(...)` so
  `get_model_info` does a fast dict hit instead of fuzzy-match (a separate
  hotspot, see finding #1 below).
- Each sample's solver calls `model.generate()` `--turns` times with a fixed
  message history of `--history` messages.
- We seed the sample store with `--store-bytes` of data so
  `track_store_changes()` has realistic work.
- Profiling uses `py-spy record --idle --rate 250 --format speedscope`,
  parsed by `bench/analyze_profile.py`.

All runs below are single-threaded Python on Linux, `--display plain`, no
real network I/O. cpu_util ≈ 1.00 in every run, confirming we are
bottlenecked on Python (matching the user's report).

## Headline numbers

| samples × turns × history | wall  | cpu/turn | calls/s |
|---------------------------|-------|----------|---------|
| 300 × 1 × 1               | 1.97s | 6.34ms   | 152     |
| 300 × 5 × 30              | 5.28s | 3.51ms   | 284     |
| 300 × 30 × 30             | 20.9s | 2.32ms   | 430     |

Steady-state is roughly **~2.3ms of CPU per generate turn**. With 300 in-flight
samples and a typical single-process eval, the event loop is the bottleneck —
adding more network concurrency past that point can't help, because we're
already pegged on Python.

## What the profile actually says

Aggregated by subsystem from a 28s run (300 × 30 × 30, mock model
registered) — full data in `profiles/profile_300x30x30.json`:

| subsystem                  | inclusive % |
|----------------------------|-------------|
| anyio                      | 92.7%       |
| asyncio                    | 92.5%       |
| eval pipeline              | 99.1%       |
| **model layer**            | **54.3%**   |
| **pydantic**               | **44.5%**   |
| mockllm provider           | 18.0%       |
| log recorder               | 8.4%        |
| copy.deepcopy              | 6.0%        |
| span/transcript            | 1.4%        |
| store                      | 0.3%        |

(Subsystem times overlap because they're inclusive — a frame can belong to
multiple categories along its stack.)

The clearest **self-time** offenders:

| %     | function                                | location             |
|-------|------------------------------------------|----------------------|
| 15.1% | `BaseModel.__init__`                     | pydantic main.py:263 |
|  7.1% | `BaseModel.model_copy`                   | pydantic main.py:424 |
|  3.3% | `copy.copy`                              | stdlib copy.py:76    |
|  3.3% | `to_jsonable_python`                     | pydantic_core        |
|  2.9% | `BaseModel.__eq__`                       | pydantic main.py:1194|
|  1.9% | `uuid4`                                  | uuid.py:723          |
|  1.8% | `model_dump`                             | pydantic main.py:475 |

By inclusive time, the dominant single non-event-loop frame is
`condense_sample` at **21.5%** (`src/inspect_ai/log/_condense.py:138`),
followed by its children `walk_events`, `walk_event`, `walk_model_event`,
`walk_chat_messages` (all in the 20-22% range).

## Findings (revised priorities vs. the original plan)

### #1 — `get_model_info()` is O(N) for any model not in the static DB

`src/inspect_ai/model/_model_info.py:244-310`. `get_model_info()` is called
once per turn from `record_and_check_model_usage`. The lookup path is:

1. Direct dict hit.
2. Normalized-key index hit.
3. **Fuzzy match across the entire 266-entry DB** (`_fuzzy_match` at line 184)
   — for *every* DB entry, computes `_extract_model_name` +
   `_normalize_for_fuzzy` (regex) + `_compute_match_score`.
4. Falls back to instantiating a real `Model` via
   `get_model(model, api_key="__model_info_lookup__")` and re-running the
   lookup with the canonical name (which can hit the fuzzy path *again*).

Microbenchmark (`get_model_info` only):

| input                                 | per-call |
|---------------------------------------|----------|
| `openai/gpt-4o` (in DB)                | 1.1µs    |
| `anthropic/claude-3-5-sonnet-…`        | 1.1µs    |
| `openai/some-fine-tuned-model` (miss)  | 700µs    |
| `mockllm/model` (miss + provider fallback) | **1358µs** |

Anyone calling Inspect with a model name that isn't in the static DB
(custom proxies, fine-tuned model strings, OpenRouter-style routes,
self-hosted gateways) pays this cost on **every turn**. At 300×5 turns =
1500 calls, this is ~1s of pure Python before any other work.

In my benchmark, registering the mock model removed **22% of wall time**
(6.83s → 5.28s).

**Fix:** add `functools.lru_cache(maxsize=…)` (or a hand-rolled dict cache
keyed on `model: str`) on `get_model_info`. Misses are also stable — they
should be cached as `None`. This is a one-line, fully safe change.

### #2 — `condense_sample()` walks every event × every message at sample completion

`src/inspect_ai/log/_condense.py:114-165` is **21.5% of total CPU** at 30
turns × 30-message history. For each `ModelEvent`, `walk_model_event`
(`:386-398`) does `model_copy(update={...})` on the event, walks every input
message via `walk_chat_messages` (`:510-515`), walks the output choices,
and walks the `ModelCall.request` / `response` JSON.

Each `walk_chat_message` (`:518`) ends with another `model_copy(update=...)`.
Pydantic v2 `model_copy` allocates and re-validates a fresh model instance.

There is a `message_cache` keyed on `message.id` (`:521-525`) that's intended
to dedupe identical messages across events, but the cache hit check is
`hit == message` — a full pydantic `BaseModel.__eq__`, which appears as
**2.9% self time** in the profile. So the cache check itself is non-trivial
even when it hits.

Worst of all, every `ModelEvent` in the transcript carries the *full input
history at that turn*. So an N-turn sample with H-message history walks
roughly N × H messages at log time. That's ~900 message walks per sample at
my settings, × 300 samples = ~270K walks. The condense work scales as
**O(turns² × history)** in the worst case (because each turn's event
contains all prior turns' messages).

**Fixes (any combination):**

- Replace the `hit == message` check with `hit is message` — for the common
  case where the same message object is passed turn after turn, identity
  comparison is sufficient and 100× faster. Fall back to `==` only on
  identity miss, or drop the equality re-check entirely (the cache is keyed
  on the message id, which we already trust elsewhere).
- Skip `model_copy` when content_fn is the identity transform. Many fields
  don't need attachment substitution (e.g. messages where every text segment
  is already < `max_length`). Check first, copy only if a transform actually
  produced a different value.
- Avoid storing the full input history in every `ModelEvent` — emit a delta
  per turn (the new user message and the assistant response) and reconstruct
  the snapshot at read time. This is a larger refactor but eliminates the
  quadratic blow-up entirely.

### #3 — Pydantic construction / copy / equality is everywhere

Together, `BaseModel.__init__` (15.1% self), `model_copy` (7.1%),
`__eq__` (2.9%), `model_dump` (1.8%) account for ~27% of CPU. This isn't a
single hotspot to remove, but two specific places amplify it:

- `_record_model_interaction` at `src/inspect_ai/model/_model.py:1187-1199`
  builds `ModelEvent(...)` (with full validation) for the *pending* event
  on every turn, and then mutates fields and emits `_event_updated`. The
  pending event's inputs are already-validated objects. Use
  `ModelEvent.model_construct(...)` to bypass validation for the pending
  case.
- `condense_sample`'s many `model_copy(update={...})` calls do full
  re-validation of the new instance. For trusted internal walks consider
  `model_construct` plus a one-time validation at the boundary (the
  serialized log already encodes the schema).

### #4 — `deepcopy` of message lists per turn

`src/inspect_ai/model/_model.py:1551-1552` (`simple_input_messages`) and
`:1657` (`resolve_tool_model_input`) do `deepcopy(input)` even when only one
or two messages will actually be mutated. `copy.copy` shows up at 3.3% self
time and `_deepcopy_dict` at 0.6%. Smaller than condense but still real.

**Fix:** replace `deepcopy(input)` with a list copy + targeted
`message.model_copy(update={...})` of just the messages we mutate.

### #5 — Provider-side per-message `model_dump`

`src/inspect_ai/model/_providers/mockllm.py:80-82` does
`[m.model_dump() for m in input]` to build the request dict. `mockllm
provider` is **18.0%** of inclusive time in the profile. Real providers
(`_openai.py`, `_anthropic.py`, etc.) have analogous per-message conversion
loops with awaits sprinkled in. Worth auditing each provider's
`generate()` for hot loops that repeatedly serialize the same messages
(history doesn't change much turn-to-turn — a content-addressed cache
keyed on `message.id` would work).

### #6 — Span/transcript and store change tracking are *not* the dominant cost

This is contrary to my original hypothesis. With a small store, span +
transcript + `track_store_changes` together are **<2%** of CPU. Even at
64 KB of store data they're a rounding error.

They *do* become significant when the store gets very large:

| store size | wall  | cpu/turn |
|------------|-------|----------|
| 0          | 21.3s | 2.36ms   |
| 64 KB      | 21.7s | 2.40ms   |
| 1 MB       | 27.3s | 3.03ms   |

So the original plan's P0 (a store-dirty-counter to skip `track_store_changes`)
is still a sensible cheap optimization — it just isn't where the bulk of the
pain comes from for typical workloads. Keeping it on the list, but
demoted.

## Revised priority list

The original plan ranked store/transcript first; the data says we should
start elsewhere:

1. **Cache `get_model_info`.** One-line `lru_cache`. ~20% wall-time
   improvement for any user whose model isn't in the static DB.
2. **Fix the `walk_chat_message` cache check (`hit is message`).**
   Plausibly a multi-percent improvement for any non-trivial multi-turn
   eval.
3. **`model_construct` for the pending `ModelEvent`** in
   `_record_model_interaction`. Cheap and obviously safe.
4. **Replace `deepcopy(input)` with targeted `model_copy`** in
   `simple_input_messages` and `resolve_tool_model_input`.
5. Audit provider `generate()` per-message conversion loops for redundant
   `model_dump`/serialization across turns.
6. Investigate `condense_sample`'s O(turns² × history) growth — likely the
   biggest single win on long-running, deep-conversation workloads, but
   the hardest to do safely (changes log on-disk format).
7. Original plan's store-dirty-counter for `track_store_changes` — keep,
   but only meaningful when users put large data in the store.

## How to reproduce

```bash
# Quick CPU/turn measurement
python bench/cpu_concurrency.py --samples 300 --turns 30 --history 30

# Capture a flamegraph
py-spy record --idle --rate 250 -o bench/profiles/flame.svg -- \
    python bench/cpu_concurrency.py --samples 300 --turns 30 --history 30

# Capture a speedscope profile and analyze top frames
py-spy record --idle --rate 250 --format speedscope \
    -o bench/profiles/profile.json -- \
    python bench/cpu_concurrency.py --samples 300 --turns 30 --history 30
python bench/analyze_profile.py bench/profiles/profile.json 30
```

Flamegraphs/speedscope JSON live under `bench/profiles/` and are
gitignored; rerun the commands above to regenerate.
