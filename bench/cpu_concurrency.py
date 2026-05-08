"""CPU benchmark for high concurrent-connections regime.

Goal: reproduce the high-CPU scenario users report at ~300 concurrent model
connections, while bottlenecked on Python (not network or provider latency).

How it works:
- Uses the `mockllm/` provider with a callable `custom_outputs` that returns
  a `ModelOutput` with a pre-populated `usage` (so mockllm itself doesn't run
  `count_tokens` on every call).
- Drives the full eval pipeline (`inspect_ai.eval`) so spans, transcripts,
  store-change tracking, sample bookkeeping all run.
- Each sample's solver runs many turns of `get_model().generate(...)`. Every
  turn passes a fixed-size message history (so per-turn message work is
  realistic) plus a small amount of additional content per turn.

Usage:
    python bench/cpu_concurrency.py [--samples 300] [--turns 5] [--history 20]
                                    [--store-bytes 4096]

For profiling:
    py-spy record --idle --rate 250 -o flame.svg -- \
        python bench/cpu_concurrency.py --samples 300 --turns 8 --history 30
"""

from __future__ import annotations

import argparse
import time

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.model import ModelInfo, set_model_info
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util import store


def _make_history(n_messages: int) -> list[ChatMessage]:
    """Build a synthetic conversation history of ~n_messages messages.

    Alternates user / assistant. Each message has a few hundred chars of
    content so message-conversion / deepcopy / serialization work is
    non-trivial but not pathological.
    """
    body = ("lorem ipsum " * 20).strip()
    msgs: list[ChatMessage] = []
    for i in range(n_messages):
        if i % 2 == 0:
            msgs.append(ChatMessageUser(content=f"[{i}] user: {body}"))
        else:
            msgs.append(ChatMessageAssistant(content=f"[{i}] assistant: {body}"))
    return msgs


def _mock_callable(
    input: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice,
    config: GenerateConfig,
) -> ModelOutput:
    out = ModelOutput.from_content(
        model="mockllm/model",
        content="ok",
    )
    out.usage = ModelUsage(input_tokens=100, output_tokens=2, total_tokens=102)
    return out


@solver
def turn_loop(turns: int, history: int, store_bytes: int):
    """Solver that calls generate() `turns` times with a fixed history size.

    Also seeds the sample store with `store_bytes` of data so that
    `track_store_changes()` has realistic work to do on every span.
    """

    base_history = _make_history(history)
    blob = "x" * store_bytes if store_bytes > 0 else ""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if blob:
            store().set("blob", blob)
            store().set("counter", 0)

        model = get_model()
        for t in range(turns):
            if blob:
                store().set("counter", t)
            messages = list(base_history)
            messages.append(ChatMessageUser(content=f"turn {t}"))
            await model.generate(input=messages, tools=[], cache=False)
        return state

    return solve


def make_task(samples: int, turns: int, history: int, store_bytes: int) -> Task:
    return Task(
        dataset=[Sample(input="bench", target="ok") for _ in range(samples)],
        solver=turn_loop(turns=turns, history=history, store_bytes=store_bytes),
        name="cpu-concurrency-bench",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--store-bytes", type=int, default=4096)
    p.add_argument("--max-connections", type=int, default=None)
    p.add_argument(
        "--register-model",
        action="store_true",
        help="Register mockllm/model in the info DB to bypass the get_model_info "
        "fuzzy-match path (use to isolate other per-turn costs).",
    )
    args = p.parse_args()

    max_conn = args.max_connections or args.samples

    if args.register_model:
        set_model_info(
            "mockllm/model",
            ModelInfo(context_length=128_000, output_tokens=4096, organization="mock"),
        )

    model = get_model(
        "mockllm/model",
        config=GenerateConfig(max_connections=max_conn),
        custom_outputs=_mock_callable,
    )

    total_calls = args.samples * args.turns
    print(
        f"running: samples={args.samples} turns={args.turns} "
        f"history={args.history} store_bytes={args.store_bytes} "
        f"max_connections={max_conn} total_generate_calls={total_calls}"
    )

    t0 = time.perf_counter()
    cpu0 = time.process_time()
    eval(
        make_task(args.samples, args.turns, args.history, args.store_bytes),
        model=model,
        max_samples=args.samples,
        max_subprocesses=1,
        log_dir="bench/.logs",
        display="plain",
    )
    wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0

    print(
        f"done: wall={wall:.2f}s cpu={cpu:.2f}s "
        f"calls/s={total_calls / wall:.0f} "
        f"cpu_per_call_ms={(cpu / total_calls) * 1000:.2f} "
        f"cpu_util={cpu / wall:.2f}"
    )


if __name__ == "__main__":
    main()
