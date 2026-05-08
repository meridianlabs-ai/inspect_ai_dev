"""Summarize a py-spy speedscope profile by self-time and inclusive-time.

Usage:
    python bench/analyze_profile.py bench/profiles/profile_300x5x30.json [N]

Reports:
- Top N functions by self time
- Top N functions by inclusive (cumulative) time
- Aggregated time inside specific subsystems we care about
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def frame_label(frame: dict) -> str:
    name = frame.get("name", "?")
    file = frame.get("file", "")
    short = file.split("/")[-1] if file else ""
    return f"{name} ({short})" if short else name


def categorize(frame: dict) -> str | None:
    file = (frame.get("file") or "").replace("\\", "/")
    name = frame.get("name", "")
    if "/inspect_ai/util/_span.py" in file or "/inspect_ai/log/_transcript.py" in file:
        return "span/transcript"
    if "/inspect_ai/util/_store.py" in file:
        return "store"
    if "/inspect_ai/model/_model.py" in file:
        return "model layer"
    if "/inspect_ai/model/_openai.py" in file or "/inspect_ai/model/_anthropic.py" in file:
        return "provider message conversion"
    if "/inspect_ai/model/_providers/mockllm.py" in file:
        return "mockllm provider"
    if "/inspect_ai/_eval/" in file:
        return "eval pipeline"
    if "/inspect_ai/log/_recorders/" in file or "/inspect_ai/log/_log.py" in file:
        return "log recorder"
    if "/pydantic/" in file or "pydantic_core" in name or "pydantic_core" in file:
        return "pydantic"
    if "/anyio/" in file:
        return "anyio"
    if "/asyncio/" in file:
        return "asyncio"
    if "/copy.py" in file:
        return "copy.deepcopy"
    if "tiktoken" in file:
        return "tiktoken"
    if "json" in file and "/json/" in file:
        return "stdlib json"
    return None


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "bench/profiles/profile_300x5x30.json"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    with open(path) as f:
        data = json.load(f)

    frames = data["shared"]["frames"]
    profile = data["profiles"][0]
    samples = profile["samples"]
    weights = profile["weights"]

    self_time: dict[int, float] = defaultdict(float)
    inclusive_time: dict[int, float] = defaultdict(float)
    category_time: dict[str, float] = defaultdict(float)
    total = 0.0

    for stack, w in zip(samples, weights):
        if not stack:
            continue
        total += w
        # leaf = self time
        self_time[stack[-1]] += w
        # any frame in the stack = inclusive time
        seen = set()
        for fi in stack:
            if fi not in seen:
                seen.add(fi)
                inclusive_time[fi] += w
        # category time: count this sample once per category present in stack
        cats = set()
        for fi in stack:
            cat = categorize(frames[fi])
            if cat is not None:
                cats.add(cat)
        for c in cats:
            category_time[c] += w

    print(f"total sampled time: {total:.2f}s   ({len(samples)} samples)")
    print()

    print(f"=== top {top_n} by SELF time ===")
    for fi, t in sorted(self_time.items(), key=lambda x: -x[1])[:top_n]:
        f = frames[fi]
        pct = 100 * t / total
        loc = f"{(f.get('file') or '').split('/')[-1]}:{f.get('line') or '?'}"
        print(f"  {pct:5.1f}%  {t:6.2f}s  {f.get('name','?'):40.40}  {loc}")
    print()

    print(f"=== top {top_n} by INCLUSIVE time ===")
    for fi, t in sorted(inclusive_time.items(), key=lambda x: -x[1])[:top_n]:
        f = frames[fi]
        pct = 100 * t / total
        loc = f"{(f.get('file') or '').split('/')[-1]}:{f.get('line') or '?'}"
        print(f"  {pct:5.1f}%  {t:6.2f}s  {f.get('name','?'):40.40}  {loc}")
    print()

    print("=== inclusive time by SUBSYSTEM (stack appears anywhere in sample) ===")
    for cat, t in sorted(category_time.items(), key=lambda x: -x[1]):
        pct = 100 * t / total
        print(f"  {pct:5.1f}%  {t:6.2f}s  {cat}")


if __name__ == "__main__":
    main()
