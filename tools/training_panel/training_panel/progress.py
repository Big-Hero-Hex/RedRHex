"""Parse rsl_rl training progress out of a captured process log.

rsl_rl prints one block per learning iteration (rsl_rl/runners/on_policy_runner.py).
The panel tails the process log and reads the last complete block, which gives
iteration counts, throughput, and ETA without touching TensorBoard event files.
"""

from __future__ import annotations

import re

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
COMPUTATION_RE = re.compile(r"Computation:\s+([0-9.]+)\s+steps/s")
MEAN_REWARD_RE = re.compile(r"^\s*Mean reward:\s+(-?[0-9.]+)\s*$", re.MULTILINE)
EPISODE_LENGTH_RE = re.compile(r"Mean episode length:\s+(-?[0-9.]+)")
TIMESTEPS_RE = re.compile(r"Total timesteps:\s+(\d+)")
ITERATION_TIME_RE = re.compile(r"Iteration time:\s+([0-9.]+)s")
ETA_RE = re.compile(r"ETA:\s+(\d+):(\d{2}):(\d{2})")


def _search_float(pattern: re.Pattern, text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_progress(log_text: str) -> dict | None:
    """Return progress fields from the last complete iteration block, or None.

    A block counts as complete only once its trailing "ETA:" or "Iteration time:"
    line has been flushed, so a half-written tail never yields a truncated reading.
    """
    if not log_text:
        return None
    text = ANSI_RE.sub("", log_text)

    starts = [match for match in ITERATION_RE.finditer(text)]
    if not starts:
        return None

    for match in reversed(starts):
        block = text[match.end():]
        next_match = ITERATION_RE.search(block)
        if next_match:
            block = block[: next_match.start()]
        if "Iteration time:" not in block:
            continue  # block still being written — fall back to the previous one

        iteration = int(match.group(1))
        total = int(match.group(2))
        result: dict = {"iteration": iteration, "total_iterations": total}
        if total > 0:
            result["percent"] = round(min(iteration / total, 1.0) * 100.0, 1)

        steps = _search_float(COMPUTATION_RE, block)
        if steps is not None:
            result["steps_per_second"] = steps
        reward = _search_float(MEAN_REWARD_RE, block)
        if reward is not None:
            result["mean_reward"] = reward
        length = _search_float(EPISODE_LENGTH_RE, block)
        if length is not None:
            result["mean_episode_length"] = length
        timesteps = _search_float(TIMESTEPS_RE, block)
        if timesteps is not None:
            result["total_timesteps"] = int(timesteps)
        iteration_seconds = _search_float(ITERATION_TIME_RE, block)
        if iteration_seconds is not None:
            result["iteration_seconds"] = iteration_seconds

        eta = ETA_RE.search(block)
        if eta:
            hours, minutes, seconds = (int(part) for part in eta.groups())
            result["eta_seconds"] = hours * 3600 + minutes * 60 + seconds
        return result

    return None
