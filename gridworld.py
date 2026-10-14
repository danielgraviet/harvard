"""Fast, CPU-only tabular Q-learning in Farama MiniGrid's Empty room."""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, TypeAlias
from uuid import uuid4

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments
from PIL import Image, ImageDraw, ImageFont


class Action(IntEnum):
    """Actions available in an empty MiniGrid room."""

    LEFT = 0
    RIGHT = 1
    FORWARD = 2


@dataclass(frozen=True, slots=True)
class State:
    """Fully observed agent position and facing direction."""

    x: int
    y: int
    direction: int


@dataclass(frozen=True, slots=True)
class Checkpoint:
    episode: int
    q_table: QTable


@dataclass(slots=True)
class PolicyRollout:
    episode: int
    frames: list[Image.Image]
    positions: list[PathPosition]
    reached_goal: bool
    steps: int


@dataclass(slots=True)
class TrainingResult:
    q_table: QTable
    checkpoints: list[Checkpoint]
    elapsed_seconds: float
    q_state_count: int


QTable: TypeAlias = dict[State, list[float]]
PathPosition: TypeAlias = tuple[int, int]

_ACTION_NAMES = {
    Action.LEFT: "left",
    Action.RIGHT: "right",
    Action.FORWARD: "forward",
}
_ARROWS = ("→", "↓", "←", "↑")
_GRID_SIZE = 8
_EPISODE_LIMIT = 64
DEMO_STARTS = (
    State(1, 1, 1),
    State(1, _GRID_SIZE - 2, 0),
    State(_GRID_SIZE - 2, 1, 1),
)


def _state_of(env: Any) -> State:
    x, y = env.unwrapped.agent_pos
    return State(int(x), int(y), int(env.unwrapped.agent_dir))


def _choose_action(
    q_table: QTable,
    state: State,
    epsilon: float,
    rng: random.Random,
) -> Action:
    if rng.random() < epsilon:
        return Action(rng.randrange(len(Action)))

    values = q_table.get(state, [0.0] * len(Action))
    best_value = max(values)
    best_actions = [
        Action(index)
        for index, value in enumerate(values)
        if value == best_value
    ]
    return rng.choice(best_actions)


def board_for_path(path: list[PathPosition], size: int = _GRID_SIZE) -> str:
    """Render a path as ASCII, rejecting positions outside the room."""
    if size < 3:
        raise ValueError("Grid size must be at least 3.")
    if not path:
        raise ValueError("Path must contain at least one position.")
    if any(not (0 <= x < size and 0 <= y < size) for x, y in path):
        raise ValueError(f"Path positions must be inside a {size}x{size} grid.")

    start = path[0]
    goal = (size - 2, size - 2)
    visited = set(path)
    special_cells = {goal: "G", start: "S"}
    for position in visited:
        special_cells.setdefault(position, "·")

    rows: list[str] = []
    for y in range(size):
        row: list[str] = []
        for x in range(size):
            position = (x, y)
            if x in (0, size - 1) or y in (0, size - 1):
                row.append("#")
                continue
            row.append(special_cells.get(position, " "))
        rows.append("".join(row))
    return "\n".join(rows)


def _set_start(env: Any, start: State) -> None:
    if not (1 <= start.x < _GRID_SIZE - 1 and 1 <= start.y < _GRID_SIZE - 1):
        raise ValueError(f"Start position must be inside the room: {start}")
    if not 0 <= start.direction < len(_ARROWS):
        raise ValueError(f"Unknown direction {start.direction}; expected 0–3.")
    base = env.unwrapped
    base.agent_pos = (start.x, start.y)
    base.agent_dir = start.direction


def _greedy_action(q_table: QTable, state: State) -> Action:
    values = q_table.get(state, [0.0] * len(Action))
    return Action(max(range(len(Action)), key=values.__getitem__))


def render_episode(
    env: Any,
    q_table: QTable,
    start: State,
    max_steps: int = 24,
) -> list[str]:
    """Run the greedy policy and return an action trace and ASCII board."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    env.reset()
    _set_start(env, start)

    trace = [f"start ({start.x}, {start.y}) facing {_ARROWS[start.direction]}"]
    path = [(start.x, start.y)]
    for _ in range(max_steps):
        action = _greedy_action(q_table, _state_of(env))
        _, reward, terminated, truncated, _ = env.step(int(action))
        trace.append(_ACTION_NAMES[action])

        position = tuple(env.unwrapped.agent_pos)
        if position != path[-1]:
            path.append(position)
        if terminated:
            trace.append(f"goal reached (reward {reward:.2f})")
            break
        if truncated:
            trace.append("time limit")
            break

    trace.append("\n" + board_for_path(path))
    return trace


def train(
    episodes: int = 2000,
    seed: int = 7,
    alpha: float = 0.35,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.04,
    report: bool = True,
) -> TrainingResult:
    """Train vanilla tabular Q-learning and return its table and checkpoints."""
    if episodes < 1:
        raise ValueError("episodes must be positive.")

    rng = random.Random(seed)
    env = gym.make(
        f"MiniGrid-Empty-{_GRID_SIZE}x{_GRID_SIZE}-v0",
        agent_start_pos=None,
        max_steps=_EPISODE_LIMIT,
    )
    env.reset(seed=seed)
    q_table: QTable = {}
    checkpoints: list[Checkpoint] = []
    recent_successes: deque[bool] = deque(maxlen=25)
    checkpoint_interval = max(episodes // 6, 1)
    started = time.perf_counter()

    try:
        for episode_index in range(episodes):
            env.reset()
            progress = episode_index / max(episodes - 1, 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
            state = _state_of(env)
            succeeded = False

            for _ in range(_EPISODE_LIMIT):
                action = _choose_action(q_table, state, epsilon, rng)
                _, reward, terminated, truncated, _ = env.step(int(action))
                next_state = _state_of(env)

                values = q_table.setdefault(state, [0.0] * len(Action))
                next_best = max(q_table.get(next_state, [0.0] * len(Action)))
                terminal = terminated or truncated
                target = reward if terminal else reward + gamma * next_best
                values[action] += alpha * (target - values[action])
                state = next_state

                if terminal:
                    succeeded = bool(terminated)
                    break

            recent_successes.append(succeeded)
            episode = episode_index + 1
            is_checkpoint = episode % checkpoint_interval == 0 or episode == episodes
            if report and (episode == 1 or is_checkpoint):
                success_rate = sum(recent_successes) / len(recent_successes)
                print(
                    f"episode {episode:>4}/{episodes}  "
                    f"recent success {success_rate:>5.0%}  "
                    f"epsilon {epsilon:>4.2f}  "
                    f"Q states {len(q_table):>2}",
                    flush=True,
                )
            if episode == 1 or is_checkpoint:
                snapshot = {state: values.copy() for state, values in q_table.items()}
                checkpoints.append(Checkpoint(episode, snapshot))
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    if report:
        print(f"Training finished in {elapsed:.3f}s on CPU (no neural network).")
        print("\nGreedy policy on a few starting positions:")
        evaluation_env = gym.make(
            f"MiniGrid-Empty-{_GRID_SIZE}x{_GRID_SIZE}-v0",
            max_steps=_EPISODE_LIMIT,
        )
        try:
            for start in DEMO_STARTS:
                trace = render_episode(evaluation_env, q_table, start)
                actions, board = trace[:-1], trace[-1]
                print(f"  ({start.x}, {start.y}): " + " → ".join(actions))
                print("\n".join(f"    {line}" for line in board.splitlines()))
        finally:
            evaluation_env.close()

    return TrainingResult(q_table, checkpoints, elapsed, len(q_table))


def _annotated_frame(frame: Any, caption: str) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 34), fill=(18, 27, 40))
    draw.text(
        (10, 7),
        caption,
        fill="white",
        font=ImageFont.load_default(size=18),
    )
    return image


def _policy_frames(
    env: Any,
    q_table: QTable,
    start: State,
    caption: str,
) -> list[Image.Image]:
    return _policy_rollout(env, q_table, start, caption, episode=0).frames


def _policy_rollout(
    env: Any,
    q_table: QTable,
    start: State,
    caption: str,
    episode: int,
) -> PolicyRollout:
    env.reset()
    _set_start(env, start)
    frames = [_annotated_frame(env.render(), caption)]
    positions = [(start.x, start.y)]
    reached_goal = False
    steps = 0

    for step in range(_EPISODE_LIMIT):
        action = _greedy_action(q_table, _state_of(env))
        _, _, terminated, truncated, _ = env.step(int(action))
        steps = step + 1
        position = tuple(env.unwrapped.agent_pos)
        if position != positions[-1]:
            positions.append(position)
        frames.append(_annotated_frame(env.render(), caption))
        if terminated:
            reached_goal = True
        if terminated or truncated:
            frames.extend([frames[-1]] * 3)
            break
    return PolicyRollout(episode, frames, positions, reached_goal, steps)


def policy_filename(start: State, run_id: str) -> str:
    return f"policy-{run_id}-start-{start.x}-{start.y}.gif"


def learning_filename(run_id: str) -> str:
    return f"learning-progress-{run_id}.gif"


def failure_filename(run_id: str) -> str:
    return f"failed-policy-example-{run_id}.gif"


def _select_failed_rollout(env: Any, checkpoints: list[Checkpoint]) -> PolicyRollout:
    """Find a real checkpoint policy that advances, then fails to reach goal."""
    goal = (_GRID_SIZE - 2, _GRID_SIZE - 2)
    best_rollout: PolicyRollout | None = None
    best_score = float("-inf")

    for checkpoint in checkpoints:
        for start in DEMO_STARTS:
            caption = (
                f"Episode {checkpoint.episode}: from ({start.x}, {start.y}), "
                "moves toward goal, then gets stuck"
            )
            rollout = _policy_rollout(
                env,
                checkpoint.q_table,
                start,
                caption,
                episode=checkpoint.episode,
            )
            if rollout.reached_goal or len(rollout.positions) < 4:
                continue

            distances = [abs(x - goal[0]) + abs(y - goal[1]) for x, y in rollout.positions]
            improvement = distances[0] - min(distances)
            if improvement < 1:
                continue

            unique_positions = len(set(rollout.positions))
            score = improvement * 12 + unique_positions * 3 - distances[-1]
            if score > best_score:
                best_score = score
                best_rollout = rollout

    if best_rollout is None:
        raise RuntimeError(
            "No failed checkpoint rollout moved toward the goal; "
            "increase --episodes to capture a useful learning example."
        )
    return best_rollout


def export_gifs(
    result: TrainingResult,
    output_dir: Path,
    run_id: str | None = None,
) -> list[Path]:
    """Save training progress and policy rollouts as run-unique GIF files."""
    run_id = run_id or uuid4().hex[:10]
    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        f"MiniGrid-Empty-{_GRID_SIZE}x{_GRID_SIZE}-v0",
        render_mode="rgb_array",
        tile_size=64,
        max_steps=_EPISODE_LIMIT,
    )
    outputs: list[Path] = []
    try:
        failed_rollout = _select_failed_rollout(env, result.checkpoints)
        goal = (_GRID_SIZE - 2, _GRID_SIZE - 2)
        closest_distance = min(
            abs(x - goal[0]) + abs(y - goal[1])
            for x, y in failed_rollout.positions
        )
        path_text = " → ".join(
            f"({x}, {y})" for x, y in failed_rollout.positions
        )
        print(
            f"Failed-policy example: episode {failed_rollout.episode}; "
            f"path {path_text}; closest to goal: {closest_distance} tiles"
        )
        failure_path = output_dir / failure_filename(run_id)
        _save_gif(failed_rollout.frames, failure_path, duration=180)
        outputs.append(failure_path)

        progress_frames: list[Image.Image] = failed_rollout.frames.copy()
        for checkpoint in result.checkpoints:
            if checkpoint.episode <= failed_rollout.episode:
                continue
            progress_start = DEMO_STARTS[0]
            progress_frames.extend(
                _policy_frames(
                    env,
                    checkpoint.q_table,
                    progress_start,
                    f"Training episode {checkpoint.episode}",
                )
            )

        learning_path = output_dir / learning_filename(run_id)
        _save_gif(progress_frames, learning_path, duration=140)
        outputs.append(learning_path)

        for start in DEMO_STARTS:
            frames = _policy_frames(
                env,
                result.q_table,
                start,
                f"Learned policy · start ({start.x}, {start.y})",
            )
            output_path = output_dir / policy_filename(start, run_id)
            _save_gif(frames, output_path, duration=180)
            outputs.append(output_path)
    finally:
        env.close()
    return outputs


def _save_gif(frames: list[Image.Image], output_path: Path, duration: int) -> None:
    if not frames:
        raise ValueError("Cannot save a GIF without frames.")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
