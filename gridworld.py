"""Fast, CPU-only tabular Q-learning in Farama MiniGrid's DoorKey maze."""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, TypeAlias
from uuid import uuid4

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments
from PIL import Image, ImageDraw, ImageFont


class Action(IntEnum):
    """Actions used to navigate and unlock the MiniGrid maze."""

    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    TOGGLE = 5


_ACTIONS = (Action.LEFT, Action.RIGHT, Action.FORWARD, Action.PICKUP, Action.TOGGLE)
_ACTION_INDEX = {action: index for index, action in enumerate(_ACTIONS)}


@dataclass(frozen=True, slots=True)
class State:
    """Position, facing, key possession, and door state for the maze."""

    x: int
    y: int
    direction: int
    has_key: bool = False
    door_open: bool = False


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
    seed: int = 7


QTable: TypeAlias = dict[State, list[float]]
PathPosition: TypeAlias = tuple[int, int]

_ACTION_NAMES = {
    Action.LEFT: "left",
    Action.RIGHT: "right",
    Action.FORWARD: "forward",
    Action.PICKUP: "pick up key",
    Action.TOGGLE: "toggle door",
}
_ARROWS = ("→", "↓", "←", "↑")
_GRID_SIZE = 8
_EPISODE_LIMIT = 160
_ENVIRONMENT_SEED = 7
DEMO_STARTS = (
    State(1, 1, 1),
    State(1, _GRID_SIZE - 2, 0),
    State(_GRID_SIZE - 2, 1, 1),
)


def _state_of(env: Any) -> State:
    base = env.unwrapped
    x, y = base.agent_pos
    door_pos = getattr(base, "_qlearning_door_pos", None)
    door = base.grid.get(*door_pos) if door_pos is not None else None
    if door is None or door.type != "door":
        door_pos = next(
            ((dx, dy) for dy in range(base.height) for dx in range(base.width)
             if (obj := base.grid.get(dx, dy)) is not None and obj.type == "door"),
            None,
        )
        base._qlearning_door_pos = door_pos
        door = base.grid.get(*door_pos) if door_pos is not None else None
    carrying = base.carrying
    return State(
        int(x), int(y), int(base.agent_dir),
        has_key=carrying is not None and carrying.type == "key",
        door_open=bool(door is not None and door.is_open),
    )


def _choose_action(
    q_table: QTable,
    state: State,
    epsilon: float,
    rng: random.Random,
) -> Action:
    if rng.random() < epsilon:
        return rng.choice(_ACTIONS)

    values = q_table.get(state, [0.0] * len(_ACTIONS))
    best_value = max(values)
    best_actions = [
        _ACTIONS[index]
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


def _evaluation_starts(env: Any) -> tuple[State, ...]:
    """Choose stable empty cells on both sides of the seeded locked door."""
    base = env.unwrapped
    door_pos = next(
        ((x, y) for y in range(base.height) for x in range(base.width)
         if (obj := base.grid.get(x, y)) is not None and obj.type == "door"),
        (base.width // 2, base.height // 2),
    )
    empty = [
        (x, y) for y in range(1, base.height - 1) for x in range(1, base.width - 1)
        if base.grid.get(x, y) is None
    ]
    left = [position for position in empty if position[0] < door_pos[0]]
    right = [position for position in empty if position[0] > door_pos[0]]

    def nearest(cells: list[tuple[int, int]], target: tuple[int, int], used: set[tuple[int, int]]) -> tuple[int, int] | None:
        options = [cell for cell in cells if cell not in used]
        return min(options, key=lambda cell: abs(cell[0] - target[0]) + abs(cell[1] - target[1])) if options else None

    chosen: list[tuple[int, int]] = []
    first = nearest(left, (1, 1), set()) or nearest(empty, (1, 1), set())
    if first:
        chosen.append(first)
    second = nearest(left, (1, base.height - 2), set(chosen)) or nearest(empty, (1, base.height - 2), set(chosen))
    if second:
        chosen.append(second)
    third = nearest(right, (base.width - 2, 1), set(chosen)) or nearest(empty, (base.width - 2, 1), set(chosen))
    if third:
        chosen.append(third)
    while len(chosen) < 3:
        option = nearest(empty, (base.width - 2, base.height - 2), set(chosen))
        if option is None:
            break
        chosen.append(option)
    return tuple(State(x, y, direction) for (x, y), direction in zip(chosen, (1, 0, 2)))


def evaluation_starts(seed: int = _ENVIRONMENT_SEED) -> tuple[State, ...]:
    """Return showcase positions in the shared, fixed maze layout."""
    env = gym.make(
        f"MiniGrid-DoorKey-{_GRID_SIZE}x{_GRID_SIZE}-v0",
        max_steps=_EPISODE_LIMIT,
    )
    try:
        env.reset(seed=_ENVIRONMENT_SEED)
        return _evaluation_starts(env)
    finally:
        env.close()


def _greedy_action(q_table: QTable, state: State, rng: random.Random) -> Action:
    values = q_table.get(state, [0.0] * len(_ACTIONS))
    best_value = max(values)
    best_actions = [
        _ACTIONS[index]
        for index, value in enumerate(values)
        if value == best_value
    ]
    return rng.choice(best_actions)


def _random_start(env: Any, rng: random.Random) -> State:
    """Sample an empty cell in the key's connected room and a facing."""
    base = env.unwrapped
    key_position = next(
        ((x, y) for y in range(1, base.height - 1) for x in range(1, base.width - 1)
         if (obj := base.grid.get(x, y)) is not None and obj.type == "key"),
        None,
    )
    if key_position is None:
        raise ValueError("The DoorKey maze does not contain a key.")

    # Flood-fill from the key without crossing walls or the locked door. This
    # keeps every episode on the key's side, even if the maze layout changes.
    key_side = {key_position}
    frontier = deque([key_position])
    while frontier:
        x, y = frontier.popleft()
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            nx, ny = neighbor
            if not (1 <= nx < base.width - 1 and 1 <= ny < base.height - 1):
                continue
            if neighbor in key_side:
                continue
            if base.grid.get(nx, ny) is None:
                key_side.add(neighbor)
                frontier.append(neighbor)

    empty = [
        position for position in key_side
        if base.grid.get(*position) is None
    ]
    if not empty:
        raise ValueError("The key's room does not contain an empty learner start cell.")
    x, y = rng.choice(empty)
    return State(x, y, rng.randrange(4))


def render_episode(
    env: Any,
    q_table: QTable,
    start: State,
    max_steps: int = 24,
    seed: int | None = None,
) -> list[str]:
    """Run the greedy policy and return an action trace and ASCII board."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    env.reset(seed=_ENVIRONMENT_SEED)
    _set_start(env, start)
    rng = random.Random(seed)

    trace = [f"start ({start.x}, {start.y}) facing {_ARROWS[start.direction]}"]
    path = [(start.x, start.y)]
    for _ in range(max_steps):
        action = _greedy_action(q_table, _state_of(env), rng)
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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainingResult:
    """Train vanilla tabular Q-learning and return its table and checkpoints."""
    if episodes < 1:
        raise ValueError("episodes must be positive.")

    rng = random.Random(seed)
    env = gym.make(
        f"MiniGrid-DoorKey-{_GRID_SIZE}x{_GRID_SIZE}-v0",
        max_steps=_EPISODE_LIMIT,
    )
    env.reset(seed=_ENVIRONMENT_SEED)
    q_table: QTable = {}
    checkpoints: list[Checkpoint] = []
    recent_successes: deque[bool] = deque(maxlen=25)
    checkpoint_interval = max(episodes // 6, 1)
    started = time.perf_counter()

    try:
        for episode_index in range(episodes):
            env.reset(seed=_ENVIRONMENT_SEED)
            _set_start(env, _random_start(env, rng))
            progress = episode_index / max(episodes - 1, 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
            state = _state_of(env)
            episode_start = [int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1])]
            episode_path = [episode_start]
            succeeded = False
            episode = episode_index + 1
            trace_interval = max(episodes // 8, 1)
            trace_episode = episode == 1 or episode % trace_interval == 0 or episode == episodes

            for step_index in range(_EPISODE_LIMIT):
                action = _choose_action(q_table, state, epsilon, rng)
                _, reward, terminated, truncated, _ = env.step(int(action))
                next_state = _state_of(env)
                position = [int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1])]
                if position != episode_path[-1]:
                    episode_path.append(position)

                values = q_table.setdefault(state, [0.0] * len(_ACTIONS))
                next_best = max(q_table.get(next_state, [0.0] * len(_ACTIONS)))
                terminal = terminated or truncated
                target = reward if terminal else reward + gamma * next_best
                action_index = _ACTION_INDEX[action]
                values[action_index] += alpha * (target - values[action_index])
                state = next_state

                if progress_callback and trace_episode and (
                    step_index % 8 == 7 or terminated or truncated
                ):
                    geometry = _world_geometry(env)
                    progress_callback({
                        "episode": episode,
                        "episodes": episodes,
                        "success_rate": sum(recent_successes) / max(len(recent_successes), 1),
                        "epsilon": epsilon,
                        "states": len(q_table),
                        "live_rollout": {
                            "start": episode_start,
                            "path": episode_path.copy(),
                            "reached_goal": bool(terminated),
                            "steps": step_index + 1,
                            "facing": int(env.unwrapped.agent_dir),
                            **geometry,
                        },
                    })

                if terminal:
                    succeeded = bool(terminated)
                    break

            recent_successes.append(succeeded)
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
            if progress_callback and (episode == 1 or is_checkpoint):
                progress_callback({
                    "episode": episode,
                    "episodes": episodes,
                    "success_rate": sum(recent_successes) / len(recent_successes),
                    "epsilon": epsilon,
                    "states": len(q_table),
                    "live_rollout": {
                        "start": episode_start,
                        "path": episode_path.copy(),
                        "reached_goal": succeeded,
                        "steps": step_index + 1,
                        "facing": int(env.unwrapped.agent_dir),
                        **_world_geometry(env),
                    },
                    "rollouts": evaluate_policy(q_table, seed=seed),
                })
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
            f"MiniGrid-DoorKey-{_GRID_SIZE}x{_GRID_SIZE}-v0",
            max_steps=_EPISODE_LIMIT,
        )
        try:
            evaluation_env.reset(seed=_ENVIRONMENT_SEED)
            evaluation_starts = _evaluation_starts(evaluation_env)
            for start in evaluation_starts:
                trace = render_episode(evaluation_env, q_table, start, seed=_ENVIRONMENT_SEED)
                actions, board = trace[:-1], trace[-1]
                print(f"  ({start.x}, {start.y}): " + " → ".join(actions))
                print("\n".join(f"    {line}" for line in board.splitlines()))
        finally:
            evaluation_env.close()

    return TrainingResult(q_table, checkpoints, elapsed, len(q_table), seed)


def _world_geometry(env: Any) -> dict[str, Any]:
    base = env.unwrapped
    walls: list[list[int]] = []
    door: dict[str, Any] | None = None
    key: list[int] | None = None
    goal: list[int] | None = None
    for y in range(base.height):
        for x in range(base.width):
            obj = base.grid.get(x, y)
            if obj is None:
                continue
            if obj.type == "wall" and 0 < x < base.width - 1 and 0 < y < base.height - 1:
                walls.append([x, y])
            elif obj.type == "door":
                door = {"position": [x, y], "open": bool(obj.is_open), "locked": bool(obj.is_locked)}
            elif obj.type == "key":
                key = [x, y]
            elif obj.type == "goal":
                goal = [x, y]
    return {"walls": walls, "door": door, "key": key, "goal": goal}


def evaluate_policy(q_table: QTable, seed: int = 7) -> list[dict[str, Any]]:
    """Evaluate the greedy policy from the showcase starts for the web UI."""
    env = gym.make(
        f"MiniGrid-DoorKey-{_GRID_SIZE}x{_GRID_SIZE}-v0",
        max_steps=_EPISODE_LIMIT,
    )
    results: list[dict[str, Any]] = []
    rng = random.Random(seed)
    try:
        env.reset(seed=_ENVIRONMENT_SEED)
        starts = _evaluation_starts(env)
        for start in starts:
            env.reset(seed=_ENVIRONMENT_SEED)
            _set_start(env, start)
            initial_geometry = _world_geometry(env)
            path = [[start.x, start.y]]
            reached_goal = False
            steps = 0
            for step in range(_EPISODE_LIMIT):
                action = _greedy_action(q_table, _state_of(env), rng)
                _, _, terminated, truncated, _ = env.step(int(action))
                raw_position = env.unwrapped.agent_pos
                position = [int(raw_position[0]), int(raw_position[1])]
                if position != path[-1]:
                    path.append(position)
                steps = step + 1
                if terminated:
                    reached_goal = True
                if terminated or truncated:
                    break
            geometry = _world_geometry(env)
            geometry["key"] = initial_geometry["key"]
            results.append({
                "start": [start.x, start.y],
                "path": path,
                "reached_goal": reached_goal,
                "steps": steps,
                **geometry,
            })
    finally:
        env.close()
    return results


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
    seed: int | None = None,
) -> list[Image.Image]:
    return _policy_rollout(env, q_table, start, caption, episode=0, seed=_ENVIRONMENT_SEED).frames


def _policy_rollout(
    env: Any,
    q_table: QTable,
    start: State,
    caption: str,
    episode: int,
    seed: int | None = None,
) -> PolicyRollout:
    env.reset(seed=_ENVIRONMENT_SEED)
    _set_start(env, start)
    rng = random.Random(seed)
    frames = [_annotated_frame(env.render(), caption)]
    positions = [(start.x, start.y)]
    reached_goal = False
    steps = 0

    for step in range(_EPISODE_LIMIT):
        action = _greedy_action(q_table, _state_of(env), rng)
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


def _select_failed_rollout(env: Any, checkpoints: list[Checkpoint], seed: int = 7) -> PolicyRollout:
    """Find a real checkpoint policy that advances, then fails to reach goal."""
    goal = (_GRID_SIZE - 2, _GRID_SIZE - 2)
    best_rollout: PolicyRollout | None = None
    best_score = float("-inf")

    env.reset(seed=_ENVIRONMENT_SEED)
    starts = _evaluation_starts(env)
    for checkpoint in checkpoints:
        for start in starts:
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
                seed=_ENVIRONMENT_SEED,
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
        f"MiniGrid-DoorKey-{_GRID_SIZE}x{_GRID_SIZE}-v0",
        render_mode="rgb_array",
        tile_size=64,
        max_steps=_EPISODE_LIMIT,
    )
    outputs: list[Path] = []
    try:
        failed_rollout = _select_failed_rollout(env, result.checkpoints, result.seed)
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
            env.reset(seed=_ENVIRONMENT_SEED)
            progress_start = _evaluation_starts(env)[0]
            progress_frames.extend(
                _policy_frames(
                    env,
                    checkpoint.q_table,
                    progress_start,
                    f"Training episode {checkpoint.episode}",
                    seed=_ENVIRONMENT_SEED,
                )
            )

        learning_path = output_dir / learning_filename(run_id)
        _save_gif(progress_frames, learning_path, duration=140)
        outputs.append(learning_path)

        env.reset(seed=_ENVIRONMENT_SEED)
        for start in _evaluation_starts(env):
            frames = _policy_frames(
                env,
                result.q_table,
                start,
                f"Learned policy · start ({start.x}, {start.y})",
                seed=_ENVIRONMENT_SEED,
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
