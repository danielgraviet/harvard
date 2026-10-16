"""Small stdout-event worker uploaded to each Daytona fleet sandbox."""

from __future__ import annotations

import argparse
import json

from gridworld import evaluate_policy, train


def emit(event: dict[str, object]) -> None:
    print("FLEET_EVENT " + json.dumps(event), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.04)
    args = parser.parse_args()

    result = train(
        episodes=args.episodes,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        report=False,
        progress_callback=lambda progress: emit({
            "type": "progress",
            "progress": progress,
        }),
    )
    emit({
        "type": "complete",
        "result": {
            "elapsed_seconds": result.elapsed_seconds,
            "q_state_count": result.q_state_count,
            "rollouts": evaluate_policy(result.q_table, seed=args.seed),
        },
    })


if __name__ == "__main__":
    main()
