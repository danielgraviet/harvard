"""Command-line entry point for the MiniGrid Daytona workshop demo."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from gridworld import export_gifs, train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "daytona"), default="local")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent learners to run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of simultaneous learners (default: 4).",
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.04)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "display",
    )
    parser.add_argument(
        "--run-id",
        help="Optional asset identifier; generated automatically when omitted.",
    )
    parser.add_argument("--keep-sandbox", action="store_true")
    return parser


async def _run_batch(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")

    if args.backend == "daytona":
        load_dotenv()
        from daytona_runner import run_daytona

    semaphore = asyncio.Semaphore(args.concurrency)
    base_run_id = args.run_id or uuid4().hex[:10]

    async def run_one(index: int) -> list[Path]:
        run_id = base_run_id if args.runs == 1 else f"{base_run_id}-{index + 1:04d}"
        async with semaphore:
            print(f"Starting learner {index + 1}/{args.runs} (run {run_id})")
            if args.backend == "daytona":
                return await asyncio.to_thread(
                    run_daytona,
                    project_dir=Path(__file__).resolve().parent,
                    episodes=args.episodes,
                    seed=args.seed + index,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon_start=args.epsilon_start,
                    epsilon_end=args.epsilon_end,
                    output_dir=args.artifacts_dir,
                    run_id=run_id,
                    keep_sandbox=args.keep_sandbox,
                )

            def run_local() -> list[Path]:
                result = train(
                    episodes=args.episodes,
                    seed=args.seed + index,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon_start=args.epsilon_start,
                    epsilon_end=args.epsilon_end,
                )
                return export_gifs(result, args.artifacts_dir, run_id)

            return await asyncio.to_thread(run_local)

    batches = await asyncio.gather(*(run_one(index) for index in range(args.runs)))
    print("\nGIFs ready for display:")
    for artifacts in batches:
        for artifact in artifacts:
            print(f"  {artifact}")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        asyncio.run(_run_batch(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
