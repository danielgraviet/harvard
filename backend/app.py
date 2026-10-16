from __future__ import annotations

import json
import logging
import asyncio
import os
import queue
import random
import shlex
import statistics
import threading
import time
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from daytona import SessionExecuteRequest

from gridworld import evaluate_policy, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
app = FastAPI(title="Daytona GridWorld Lab", version="0.1.0")
logger = logging.getLogger("gridworld.api")
logger.setLevel(logging.INFO)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
run_lock = threading.Lock()
fleet_lock = asyncio.Lock()
load_dotenv()


class TrainingRequest(BaseModel):
    episodes: int = Field(default=2000, ge=100, le=10000)
    seed: int = Field(default=7, ge=0, le=999999)
    alpha: float = Field(default=0.35, ge=0.01, le=1)
    gamma: float = Field(default=0.95, ge=0, le=1)
    epsilon_start: float = Field(default=1, ge=0, le=1)
    epsilon_end: float = Field(default=0.04, ge=0, le=1)


class FleetRequest(BaseModel):
    mode: str = Field(default="train", pattern="^(train|grid_search)$")
    backend: str = Field(default="local", pattern="^(local|daytona)$")
    workers: int = Field(default=200, ge=1, le=200)
    concurrency: int = Field(default=200, ge=1, le=200)
    episodes: int = Field(default=200, ge=100, le=10000)
    seed: int = Field(default=7, ge=0, le=999999)
    alpha: float = Field(default=0.35, ge=0.01, le=1)
    gamma: float = Field(default=0.95, ge=0, le=1)
    epsilon_start: float = Field(default=1, ge=0, le=1)
    epsilon_end: float = Field(default=0.04, ge=0, le=1)


BEST_PARAMS_PATH = Path(__file__).with_name("best_params.json")
GRID_SEARCH_CONFIGURATIONS = 20
GRID_SEARCH_REPEATS = 10
GRID_SEARCH_RUNS = GRID_SEARCH_CONFIGURATIONS * GRID_SEARCH_REPEATS
GRID_SEARCH_PARAMETER_RANGES = ((0.03, 0.95), (0.5, 0.995), (0.45, 1.0), (0.01, 0.35))
T_CRITICAL_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def grid_search_configurations() -> list[dict[str, float]]:
    """Create a repeatable Latin-hypercube spread over all four UI knobs."""
    rng = random.Random(90210)
    strata: list[list[int]] = []
    for _ in GRID_SEARCH_PARAMETER_RANGES:
        order = list(range(GRID_SEARCH_CONFIGURATIONS))
        rng.shuffle(order)
        strata.append(order)
    configurations = []
    for index in range(GRID_SEARCH_CONFIGURATIONS):
        values = []
        for dimension, (low, high) in enumerate(GRID_SEARCH_PARAMETER_RANGES):
            quantile = (strata[dimension][index] + rng.random()) / GRID_SEARCH_CONFIGURATIONS
            values.append(round(low + (high - low) * quantile, 3))
        configurations.append({
            "alpha": values[0], "gamma": values[1],
            "epsilon_start": values[2], "epsilon_end": values[3],
        })
    return configurations


def grid_search_candidates(seed: int) -> list[dict[str, Any]]:
    configurations = grid_search_configurations()
    return [
        {
            "config_index": config_index,
            "trial": trial,
            "trial_count": GRID_SEARCH_REPEATS,
            "seed": seed + (trial - 1) * 104729,
            "params": params,
        }
        for trial in range(1, GRID_SEARCH_REPEATS + 1)
        for config_index, params in enumerate(configurations)
    ]


def score_candidate(index: int, candidate: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    rollouts = result.get("rollouts", [])
    solved = [rollout for rollout in rollouts if rollout.get("reached_goal")]
    return {
        "index": index,
        "config_index": candidate["config_index"],
        "trial": candidate["trial"],
        "seed": candidate["seed"],
        "params": candidate["params"],
        "success_count": len(solved),
        "evaluation_count": len(rollouts),
        "success_rate": len(solved) / len(rollouts) if rollouts else 0.0,
        "successful_steps_total": sum(rollout["steps"] for rollout in solved),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "q_state_count": result.get("q_state_count"),
    }


def aggregate_candidates(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for trial in trials:
        grouped.setdefault(trial["config_index"], []).append(trial)
    aggregates = []
    for config_index, rows in grouped.items():
        rates = [row["success_rate"] for row in rows]
        rate = statistics.mean(rates)
        deviation = statistics.stdev(rates) if len(rates) > 1 else 0.0
        if len(rates) > 1:
            t_critical = T_CRITICAL_95.get(len(rates) - 1, 1.96)
            margin = t_critical * deviation / sqrt(len(rates))
            interval = [max(0.0, rate - margin), min(1.0, rate + margin)]
        else:
            interval = None
        solved = sum(row["success_count"] for row in rows)
        steps = sum(row["successful_steps_total"] for row in rows)
        aggregates.append({
            "config_index": config_index,
            "params": rows[0]["params"],
            "trial_count": len(rows),
            "success_count": solved,
            "evaluation_count": sum(row["evaluation_count"] for row in rows),
            "success_rate": rate,
            "success_rate_stddev": deviation,
            "confidence_interval_95": interval,
            "average_steps": steps / solved if solved else None,
            "trial_success_rates": rates,
        })
    return aggregates


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, int]:
    average_steps = candidate["average_steps"]
    return (
        -candidate["success_rate"],
        average_steps if average_steps is not None else float("inf"),
        candidate["config_index"],
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    logger.debug("Health check")
    return {"status": "ok", "training": "local CPU"}


@app.get("/api/params/best")
def get_best_params() -> dict[str, Any]:
    if not BEST_PARAMS_PATH.exists():
        return {"best_candidate": None}
    try:
        return json.loads(BEST_PARAMS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read saved grid-search parameters from %s", BEST_PARAMS_PATH)
        return {"best_candidate": None, "error": "Saved parameters could not be read."}


@app.post("/api/train")
def train_agent(request: TrainingRequest) -> StreamingResponse:
    run_id = uuid4().hex[:8]
    if request.epsilon_end > request.epsilon_start:
        from fastapi import HTTPException

        logger.warning(
            "run=%s rejected: epsilon_end %.3f exceeds epsilon_start %.3f",
            run_id,
            request.epsilon_end,
            request.epsilon_start,
        )
        raise HTTPException(422, "epsilon_end must be no greater than epsilon_start")
    if not run_lock.acquire(blocking=False):
        from fastapi import HTTPException

        logger.warning("run=%s rejected: another training run is active", run_id)
        raise HTTPException(409, "A training run is already in progress")

    started_at = time.perf_counter()
    logger.info(
        "run=%s accepted episodes=%d seed=%d alpha=%.3f gamma=%.3f epsilon_start=%.3f epsilon_end=%.3f",
        run_id,
        request.episodes,
        request.seed,
        request.alpha,
        request.gamma,
        request.epsilon_start,
        request.epsilon_end,
    )
    events: queue.Queue[dict[str, Any]] = queue.Queue()

    def report_progress(progress: dict[str, Any]) -> None:
        if "rollouts" in progress:
            logger.info(
                "run=%s progress episode=%d/%d success_recent=%.0f%% states=%d epsilon=%.3f",
                run_id,
                progress["episode"],
                progress["episodes"],
                progress["success_rate"] * 100,
                progress["states"],
                progress["epsilon"],
            )
        events.put({"type": "progress", **progress})

    def run() -> None:
        try:
            result = train(
                episodes=request.episodes,
                seed=request.seed,
                alpha=request.alpha,
                gamma=request.gamma,
                epsilon_start=request.epsilon_start,
                epsilon_end=request.epsilon_end,
                report=False,
                progress_callback=report_progress,
            )
            rollouts = evaluate_policy(result.q_table, seed=request.seed)
            logger.info(
                "run=%s complete elapsed=%.2fs q_states=%d goals=%d/%d",
                run_id,
                result.elapsed_seconds,
                result.q_state_count,
                sum(rollout["reached_goal"] for rollout in rollouts),
                len(rollouts),
            )
            events.put({
                "type": "complete",
                "run_id": run_id,
                "elapsed_seconds": result.elapsed_seconds,
                "q_state_count": result.q_state_count,
                "rollouts": rollouts,
            })
        except Exception as exc:
            logger.exception("run=%s failed after %.2fs", run_id, time.perf_counter() - started_at)
            events.put({"type": "error", "run_id": run_id, "message": str(exc) or type(exc).__name__})
        finally:
            run_lock.release()
            events.put({"type": "end"})

    threading.Thread(target=run, daemon=True).start()

    def stream():
        logger.info("run=%s streaming response opened", run_id)
        yield f"data: {json.dumps({'type': 'started', 'run_id': run_id, 'episodes': request.episodes})}\n\n"
        try:
            while True:
                event = events.get()
                if event["type"] == "end":
                    break
                try:
                    encoded = json.dumps(event)
                except (TypeError, ValueError):
                    logger.exception("run=%s could not serialize an SSE event", run_id)
                    encoded = json.dumps({
                        "type": "error",
                        "run_id": run_id,
                        "message": "The server could not serialize a training update. Check the API terminal logs.",
                    })
                    yield f"data: {encoded}\n\n"
                    break
                yield f"data: {encoded}\n\n"
        except Exception:
            logger.exception("run=%s failed while streaming training events", run_id)
            # The HTTP status is already committed; return a final SSE error if
            # the connection is still writable so the browser can show context.
            yield f"data: {json.dumps({'type': 'error', 'run_id': run_id, 'message': 'The training stream failed. Check the API terminal logs.'})}\n\n"
        except GeneratorExit:
            logger.warning("run=%s client disconnected; training continues in background", run_id)
            raise
        finally:
            logger.info("run=%s streaming response closed", run_id)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Run-ID": run_id},
    )


@app.post("/api/fleet")
async def start_fleet(request: FleetRequest) -> StreamingResponse:
    if request.mode == "grid_search" and request.backend != "daytona":
        from fastapi import HTTPException

        raise HTTPException(422, "Grid search currently requires the Daytona backend")
    if request.epsilon_end > request.epsilon_start:
        from fastapi import HTTPException

        raise HTTPException(422, "epsilon_end must be no greater than epsilon_start")
    if request.backend == "daytona" and not os.getenv("DAYTONA_API_KEY"):
        from fastapi import HTTPException

        raise HTTPException(400, "DAYTONA_API_KEY is missing from the API server environment or .env file")
    if fleet_lock.locked():
        from fastapi import HTTPException

        raise HTTPException(409, "A fleet run is already active")

    await fleet_lock.acquire()
    run_id = uuid4().hex[:8]
    search_candidates = grid_search_candidates(request.seed) if request.mode == "grid_search" else None
    run_count = len(search_candidates) if search_candidates is not None else request.workers
    run_concurrency = min(request.concurrency, run_count)
    events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    started_at = time.perf_counter()
    completed_candidates: list[dict[str, Any]] = []

    def params_for(index: int) -> dict[str, float]:
        if search_candidates is not None:
            return search_candidates[index]["params"]
        return {"alpha": request.alpha, "gamma": request.gamma,
                "epsilon_start": request.epsilon_start, "epsilon_end": request.epsilon_end}

    def seed_for(index: int) -> int:
        return search_candidates[index]["seed"] if search_candidates is not None else request.seed + index

    def emit(event: dict[str, Any]) -> None:
        events.put_nowait(event)

    def emit_from_thread(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(emit, event)

    async def run_local_worker(index: int, semaphore: asyncio.Semaphore) -> bool:
        params = params_for(index)
        emit({"type": "grid", "index": index, "state": "queued", "backend": "local"})
        await semaphore.acquire()
        emit({"type": "grid", "index": index, "state": "training", "backend": "local"})
        succeeded = False
        try:
            def report(progress: dict[str, Any]) -> None:
                emit_from_thread({"type": "grid", "index": index, "state": "training", "backend": "local", "progress": progress})

            result = await asyncio.to_thread(
                train,
                episodes=request.episodes,
                seed=seed_for(index),
                alpha=params["alpha"],
                gamma=params["gamma"],
                epsilon_start=params["epsilon_start"],
                epsilon_end=params["epsilon_end"],
                report=False,
                progress_callback=report,
            )
            rollouts = evaluate_policy(result.q_table, seed=request.seed + index)
            emit({
                "type": "grid",
                "index": index,
                "state": "complete",
                "backend": "local",
                "result": {
                    "elapsed_seconds": result.elapsed_seconds,
                    "q_state_count": result.q_state_count,
                    "rollouts": rollouts,
                },
            })
            succeeded = True
        except Exception as exc:
            logger.exception("fleet=%s grid=%03d local worker failed", run_id, index)
            emit({"type": "grid", "index": index, "state": "failed", "backend": "local", "error": str(exc) or type(exc).__name__})
        finally:
            semaphore.release()
        return succeeded

    async def run_daytona_worker(index: int, semaphore: asyncio.Semaphore, client: Any) -> bool:
        params = params_for(index)
        emit({"type": "grid", "index": index, "state": "queued", "backend": "daytona"})
        await semaphore.acquire()
        sandbox = None
        result: dict[str, Any] | None = None
        failure: str | None = None
        sandbox_id: str | None = None
        candidate = search_candidates[index] if search_candidates is not None else None
        grid_key = {
            "type": "grid", "index": index, "backend": "daytona", "params": params,
            "config_index": candidate["config_index"] if candidate else None,
            "trial": candidate["trial"] if candidate else None,
            "trial_count": candidate["trial_count"] if candidate else None,
        }
        try:
            logger.info("fleet=%s grid=%03d creating Daytona sandbox", run_id, index)
            emit({**grid_key, "state": "creating"})
            fleet_snapshot = os.getenv("DAYTONA_FLEET_SNAPSHOT")
            if fleet_snapshot:
                from daytona import CreateSandboxFromSnapshotParams

                sandbox = await client.create(
                    CreateSandboxFromSnapshotParams(snapshot=fleet_snapshot),
                    timeout=120,
                )
            else:
                sandbox = await client.create(timeout=120)
            sandbox_id = sandbox.id
            logger.info("fleet=%s grid=%03d sandbox=%s created", run_id, index, sandbox_id)
            if not fleet_snapshot:
                emit({**grid_key, "state": "installing", "sandbox_id": sandbox_id})
                install_cmd = shlex.join(["python", "-m", "pip", "install", "--quiet", "minigrid==3.1.0", "pillow>=10.0.0"])
                install = await sandbox.process.exec(install_cmd, timeout=300)
                if getattr(install, "exit_code", 0):
                    install_output = str(getattr(install, "result", ""))
                    raise RuntimeError(f"Dependency installation failed: {install_output[-500:]}")
            else:
                logger.info("fleet=%s grid=%03d using prepared snapshot=%s", run_id, index, fleet_snapshot)

            workdir = (await sandbox.get_work_dir()).rstrip("/")
            project_dir = Path(__file__).resolve().parent.parent
            await sandbox.fs.upload_file(str(project_dir / "gridworld.py"), f"{workdir}/gridworld.py")
            await sandbox.fs.upload_file(str(Path(__file__).resolve().parent / "fleet_worker.py"), f"{workdir}/fleet_worker.py")
            session_id = f"fleet-{run_id}-{index:03d}"
            await sandbox.process.create_session(session_id)
            command = shlex.join([
                "python", f"{workdir}/fleet_worker.py",
                "--index", str(index), "--episodes", str(request.episodes),
                "--seed", str(seed_for(index)),
                "--epsilon-end", str(params["epsilon_end"]),
                "--alpha", str(params["alpha"]), "--gamma", str(params["gamma"]),
                "--epsilon-start", str(params["epsilon_start"]),
            ])
            started = await sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=command, run_async=True),
            )
            command_id = started.cmd_id
            logger.info("fleet=%s grid=%03d sandbox=%s training started command=%s", run_id, index, sandbox_id, command_id)
            emit({**grid_key, "state": "training", "sandbox_id": sandbox_id})
            stdout_buffer = ""

            async def on_stdout(chunk: str) -> None:
                nonlocal stdout_buffer, result
                stdout_buffer += chunk
                lines = stdout_buffer.splitlines(keepends=True)
                stdout_buffer = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    stdout_buffer = lines.pop()
                for line in lines:
                    if not line.startswith("FLEET_EVENT "):
                        continue
                    try:
                        worker_event = json.loads(line.removeprefix("FLEET_EVENT "))
                    except json.JSONDecodeError:
                        logger.warning("fleet=%s grid=%03d invalid worker event: %s", run_id, index, line[:300])
                        continue
                    if worker_event.get("type") == "progress":
                        emit({**grid_key, "state": "training", "sandbox_id": sandbox_id, "progress": worker_event.get("progress")})
                    elif worker_event.get("type") == "complete":
                        result = worker_event.get("result")

            await sandbox.process.get_session_command_logs_async(session_id, command_id, on_stdout, lambda chunk: logger.warning("fleet=%s grid=%03d stderr: %s", run_id, index, chunk[-500:]))
            if stdout_buffer.startswith("FLEET_EVENT "):
                try:
                    worker_event = json.loads(stdout_buffer.removeprefix("FLEET_EVENT "))
                    if worker_event.get("type") == "complete":
                        result = worker_event.get("result")
                except json.JSONDecodeError:
                    pass
            command_result = await sandbox.process.get_session_command(session_id, command_id)
            if command_result.exit_code != 0:
                raise RuntimeError(f"Remote training exited with code {command_result.exit_code}")
            if result is None:
                # Recover the final payload if the log websocket connected after
                # a very short worker had already emitted its output.
                logs = await sandbox.process.get_session_command_logs(session_id, command_id)
                for line in logs.stdout.splitlines():
                    if not line.startswith("FLEET_EVENT "):
                        continue
                    try:
                        worker_event = json.loads(line.removeprefix("FLEET_EVENT "))
                    except json.JSONDecodeError:
                        continue
                    if worker_event.get("type") == "complete":
                        result = worker_event.get("result")
                        break
                if result is None:
                    raise RuntimeError("Remote worker finished without a completion event")
            logger.info("fleet=%s grid=%03d sandbox=%s training complete q_states=%s", run_id, index, sandbox_id, result.get("q_state_count"))
        except Exception as exc:
            failure = str(exc) or type(exc).__name__
            logger.exception("fleet=%s grid=%03d sandbox=%s failed", run_id, index, sandbox_id or "pending")
        finally:
            if request.mode == "grid_search" and result is not None:
                completed_candidates.append(score_candidate(index, search_candidates[index], result))
            if sandbox is not None:
                emit({**grid_key, "state": "cleaning", "sandbox_id": sandbox_id, "result": result})
                try:
                    await client.delete(sandbox, timeout=60)
                    logger.info("fleet=%s grid=%03d sandbox=%s deleted", run_id, index, sandbox_id)
                except Exception as exc:
                    logger.exception("fleet=%s grid=%03d sandbox=%s cleanup failed", run_id, index, sandbox_id)
                    failure = f"{failure + '; ' if failure else ''}Sandbox cleanup failed: {exc}"
            final_state = "failed" if failure else "complete"
            emit({**grid_key, "state": final_state, "sandbox_id": sandbox_id, "result": result, "error": failure})
            semaphore.release()
        return failure is None and result is not None

    async def run_fleet() -> None:
        succeeded = 0
        failed = 0
        emit({"type": "fleet_started", "run_id": run_id, "mode": request.mode, "backend": request.backend, "workers": run_count, "concurrency": run_concurrency})
        if request.mode == "grid_search":
            logger.info(
                "grid_search=%s started trials=%d configurations=%d repeats=%d concurrency=%d episodes=%d seed=%d parameter_ranges=%s",
                run_id, run_count, GRID_SEARCH_CONFIGURATIONS, GRID_SEARCH_REPEATS,
                run_concurrency, request.episodes, request.seed,
                GRID_SEARCH_PARAMETER_RANGES,
            )
        else:
            logger.info(
                "fleet=%s started mode=%s backend=%s workers=%d concurrency=%d episodes=%d seed=%d alpha=%.3f gamma=%.3f epsilon_start=%.3f epsilon_end=%.3f",
                run_id, request.mode, request.backend, run_count, run_concurrency, request.episodes,
                request.seed, request.alpha, request.gamma, request.epsilon_start, request.epsilon_end,
            )
        try:
            actual_concurrency = min(run_concurrency, 32) if request.backend == "local" else run_concurrency
            semaphore = asyncio.Semaphore(actual_concurrency)
            if actual_concurrency != run_concurrency:
                logger.info("fleet=%s local worker pool capped at %d threads", run_id, actual_concurrency)
            if request.backend == "local":
                tasks = [run_local_worker(index, semaphore) for index in range(run_count)]
                outcomes = await asyncio.gather(*tasks)
            else:
                from daytona import AsyncDaytona

                async with AsyncDaytona() as client:
                    tasks = [run_daytona_worker(index, semaphore, client) for index in range(run_count)]
                    outcomes = await asyncio.gather(*tasks)
            succeeded = sum(outcomes)
            failed = run_count - succeeded
            logger.info("fleet=%s finished in %.2fs succeeded=%d failed=%d", run_id, time.perf_counter() - started_at, succeeded, failed)
            summary: dict[str, Any] = {
                "type": "fleet_complete", "run_id": run_id, "mode": request.mode,
                "workers": run_count, "succeeded": succeeded, "failed": failed,
                "elapsed_seconds": time.perf_counter() - started_at,
                "episodes": request.episodes,
            }
            if request.mode == "grid_search" and completed_candidates:
                aggregates = aggregate_candidates(completed_candidates)
                best_candidate = min(aggregates, key=candidate_rank)
                saved = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "candidate_count": run_count,
                    "completed_count": len(completed_candidates),
                    "configuration_count": GRID_SEARCH_CONFIGURATIONS,
                    "repeat_count": GRID_SEARCH_REPEATS,
                    "episodes": request.episodes,
                    "seed": request.seed,
                    "best_candidate": best_candidate,
                    "ranked_candidates": sorted(aggregates, key=candidate_rank),
                }
                temp_path = BEST_PARAMS_PATH.with_suffix(".json.tmp")
                temp_path.write_text(json.dumps(saved, indent=2) + "\n")
                temp_path.replace(BEST_PARAMS_PATH)
                logger.info(
                    "grid_search=%s best_candidate=%d success=%d/%d params=%s saved=%s",
                    run_id, best_candidate["config_index"], best_candidate["success_count"],
                    best_candidate["evaluation_count"], best_candidate["params"], BEST_PARAMS_PATH,
                )
                summary["best_candidate"] = best_candidate
                summary["completed_candidates"] = len(completed_candidates)
                summary["completed_configurations"] = len(aggregates)
                summary["configuration_count"] = GRID_SEARCH_CONFIGURATIONS
                summary["repeat_count"] = GRID_SEARCH_REPEATS
            emit(summary)
        except Exception as exc:
            logger.exception("fleet=%s failed", run_id)
            emit({"type": "fleet_error", "run_id": run_id, "message": str(exc) or type(exc).__name__})
        finally:
            fleet_lock.release()
            emit({"type": "end"})

    asyncio.create_task(run_fleet())

    async def stream():
        yield f"data: {json.dumps({'type': 'started', 'run_id': run_id, 'mode': request.mode, 'backend': request.backend, 'workers': run_count})}\n\n"
        while True:
            event = await events.get()
            if event["type"] == "end":
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Run-ID": run_id})
