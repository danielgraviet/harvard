"""Run the GridWorld learner inside a Daytona CPU sandbox."""

from __future__ import annotations

import shlex
from pathlib import Path

from daytona import Daytona

from gridworld import (
    DEMO_STARTS,
    failure_filename,
    learning_filename,
    policy_filename,
)


def run_daytona(
    *,
    project_dir: Path,
    episodes: int,
    seed: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    output_dir: Path,
    run_id: str,
    keep_sandbox: bool,
) -> list[Path]:
    """Upload the learner, run it remotely, and download its GIF assets."""
    client = Daytona()
    sandbox = None
    try:
        print("Creating a Daytona sandbox for CPU-only MiniGrid training…")
        sandbox = client.create()
        print(f"Sandbox: {sandbox.id}  CPU: {sandbox.cpu}  GPU: {sandbox.gpu}")

        install_command = "python -m pip install --quiet " + " ".join(
            shlex.quote(package)
            for package in ("minigrid==3.1.0", "pillow>=10.0.0")
        )
        install = sandbox.process.exec(install_command, timeout=180)
        install_exit_code = getattr(
            install,
            "exit_code",
            getattr(install, "exitCode", 0),
        )
        if install_exit_code:
            raise RuntimeError(f"MiniGrid install failed: {install.result}")

        workdir = sandbox.get_work_dir().rstrip("/")
        remote_main = f"{workdir}/main.py"
        for filename in ("main.py", "gridworld.py"):
            local_path = project_dir / filename
            sandbox.fs.upload_file(local_path.read_bytes(), f"{workdir}/{filename}")

        remote_display = f"{workdir}/display"
        command_parts = [
            "python",
            shlex.quote(remote_main),
            "--backend",
            "local",
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--alpha",
            str(alpha),
            "--gamma",
            str(gamma),
            "--epsilon-start",
            str(epsilon_start),
            "--epsilon-end",
            str(epsilon_end),
            "--artifacts-dir",
            shlex.quote(remote_display),
            "--run-id",
            shlex.quote(run_id),
        ]
        response = sandbox.process.exec(" ".join(command_parts), timeout=180)
        exit_code = getattr(response, "exit_code", getattr(response, "exitCode", 0))
        print(response.result)
        if exit_code:
            raise RuntimeError(f"Demo failed inside the sandbox (exit {exit_code}).")

        output_dir.mkdir(parents=True, exist_ok=True)
        filenames = [failure_filename(run_id), learning_filename(run_id)]
        filenames.extend(policy_filename(start, run_id) for start in DEMO_STARTS)
        downloaded: list[Path] = []
        for filename in filenames:
            local_path = output_dir / filename
            sandbox.fs.download_file(f"{remote_display}/{filename}", str(local_path))
            downloaded.append(local_path)
            print(f"Saved {local_path}")
        return downloaded
    finally:
        if sandbox is not None and not keep_sandbox:
            print("Deleting the temporary Daytona sandbox…")
            client.delete(sandbox)
        elif sandbox is not None:
            print(f"Keeping sandbox {sandbox.id} (--keep-sandbox was set).")
