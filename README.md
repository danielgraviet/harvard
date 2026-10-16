# Harvard Hacks 2026 × Daytona

Workshop demo: teach a tabular Q-learning agent to find a key, unlock a door, and reach a goal in Farama’s [MiniGrid](https://github.com/Farama-Foundation/Minigrid), then run the same CPU workload inside Daytona sandboxes through the Python SDK.

## Run it

Install the project dependencies, then run locally:

```bash
uv sync
uv run python main.py --backend local
```

To run training in a Daytona CPU sandbox, configure `DAYTONA_API_KEY` (and `DAYTONA_API_URL` / `DAYTONA_TARGET` if needed), then run:

```bash
uv run python main.py --backend daytona
```

## Interactive browser demo

The workshop UI runs live training locally and streams checkpoints to the browser. In one terminal, start the FastAPI backend:

```bash
uv sync
uv run uvicorn backend.app:app --reload --port 8000
```

In another terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5174>. The experiment setup panel chooses local CPU or Daytona, the number of parallel learners, episode budget, learning rate, future reward, exploration schedule, and seed. The agent workspace shows one grid or a live wall of grids, and expands to a focus view when a run starts. Each completed policy is evaluated from several starting positions and labeled learned, partial, or didn't learn.

The API terminal logs each run ID, parameter set, training checkpoint, completion summary, and full exception traceback on failure. If the browser cannot connect, confirm the API is listening on port 8000 and inspect that terminal.

### Daytona parameter grid search

Use **Grid search · 200 Daytona sandboxes** to compare 20 stratified parameter combinations across 10 learning seeds. Every run uses the same fixed DoorKey maze; trial seeds change the learner's exploration choices, not the door, key, or maze layout. The search covers learning rate (α), future reward (γ), and both ends of the exploration schedule (ε). Each trial gets a disposable sandbox. Candidate scores report mean goal-reaching rate across trials and a 95% confidence interval; the top mean is highlighted, with shorter successful paths breaking ties. The best parameter set, score, interval, and candidate ranking are saved to `backend/best_params.json` and loaded into the UI on the next visit. Use **Apply** to configure the next experiment with the saved values. The per-candidate episode budget is configurable in setup. This requires `DAYTONA_API_KEY`; Daytona organization limits may constrain sandbox startup, and failed sandboxes remain visible as failed trials.

### Parallel fleet view

Choose 8, 32, 100, or 200 parallel learners in the experiment setup panel. **Local rehearsal** uses a local thread pool capped at 32 workers. **Daytona sandboxes** creates one real CPU sandbox per grid, streams learner checkpoints from each sandbox, and deletes each sandbox after its run. Without a prepared snapshot, each sandbox installs MiniGrid and Pillow at startup. For a faster 200-sandbox launch, set `DAYTONA_FLEET_SNAPSHOT` in `.env` to a Daytona snapshot that already contains MiniGrid 3.1.0 and Pillow; the fleet runner then skips installation. Daytona mode also requires `DAYTONA_API_KEY`. Organization limits and service availability determine how many remote sandboxes can start successfully at once. Launching a remote fleet only happens when you press the run button.

The Daytona path creates a temporary sandbox, installs MiniGrid and Pillow there, uploads the learner and CLI, runs training and GIF rendering inside the sandbox, downloads the GIFs into `display/`, and deletes the sandbox. The local `DAYTONA_API_KEY` is read from `.env` when present and is not copied into the sandbox. Add `--keep-sandbox` when inspecting the environment manually. Training defaults to 2,000 episodes in the 8×8 room and accepts `--episodes`, `--seed`, `--alpha`, `--gamma`, `--epsilon-start`, and `--epsilon-end`. Each run gets a unique asset ID; pass `--run-id` to set one explicitly. Run multiple independent learners with `--runs 100` and cap simultaneous sandboxes with `--concurrency 4`.

## Why tabular Q-learning

MiniGrid's DoorKey-8x8 task adds a locked door, a key, and an internal wall. The tabular state tracks position, direction, key possession, and door state; the learner can turn, move, pick up the key, and toggle the door. The reward is sparse, so weak exploration, low discounting, or too few episodes can leave it stuck before reaching the goal. A dictionary Q-table keeps each update visible. Runs report recent success, then evaluate the greedy policy from several starting positions.

Batch mode queues any number of independent runs and uses bounded async orchestration to keep at most `--concurrency` jobs active at once. Each run gets its own seed, sandbox, and GIF filenames.

This demo reads the agent’s position, direction, key possession, and door state from MiniGrid’s underlying simulator as a compact, fully observed state. It is intentionally a tabular-RL teaching setup, not a realistic partial-observation benchmark or a claim that a production agent can inspect simulator internals.

## Workshop plan

Use the 45-minute schedule in [PRESENTATION_PLAN.md](docs/PRESENTATION_PLAN.md). Suggested audience exercise: change epsilon decay or the learning rate, rerun, and compare the success curve and resulting policy. The display assets work in slides or on a booth screen without a live service connection.

## Display GIFs

An early policy moves close to the goal, then gets stuck two cells away:

![An early Q-learning policy moves toward the goal, then gets stuck](display/failed-policy-example-8x8-showcase.gif)

The progression then shows the policy improving over training:

![MiniGrid Q-learning progress](display/learning-progress-8x8-showcase.gif)

Final-policy rollouts: [start at (1, 1)](display/policy-8x8-showcase-start-1-1.gif) · [start at (1, 6)](display/policy-8x8-showcase-start-1-6.gif) · [start at (6, 1)](display/policy-8x8-showcase-start-6-1.gif). Each run generates a unique ID; pass `--run-id` to set one explicitly.

Other context: [CONTEXT.md](docs/CONTEXT.md) has organizer messages and the original concept; the internal schedule is in [the workbook](docs/INTERNAL%20HackHarvard2026%20Schedule.xlsx).
