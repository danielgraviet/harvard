# Harvard Hacks 2026 × Daytona: presentation and workshop plan

## What success looks like

Make Daytona memorable to Harvard hackers as the team that made it easy to safely run, inspect, and iterate on AI agents—not just as another infrastructure sponsor. Give attendees a useful mental model, a working hands-on experience, and a clear next step. Keep the messaging honest: demonstrate the product as it exists, and label any prototype or mocked component.

## What the assignment asks for

- **Opening remarks:** about 5 minutes in the Friday, October 16 opening ceremony (schedule says 3–4 PM). Slides or a demo are allowed.
- **Daytona workshop:** Saturday, October 17 at 2:30 PM. The schedule workbook lists 2:30–3:30 PM, while the organizer message says workshops are 45 minutes. Plan a 45-minute core session and keep a 15-minute flex/Q&A buffer until the duration is confirmed.
- **Sponsor booth:** Friday and Saturday presence is noted. The notes say Nikola joins Friday; Mislav, Joshua, and the organizer arrive Saturday. Confirm staffing, setup windows, and coverage directly with the team.
- **Audience:** undergraduate hackers at a free 36-hour hackathon. Tech talks should be relevant to young college-age builders; slides and demos are allowed. Organizers need the topic and A/V needs in advance.

## Unifying idea

**Working title:** *Teach an Agent to Learn in a Daytona CPU Sandbox*

Teach the core idea through Farama’s open-source MiniGrid. Attendees can change a learning parameter, run a tabular Q-learning agent, observe its success rate and policy, and compare results. Run this small workload in a Daytona CPU sandbox through the Daytona Python SDK. The technical point is that the same agent code can execute in an isolated, disposable environment; the exercise does not need a GPU, deep-learning stack, or long training run.

Keep the claim precise: this is tabular Q-learning over an intentionally compact, fully observed state, not production RL or MiniGrid’s usual partial-observation benchmark. The goal is to let hackers understand the learning loop and see their code run in a CPU sandbox.

**Core takeaway:** An agent is more useful when it can safely act in a real environment, and developers can inspect and reset that environment quickly.

## Opening remarks: 5-minute shape

Aim for a short human opening, not a product pitch. One simple slide or no slides; no live-demo dependency.

1. **0:00–0:40 — Welcome:** Recognize the builders and the intensity of a 36-hour hackathon.
2. **0:40–1:40 — One idea:** Building agents is increasingly about giving them tools and environments, not only better prompts.
3. **1:40–3:20 — Tiny story/demo clip:** Show a MiniGrid agent trying a task, failing visibly, then improving after a small change. A prerecorded clip or animated sequence keeps this reliable.
4. **3:20–4:20 — Challenge:** Ask hackers to change one learning setting and predict how the policy or success rate will change.
5. **4:20–5:00 — Send-off:** Invite them to the Daytona workshop/booth and encourage ambitious, collaborative building.

Avoid lengthy company history, unsupported performance claims, and a feature tour in the opening slot.

## Workshop: 45-minute core + optional buffer

**Working title:** *Teach an Agent to Learn in a Daytona CPU Sandbox*

Use the complete [45-minute speaker script](45_MINUTE_SCRIPT.md) to run the workshop. It includes the audience vote, the live parameter experiment, the Daytona SDK walkthrough, and the prepared fallback.

| Time | Activity | Attendee outcome |
|---|---|---|
| 0–5 min | Cold open: watch a deliberately imperfect agent tackle a tiny grid game | Understand the challenge and see the target result |
| 5–10 min | Explain the loop: observe → act → get feedback → change policy/reward → rerun | Have a simple model of agent iteration |
| 10–15 min | Use the Daytona Python SDK to create a CPU sandbox and run the demo there | Understand the SDK → sandbox → process execution path |
| 15–30 min | Guided build: adjust epsilon decay or learning rate, run the agent, inspect progress | Make and observe one meaningful change in the Q-table policy |
| 30–38 min | Challenge round: evaluate the greedy policy from several start positions; compare parameter settings | See whether a learned policy works across different starting states |
| 38–45 min | Debrief, connect the pattern to real coding agents, share starter materials and next step | Leave with a reusable idea and way to continue |
| 45–60 min | Optional Q&A / retries if the organizer confirms a 60-minute slot | Extra help without making the core session depend on it |

### Workshop design principles

- Have a browser-based or otherwise low-friction participant path. Do not make account creation, API keys, or local setup a prerequisite for the main activity.
- Keep a prepared, known-good demo and a local/offline fallback. The activity should still work if network access or a live service fails.
- Use visible feedback: a game board, attempt count, score, and agent behavior are more compelling than logs alone.
- Make Daytona’s role legible in the demo. Show an environment being created/used/reset, and explain what runs there; do not imply the game itself requires Daytona if it does not.
- Prefer a provided baseline and one or two safe controls (reward, exploration, or policy choice) over asking beginners to implement RL from scratch.
- Ensure the challenge is attainable in 15 minutes and has a satisfying result even if the agent does not “learn” robustly.
- Do not collect credentials or expose secrets in participant environments. Decide in advance whether the workshop uses a shared service, per-attendee environments, or a local simulation.

## Booth: a repeatable 2-minute interaction

1. **Hook:** “Can you get this agent through the level?” Let the visitor make one choice.
2. **Show:** Run it and display what the agent did, including failure if it fails.
3. **Reset/change:** Change one setting and rerun quickly.
4. **Connect:** Explain in one sentence how isolated environments help agents use tools safely and let developers reset experiments.
5. **Invite:** Point them to the workshop/starter repo and ask what they are building.

Use a screen with large visual feedback, a concise one-page handout/QR code, and a booth version that can be reset between visitors. Provide a non-demo conversation path when the screen or network is unavailable.

## Demo and materials to build

1. **Runnable MiniGrid/agent prototype:** MiniGrid Empty-8x8, tabular Q-learning, adjustable learning parameters, deterministic seed, visible success checkpoints, and policy rollouts.
2. **Daytona-backed execution path:** create a CPU sandbox with the Python SDK, install MiniGrid, transfer the demo script, run training in the sandbox, and clean up the sandbox. Confirm the SDK flow with the Daytona team before the event.
3. **Workshop starter kit:** prerequisites, one-click or short setup path, step-by-step exercise, troubleshooting, cleanup instructions, and take-home links.
4. **Opening visual:** one slide or a short prerecorded clip showing failure → change → improved attempt.
5. **Booth assets:** large display view, QR code to starter materials, simple sign, and a lightweight fallback demo.
6. **Presenter notes:** exact timings, handoffs, demo reset steps, and backup plan.

Keep the starter experience independent from the explanatory slide deck so the audience can continue building after the session.

## Preparation sequence

### First: lock the brief

- Confirm exact opening slot, speaker(s), workshop duration, room, A/V, internet availability, expected attendance, and whether attendees have laptops.
- Confirm workshop title/description and send them to organizers for the schedule and promotion.
- Align with Daytona colleagues on audience, product claims, demo ownership, booth shifts, staffing, and the product flow we can safely show.
- Decide the hands-on execution model and what data/account setup it requires.

### Then: prove the experience

- Build the smallest complete activity and run it end to end on a clean machine/browser.
- Measure how long first-time setup and one game iteration take; remove friction until the main exercise fits comfortably in 15 minutes.
- Test the start-position challenge with multiple seeds and parameter settings; make the success metric easy to interpret.
- Rehearse reset/fallback paths and prepare a prerecorded capture of the critical demo.

### Then: shape the story and rehearse

- Write a one-sentence message and a 5-minute opening script.
- Rehearse workshop transitions and booth pitch with a non-expert; revise where they get stuck.
- Run a timed full rehearsal with the actual laptop, adapters, browser, slides, and network assumptions.
- Prepare a compact QR-linked resource page and a clear follow-up path approved by Daytona.

## Open decisions and risks

- **Workshop timing conflict:** workbook says 60 minutes; organizer message says 45 minutes. Prepare the 45-minute core until confirmed.
- **Topic approval:** send title, abstract, audience assumptions, and A/V needs to organizers.
- **Demo dependency:** decide whether Daytona infrastructure is required for the participant activity or only shown in the demo; keep the teaching objective alive if the service is unavailable.
- **Product accuracy:** validate all Daytona descriptions, product names, and claims with the team before publishing slides or handouts.
- **Audience setup:** account creation, laptop availability, Wi-Fi, and venue A/V remain unconfirmed.
- **State scope:** the current implementation uses simulator position and direction as a compact fully observed state. Explain this simplification and avoid presenting it as a partial-observation MiniGrid benchmark.
- **Booth continuity:** clarify staffing, power/display needs, and who can troubleshoot the demo across both days.

## Measures of a strong outcome

- The opening leaves one idea with the room and sends people to a specific next step.
- Most workshop attendees make at least one change and see an interpretable effect within the session.
- The demo shows a real, accurate Daytona workflow and recovers cleanly from failure.
- Attendees can access the starter materials without depending on a live conversation.
- Collect lightweight feedback and participation counts; avoid treating raw booth traffic as proof of impact.

## Source notes

This plan is based on the organizer messages and concept captured in `CONTEXT.md`, plus the internal schedule workbook. Dates and times in the workbook are Friday October 16 through Sunday October 18, 2026. Confirm logistics and product details with the organizers and Daytona team before treating them as final.
