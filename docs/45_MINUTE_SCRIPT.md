# Workshop script: Explore vs. Exploit

## The one thing that should stick

**Exploration vs. exploitation.** Early on, try things and gather information. Later, stop exploring and use what you've learned. Every attendee is living this dial all weekend on their own project — this workshop just gives them a visible, re-runnable version of it. Daytona is not the lesson; it's the thing quietly making it cheap to try things twice. Don't pitch it. Let the last five minutes reveal it.

## How this script works

This is decision-first, not lecture-first. The audience makes a bet before they get any theory, and the "teaching" happens on demand — in answer to whatever they ask while the run is in progress — instead of as a scheduled block. If nobody asks, don't volunteer it; keep the focus on the choice and the result.

Slides are a bare-bones deck: no logos beyond a small mark on the title slide, no gradients, no bullet animations. Each slide is a single image, a single question, or a short line of plain text — something to look at while you talk, not something to read while you wait. Most of the 45 minutes is you driving the live app on screen, not the deck.

**Total slide count: 9.** Build them in that order; the script tells you when to advance.

### Before the session

- Have the browser app running and visible full-screen: backend (`uv run uvicorn backend.app:app --reload --port 8000`) and frontend (`cd frontend && npm run dev`, open `localhost:5174`) both up and warm before the room fills. Load it once yourself so the first real run isn't also the first-ever request.
- Set "Where should it run?" to **Local CPU** and leave it there until the reveal at minute 30. Keep seed fixed at `7` for both rounds so the only thing that changes is the dial the room picked.
- Have the failure GIF (`display/failed-policy-example-8x8-showcase.gif`) and the progress GIF open locally as a network-independent fallback. If the app or Wi-Fi dies, say so plainly and switch to the GIFs — don't pretend it's live.
- `DAYTONA_API_KEY` lives in `.env` on the presentation laptop only. Never on screen, never typed live.
- Decide who is "room 1" and "room 2" for the vote (e.g., left half / right half, or odd/even rows) so counting hands is fast.

### If someone asks before you planned to explain it

Keep these one-liners ready, but don't recite them unprompted — only answer what's actually asked.

- *What's actually being learned?* A table of values, one row per (position, facing, key possession, and door state), one column per action. Training nudges the values toward "what worked."
- *What's alpha / gamma?* Alpha: how much each new result changes the table. Gamma: how much the agent cares about rewards further in the future.
- *Why not a neural network?* The maze is small enough that a plain table is fast to build and makes it easy to show exactly which action values changed.
- *What is Daytona doing right now?* (Early ask) "We're running this on this laptop's CPU right now — hang on, that's actually where we're headed at the end." Then move on; don't jump ahead of the script.

---

## 0:00–3:00 | Open cold, no explanation

**Slide 1 — title.** Workshop name only, small Daytona mark in the corner. Nothing else. Advance to Slide 2 as you start talking.

**Slide 2 — the failure GIF, full-bleed, no caption.**

*Let it play for a few seconds before saying anything.*

"This agent tried to get from the top-left of this room to the goal in the bottom-right. Watch what happens."

*Let it finish failing.*

"It gets close — two cells away — and doesn't make it. I'm not going to tell you why yet. Keep that picture in your head."

---

## 3:00–9:00 | The choice

**Slide 3 — two words, side by side: "Explore" / "Exploit."** One short line under each: *Explore — try something new, see what happens. Exploit — use what already worked.* Nothing else on the slide.

"Every agent — and honestly, every one of you this weekend — is running on a dial between these two. Explore: try the untested thing. Exploit: do the thing you already know works. Too much of either one, and you waste your 36 hours."

"The agent you just watched has exactly this dial. It's called epsilon, and it controls how often the agent still takes a random action instead of its best-known one. We're going to set where that dial ends up — high or low — and bet on what happens."

**Slide 4 — the two choices.** A screenshot of the app's "Ending exploration · ε" slider, with two marks on it: `0.04` and `0.4`. Caption: *Low ε at the end = mostly exploit. High ε at the end = keep exploring, even late.*

"Room 1, you're betting on ending at 0.04 — mostly exploiting by the end. Room 2, you're betting on 0.4 — still exploring a lot, even late in training. Same seed, same episode count, only this one number changes."

*Take the vote by raised hands. Ask one person from each side for their one-sentence reasoning — don't let this run long.*

"Hold that prediction. We're about to find out."

---

## 9:00–20:00 | Run it live

*Switch from slides to the browser app, full screen. No slide for this block — the app is the visual.*

"I'm setting ending exploration to [the room's pick] and leaving everything else where it is. Seed 7, same as always."

*Set the slider, hit **Train the agent**. While it runs, the progress bar and recent-success metric update live — point at them, but don't lecture over them.*

"Watch 'recent success' and 'states learned' while this runs. We'll come back to what they mean once we have a result to look at."

*This is dead time by design. Use it for whatever the room actually asks — pull from the one-liners above. If nobody asks anything, that's fine; let the run finish in near-silence. Don't fill it with a prepared lecture.*

---

## 20:00–30:00 | Compare to the bet

*When the run finishes, the rollout cards appear — one per starting position, each marked GOAL REACHED or DIDN'T REACH GOAL.*

"Here's the result. Room [whoever guessed right] — you called it. Room [the other], what do you think happened?"

*Take one or two real answers before you explain anything. Let them use the rollout cards and paths, not just the label.*

**Slide 5 — the same two words again: "Explore" / "Exploit," but now with the actual result filled in underneath each.** Keep it to the two numbers/outcomes, nothing more.

"One run on one seed is a signal, not a proof. If we wanted to say '0.04 is just better,' we'd need to repeat this across a bunch of seeds and look at how much it varies. What we do know, from this run: [one sentence describing what actually happened]."

**Slide 6 — one question, nothing else: "Where are you right now — exploring or exploiting?"**

"Take ninety seconds with the person next to you. Tell them which one you're doing right now in your own project, and whether you think that's the right call for where you are in the weekend."

*Give them the full ninety seconds. Take two answers.*

---

## 30:00–40:00 | Second round, and the quiet reveal

**Slide 7 — "Your turn: pick one dial."** List the four: *learning rate (α) · future reward (γ) · starting exploration · ending exploration.* No definitions here — they've either asked already or are about to pick blind, which is fine.

"New pick. Call out one of these four and a direction — higher or lower — and tell me what you expect to change: the success rate, or the path itself."

*Take the call, set it, hit **Train the agent** again. While the second run goes, do the reveal:*

**Slide 8 — a screenshot of the "Where should it run?" toggle: Local CPU / Daytona.** Caption: *Same button. Same code. Different place it runs.*

"Quick thing before this run finishes. Everything you've watched so far ran on this laptop's CPU. This toggle right here — same experiment, same button — can send that exact code to run in a disposable sandbox instead: one gets created, your code runs inside it, the result comes back, and the sandbox is deleted. That's Daytona. It's the thing that let us run this twice in front of you in ten minutes instead of babysitting one machine."

*Flip the toggle, optionally re-fire the same run live if time and network allow — otherwise just leave the explanation as spoken and move on. Don't dwell here; one breath, then back to the result on screen.*

"Back to your pick — here's what happened."

*Compare the second run's rollout cards to the prediction, same as before. Keep this shorter than the first comparison — they already know how to read the cards.*

---

## 40:00–45:00 | Close

**Slide 9 — one sentence, large text, nothing else: "Explore first. Exploit once you know enough. Decide on purpose."**

"That's the whole thing. Explore first, exploit once you know enough, and decide on purpose instead of by default. The agent in this room did it with one number. You're going to do it all weekend with your project, your stack, your team."

"The code, the starter kit, and today's slides are at [repo / QR]. Run it yourself, change a dial, see what happens. If you want to run it in an actual sandbox instead of your laptop, that's the same toggle you just saw."

*Take one or two final questions if the schedule allows. End on the sentence on screen, not on a Daytona pitch.*

---

## Sources and presenter checks

- The live demo is the browser app (`backend/app.py` + `frontend/`), not the CLI — confirm both processes are warm before the room fills.
- Keep seed fixed at `7` across both rounds so only the picked dial changes between comparisons.
- The failure and progress GIFs in `display/` are the offline fallback if the app or network fails during the session.
- The Daytona reveal (minute 30) is deliberately short — one toggle, one sentence on the lifecycle, back to the result. Resist expanding it into a benchmark or feature tour.
- Confirm event A/V, Wi-Fi, and room layout (for the two-room vote) before the workshop.
