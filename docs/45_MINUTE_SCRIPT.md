# Teach an Agent to Learn in a Daytona CPU Sandbox

## Session setup

This script is written for a 45-minute workshop with undergraduate hackers. The presenter drives the Daytona account and terminal. Attendees do not need an API key or a local environment. They make one decision about the learner, predict what will happen, and compare the resulting path with a prepared baseline.

The throughline is that an agent's work happens inside an environment. The audience sees one small environment, watches an agent fail, changes one training setting, and follows the run from sandbox creation to the saved GIF. The MiniGrid example teaches the experiment loop, while the separate Daytona infrastructure benchmark measures large-scale sandbox performance.

### Before the session

- Run the default Daytona demo with seed `7` and `2,000` episodes. Keep the resulting GIFs and command output available offline.
- Test the live Daytona command below from the presentation laptop. Keep the API key in the local environment. Do not put it on screen or in the sandbox.
- Keep the failure example and progress GIFs open in a local viewer. The failure example is an early policy that moves from `(1, 1)` to `(4, 6)`, comes within two cells of the goal, and fails to finish.
- Keep the command output or a short terminal recording ready in case the network or service is unavailable. Explain when you are showing a prepared result.
- Ask the room to vote by raised hands. No one needs to create an account or install packages during the session.

The prepared default run uses `--epsilon-end 0.04`. For the live experiment, ask the audience to choose between that value and `0.4`. The second value keeps more random exploration in the later episodes. Treat the choice as a hypothesis. The run may confirm or challenge what the audience expects.

The time blocks include the audience's responses, short pair discussions, and the live Daytona run. Leave silence after each question so attendees have time to think. Do not fill every pause with more explanation.

## 0:00–5:00 | Open with the failed run

*Show the failed-policy GIF. Let the agent move for a few seconds before speaking over it.*

“I want to start with an agent that almost makes it. It starts in the upper-left part of the room, travels across the grid, and gets close to the goal. The agent still fails to reach the goal. Take a moment to look at the path. Where do you think it got stuck?”

*Give the room a few seconds to answer. Take two or three responses, then ask attendees to compare their explanations with the person next to them for one minute.*

“Before we explain the learner, take one minute with the person next to you. Talk about what you would want to inspect before changing the agent. Would you look at its route, the number of steps, or the reward?”

*After one minute, take two brief answers and connect them to the saved trajectory.*

“The goal is in the lower-right corner. This run reaches `(4, 6)`, while the goal is `(6, 6)`. It gets close, but the policy it has learned so far does not finish the task. The useful part is that we can see what it tried. We have a specific run to inspect, rather than only a score that says it failed.”

“You are here during a 36-hour hackathon, so you will probably make many attempts at your own projects. Some attempts will work, and some will fail in ways that are difficult to reproduce. Today we will use a small gridworld to look at one complete loop: run an agent, inspect its behavior, change one setting, and compare the next result.”

“By the end, I want you to have one practical pattern you can take into your own project. Give an agent a place to run, keep a record of what happened, and make it easy to run the experiment again.”

## 5:00–10:00 | Explain the environment and learner

*Show the 8×8 board and point out the agent, the walls, and the goal.*

“This room is an 8-by-8 MiniGrid environment. The outer cells are walls, and the goal is in the lower-right corner. The agent has three actions. It can turn left, turn right, or move forward. Each episode ends when the agent reaches the goal or reaches the step limit.”

“For this introduction, the state contains the agent's position and direction. That gives us 144 possible states: 36 walkable positions and four directions. The state is fully observed because we read those values from the simulator. That is a deliberate simplification. MiniGrid can support partially observed tasks, but we are using this room to make the learning loop easy to see.”

“The learner keeps a table of values for each state and action. When an action leads to a useful result, the table updates. When the agent does not reach the goal, it still records the transitions it tried. The next action is selected using those values, with some random exploration mixed in during training.”

“The update moves the current value toward the reward plus the value of the best next action. The learning rate controls how much the new experience changes the table. The discount factor controls how much future rewards count. We keep the code tabular so we can connect each decision to a state, an action, and an update.”

“The exploration setting is called epsilon. A higher epsilon means the agent is more likely to take a random action. A lower epsilon means it is more likely to use the actions that already have higher values. Too little exploration can leave the agent with an incomplete picture of the room. Too much exploration can keep it taking random actions after it has found a useful route.”

*Show the progress GIF and let the audience watch the early trajectory transition into the later policy.*

“This GIF comes from checkpoints in one training run. The early policy makes a visible mistake. Later checkpoints reach the goal from the same demonstration starts. We can compare the behavior directly because the run saves the paths as images.”

“Take thirty seconds to follow one route with your eyes. Tell your partner where the agent first moves closer to the goal. Then tell them what you would inspect if the next checkpoint still failed.”

*Invite one person to point out the change between an early and a later checkpoint.*

## 10:00–15:00 | Show what Daytona runs

*Show the Daytona command and then the relevant part of `daytona_runner.py`.*

“The learner itself is ordinary Python code. Daytona provides the environment where the code runs. The Python SDK creates a sandbox. The runner installs MiniGrid and Pillow, uploads the learner and command-line entry point, and starts the training process inside that sandbox.”

“The run reports the sandbox CPU and GPU allocation before training starts. This demo uses one CPU and zero GPUs. The Q-table is small, and the workload does not need a neural network or a GPU.”

“When training and GIF rendering finish, the runner downloads the five images into our local `display` directory. It then deletes the temporary sandbox. That gives us a visible result from the run and a clean environment for a later attempt.”

*Show the lifecycle on screen: create the sandbox, execute the process, download the GIFs, and delete the sandbox.*

*Put this short excerpt from `daytona_runner.py` on screen.*

```python
client = Daytona()
sandbox = client.create()
response = sandbox.process.exec(command, timeout=180)
sandbox.fs.download_file(remote_path, local_path)
client.delete(sandbox)
```

“These calls show the shape of the integration. The SDK creates the sandbox, the process API runs the command, the filesystem API brings the result back, and the client deletes the sandbox after the run.”

“Take one minute with your partner and match each call to the lifecycle step it performs. Then decide which result you would want to preserve after the sandbox is gone.”

*Ask for one answer about the lifecycle and one answer about the saved artifact.*

“For this workshop, you do not need to provision anything yourself. I will run the SDK flow from the presentation laptop. You will choose the experiment and tell me what you expect to see. That lets us focus on the result without asking everyone to configure credentials.”

## 15:00–20:00 | Let the audience choose the experiment

“We have a prepared baseline that ends training with epsilon at `0.04`. I want to compare it with a run that ends at `0.4`. In that run, the agent will continue taking random actions more often in the later episodes.”

“Before we run it, make a prediction. Do you expect the higher exploration rate to help the agent find a better route, or do you expect it to keep getting in its own way? Raise your hand for the first prediction. Raise your hand for the second.”

*Count the responses. Ask one person from each side to explain their reasoning in one sentence.*

“We will hold the seed and episode count fixed, then change the final epsilon value. The seed makes this particular comparison repeatable. The parameter change gives us a clear question to investigate. We should look at both the success report and the path, because a single number does not show how the agent behaved.”

“If you are working with someone next to you, take one minute to decide what evidence would change your mind. Would you look at the recent success rate? Would you look at whether the agent reaches the goal from more than one start? Would you look at the path it takes?”

*Take one or two answers, then prepare the live command.*

“Write down your prediction in a short form. Record the epsilon value we chose, what you expect to happen to exploration, and one result you will use to judge the run. Keep that prediction beside you until we compare the GIFs.”

## 20:00–30:00 | Run the audience's experiment

*Run the selected Daytona command from the project directory.*

```bash
uv run python main.py \
  --backend daytona \
  --episodes 2000 \
  --seed 7 \
  --epsilon-end 0.4
```

“The command is using the audience's setting. Daytona is creating a CPU sandbox now. While it starts, look at the baseline failure again and decide what you will compare when this run finishes.”

*While the sandbox starts and installs its dependencies, explain the run output.*

“While the job is starting, work with the person next to you on a second experiment you might run after this one. Choose one parameter, keep the seed fixed, and name the metric or trajectory detail you would compare. You have two minutes to agree on a prediction.”

*Give the room two minutes. Ask one pair to describe its proposed experiment while the sandbox continues.*

“The process prints its recent success rate at checkpoints. It also reports the number of states that have entered the Q-table. Those numbers tell us how much of the state space the agent has encountered and whether recent episodes are reaching the goal.”

“On our last full run, training took about five seconds on one CPU. The end-to-end call also includes starting the sandbox and installing its dependencies. The workload is small enough that training itself is quick, and the remote run still gives us a real environment lifecycle to inspect.”

“The visual result matters too. The runner exports an early failed policy, the learning progression, and a final policy from three start positions. If the agent's score changes, we can inspect whether the route changed with it.”

*When the output appears, point out the reported CPU and GPU allocation, the training checkpoints, the saved paths, and the sandbox cleanup.*

“The training loop completed inside the sandbox. The GIFs were downloaded to the laptop, and the temporary sandbox was deleted. The result stays available after the environment has been cleaned up.”

*If the live run is still starting after two minutes, switch to the prepared Daytona GIFs and the saved output from the successful run. Tell the audience that you are using the prepared capture, then continue with the comparison.*

## 30:00–37:00 | Compare the result with the prediction

“Let us return to the prediction. The baseline ended with epsilon at `0.04`. Our audience-selected run ended at `0.4`. What changed in the success report?”

*Read the observed output. Do not claim an outcome that the run did not produce.*

“Now let us compare the paths. The failure example moved close to the goal but did not finish. The later policy shows whether this setting found a route and whether that route works from the other demonstration starts.”

*Show the two GIFs side by side. Ask the room to name one difference in the paths before giving your interpretation.*

“Take ninety seconds with your partner. Compare the baseline failure with the audience-selected run. Write down one difference you can point to in the path and one limitation of the comparison.”

*Give the room ninety seconds. Ask two pairs to report their observations before you interpret the output.*

“The setting changed how often the learner explored during training. The result tells us how that choice affected this seeded run. To assess whether one epsilon value works better in general, we would repeat the comparison across more seeds and report the variation.”

“That distinction is useful in your own projects. One successful run gives us a useful signal to inspect. Broader claims call for repeated trials across multiple seeds, with the variation reported.”

“We can also see why saving the trajectory helps. If the result is worse, we can inspect whether the agent never found the goal, reached a dead end, or kept taking random actions. The next experiment can respond to what happened in the previous one.”

## 37:00–42:00 | Connect one learner to agent infrastructure

“We have looked at one agent in one room. A coding agent has a similar need for an environment. It needs files, dependencies, a shell, and a way to run its tools and tests. If we want to run many experiments at once, each attempt needs an environment that can be created, used, and replaced.”

“That is the infrastructure question behind this small demonstration. The gridworld is easy to understand, so we can focus on the lifecycle. The same pattern applies when an agent is changing code, running tests, or exploring a reinforcement-learning task.”

“In my recent Daytona article, we measured sandbox startup for a separate CPU workload on NVIDIA Vera. In that test, 2,000 sandboxes were online in 27.4 seconds. That figure describes the infrastructure benchmark. Our MiniGrid run gives us a small, visible example of why agent work needs repeatable environments.”

*Show the article title and the single benchmark result. Keep this section to the stated measurement and its workload.*

“For a small experiment, we can run one sandbox. For a larger experiment, our command can queue independent runs and bound how many are active at once. The resource allocation and the workload still matter. We should measure the task we care about instead of assuming that a small demonstration predicts fleet performance.”

“Take one minute to imagine that each team in this room has an agent that needs to edit code and run tests. What does each attempt need from its environment? Tell the person next to you one thing the environment must contain and one result the team should keep.”

*Ask for two answers. Connect files, dependencies, processes, and test output to the environment each agent needs.*

## 42:00–45:00 | Close with a practical takeaway

“Here is the pattern I hope you take into the hackathon. Give the agent an environment where it can act. Keep enough output to understand what it did. Change one thing at a time, and make it straightforward to start another run.”

“In this room, the agent came within two cells of the goal and failed. We could see the path, change its exploration setting, and run it again in a Daytona CPU sandbox. The same experiment loop can help you understand an agent that edits files, runs tests, or uses tools.”

“The Q-learning code and the GIFs are in the starter project. You can run the learner locally, or use the Daytona backend to create a CPU sandbox and collect the output. You can also change the seed or exploration setting and see whether the result changes.”

“Before we finish, think about one task in your project where you would want an agent to try, inspect, and retry its work. What environment would that agent need?”

*Take one or two final answers. Point the audience to the project and Daytona resources, then leave time for questions if the schedule permits.*

## Sources and presenter checks

- The Daytona API lifecycle and resource allocation described here match the current runner in `daytona_runner.py`. The default run creates one CPU sandbox with no GPU, downloads five GIFs, and deletes the sandbox.
- The example failure path and success results come from the seeded MiniGrid demo. Use the current artifacts in `display/` for the presentation.
- The 2,000-sandbox startup result comes from [When 2,000 Agents Share One CPU](https://www.daytona.io/dotfiles/when-2000-agents-share-one-cpu). That article reports a distinct benchmark on NVIDIA Vera.
- Confirm the event schedule, A/V, Wi-Fi, and Daytona product language before the workshop. Keep the prepared GIFs and terminal output available if the network is unavailable.
