export const RANDOM_PRESET_EPISODES = 500
export const OPTIMAL_PRESET_EPISODES = 4000

export function resolveTrainingEpisodes({ gridSearch, gridSearchEpisodes, overrideEpisodes, count, fleetEpisodes, episodes }) {
  return gridSearch
    ? gridSearchEpisodes
    : overrideEpisodes ?? (count > 1 ? fleetEpisodes : episodes)
}

export function fleetRequestPayload({ gridSearch, backend, count, episodes, seed, params }) {
  return {
    mode: gridSearch ? 'grid_search' : 'train',
    backend,
    workers: count,
    concurrency: count,
    episodes,
    seed,
    alpha: params.alpha,
    gamma: params.gamma,
    epsilon_start: params.epsilon_start,
    epsilon_end: params.epsilon_end,
  }
}

export function evaluationOutcome(rollouts) {
  if (!rollouts?.length) return null
  const solved = rollouts.filter(rollout => rollout.reached_goal).length
  if (solved === rollouts.length) return { kind: 'success', label: 'SUCCESS', detail: `Reached the goal from all ${rollouts.length} start positions`, solved, total: rollouts.length }
  if (solved > 0) return { kind: 'partial', label: 'PARTIAL SUCCESS', detail: `Reached the goal from ${solved} of ${rollouts.length} start positions`, solved, total: rollouts.length }
  return { kind: 'failed', label: 'FAILED', detail: `Reached the goal from 0 of ${rollouts.length} start positions`, solved, total: rollouts.length }
}
