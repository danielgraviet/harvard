import test from 'node:test'
import assert from 'node:assert/strict'
import {
  RANDOM_PRESET_EPISODES,
  OPTIMAL_PRESET_EPISODES,
  resolveTrainingEpisodes,
  fleetRequestPayload,
  evaluationOutcome,
} from '../src/runConfig.js'

test('the Random preset budget is 500 episodes and Optimal remains 4,000', () => {
  assert.equal(RANDOM_PRESET_EPISODES, 500)
  assert.equal(OPTIMAL_PRESET_EPISODES, 4000)
})

test('a single sandbox uses its selected training episode budget', () => {
  assert.equal(resolveTrainingEpisodes({
    gridSearch: false,
    gridSearchEpisodes: 2000,
    count: 1,
    fleetEpisodes: 200,
    episodes: RANDOM_PRESET_EPISODES,
  }), 500)
})

test('Random preset sends 500 episodes even if the previous hidden value was 4,000', () => {
  const params = { alpha: 0.34, gamma: 0.99, epsilon_start: 0.82, epsilon_end: 0.34 }
  const episodes = resolveTrainingEpisodes({
    gridSearch: false,
    gridSearchEpisodes: 2000,
    overrideEpisodes: RANDOM_PRESET_EPISODES,
    count: 1,
    fleetEpisodes: 200,
    episodes: OPTIMAL_PRESET_EPISODES,
  })
  const request = fleetRequestPayload({ gridSearch: false, backend: 'daytona', count: 1, episodes, seed: 7, params })

  assert.equal(request.episodes, 500)
  assert.equal(request.mode, 'train')
  assert.equal(request.alpha, params.alpha)
  assert.equal(request.gamma, params.gamma)
})

test('multi-sandbox training uses the per-learner episode budget', () => {
  assert.equal(resolveTrainingEpisodes({
    gridSearch: false,
    gridSearchEpisodes: 2000,
    count: 200,
    fleetEpisodes: 500,
    episodes: 4000,
  }), 500)
})

test('explicit validation and grid-search budgets take precedence', () => {
  const options = { gridSearch: false, gridSearchEpisodes: 2000, count: 1, fleetEpisodes: 200, episodes: 500 }
  assert.equal(resolveTrainingEpisodes({ ...options, overrideEpisodes: 4000 }), 4000)
  assert.equal(resolveTrainingEpisodes({ ...options, gridSearch: true }), 2000)
})

test('evaluation outcome is distinct from recent exploratory training success', () => {
  const recentTrainingSuccess = 0.92
  const finalEvaluation = evaluationOutcome([
    { reached_goal: false },
    { reached_goal: false },
    { reached_goal: false },
  ])

  assert.equal(recentTrainingSuccess, 0.92)
  assert.deepEqual(finalEvaluation, {
    kind: 'failed',
    label: 'FAILED',
    detail: 'Reached the goal from 0 of 3 start positions',
    solved: 0,
    total: 3,
  })
})

test('evaluation marks partial and all-start success accurately', () => {
  assert.equal(evaluationOutcome([{ reached_goal: true }, { reached_goal: false }]).label, 'PARTIAL SUCCESS')
  assert.equal(evaluationOutcome([{ reached_goal: true }, { reached_goal: true }]).label, 'SUCCESS')
})
