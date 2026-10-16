export const RANDOM_PRESET_EPISODES: 500
export const OPTIMAL_PRESET_EPISODES: 4000

export function resolveTrainingEpisodes(options: {
  gridSearch: boolean
  gridSearchEpisodes: number
  overrideEpisodes?: number
  count: number
  fleetEpisodes: number
  episodes: number
}): number

export function fleetRequestPayload(options: {
  gridSearch: boolean
  backend: string
  count: number
  episodes: number
  seed: number
  params: { alpha: number; gamma: number; epsilon_start: number; epsilon_end: number }
}): {
  mode: 'train' | 'grid_search'
  backend: string
  workers: number
  concurrency: number
  episodes: number
  seed: number
  alpha: number
  gamma: number
  epsilon_start: number
  epsilon_end: number
}

export type EvaluationRollout = { reached_goal: boolean }
export type EvaluationOutcome = {
  kind: 'success' | 'partial' | 'failed'
  label: 'SUCCESS' | 'PARTIAL SUCCESS' | 'FAILED'
  detail: string
  solved: number
  total: number
}
export function evaluationOutcome(rollouts?: EvaluationRollout[]): EvaluationOutcome | null
