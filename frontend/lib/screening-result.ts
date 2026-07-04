export const LAST_SCREENING_KEY = "retinaai-last-screening"

type StoredScreeningProbe = {
  metadata?: { run_id?: string; source?: string }
  model?: { kind?: string | null }
  prediction?: { status?: string | null }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

export function isSyntheticScreeningResult(value: unknown) {
  if (!isRecord(value)) return false
  const result = value as StoredScreeningProbe
  const source = result.metadata?.source || ""
  return (
    source.includes("demo") ||
    result.metadata?.run_id === "sample_case" ||
    result.model?.kind === "frontend_demo" ||
    result.prediction?.status === "demo_sample"
  )
}

export function readStoredScreeningResult<T>() {
  const stored = window.localStorage.getItem(LAST_SCREENING_KEY)
  if (!stored) return null
  try {
    const parsed = JSON.parse(stored) as T
    if (isSyntheticScreeningResult(parsed)) {
      window.localStorage.removeItem(LAST_SCREENING_KEY)
      return null
    }
    return parsed
  } catch {
    return null
  }
}

export function storeScreeningResult(result: unknown) {
  if (isSyntheticScreeningResult(result)) return
  window.localStorage.setItem(LAST_SCREENING_KEY, JSON.stringify(result))
}