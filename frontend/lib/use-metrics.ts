"use client"

import { useEffect, useState } from "react"
import type { MetricsPayload } from "@/lib/data"

type MetricsState = {
  metrics: MetricsPayload | null
  source: "loading" | "generated" | "unavailable"
}

export function useMetrics(): MetricsState {
  const [state, setState] = useState<MetricsState>({ metrics: null, source: "loading" })

  useEffect(() => {
    let cancelled = false
    fetch("/api/metrics", { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("metrics unavailable")
        return await response.json() as MetricsPayload
      })
      .then((payload) => {
        if (cancelled) return
        setState(payload.status === "no_metrics" ? { metrics: null, source: "unavailable" } : { metrics: payload, source: "generated" })
      })
      .catch(() => {
        if (!cancelled) setState({ metrics: null, source: "unavailable" })
      })
    return () => {
      cancelled = true
    }
  }, [])

  return state
}