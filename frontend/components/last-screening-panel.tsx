"use client"

import { useMemo, useSyncExternalStore } from "react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LAST_SCREENING_KEY, readStoredScreeningResult } from "@/lib/screening-result"

type ScreeningResult = {
  quality?: { status?: string }
  prediction?: { status?: string; class_name?: string | null; confidence?: number | null }
  uncertainty?: { manual_review?: boolean; reason?: string; entropy?: number | null; margin?: number | null }
  outputs?: { report_url?: string | null; explanation_url?: string | null }
}

function subscribe(callback: () => void) {
  window.addEventListener("storage", callback)
  window.addEventListener("retinaai-last-screening-updated", callback)
  return () => {
    window.removeEventListener("storage", callback)
    window.removeEventListener("retinaai-last-screening-updated", callback)
  }
}

function getSnapshot() {
  return window.localStorage.getItem(LAST_SCREENING_KEY)
}

function getServerSnapshot() {
  return null
}

export function LastScreeningPanel() {
  const stored = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
  const result = useMemo(() => stored ? readStoredScreeningResult<ScreeningResult>() : null, [stored])

  if (!result) {
    return (
      <Card className="medical-glass mt-6">
        <CardHeader>
          <CardTitle>Latest connected screening</CardTitle>
          <CardDescription>Upload an image on the Upload page to populate this panel from a real API response.</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card className="medical-glass mt-6">
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <CardTitle>Latest connected screening</CardTitle>
          <CardDescription>Persisted locally from the upload API response.</CardDescription>
        </div>
        <Badge variant={result.uncertainty?.manual_review ? "danger" : "teal"}>
          {result.uncertainty?.manual_review ? "Manual review" : "Pipeline output"}
        </Badge>
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-4">
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground">Quality</p>
          <p className="mt-2 font-semibold text-foreground">{result.quality?.status || "unknown"}</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground">Severity</p>
          <p className="mt-2 font-semibold text-foreground">{result.prediction?.class_name || result.prediction?.status || "n/a"}</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground">Confidence</p>
          <p className="mt-2 font-semibold text-foreground">{typeof result.prediction?.confidence === "number" ? `${Math.round(result.prediction.confidence * 100)}%` : "n/a"}</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <p className="text-xs text-muted-foreground">Routing</p>
          <p className="mt-2 font-semibold text-foreground">{result.uncertainty?.reason || "n/a"}</p>
        </div>
      </CardContent>
    </Card>
  )
}