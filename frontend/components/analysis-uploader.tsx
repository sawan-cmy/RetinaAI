"use client"

import { AlertTriangle, CheckCircle2, FileImage, Loader2, UploadCloud } from "lucide-react"
import { useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

type ScreeningResult = {
  quality?: { status?: string; reasons?: string[]; blur_score?: number; brightness_score?: number; contrast_score?: number }
  prediction?: { status?: string; class_name?: string | null; confidence?: number | null; probabilities?: number[] | null }
  uncertainty?: { manual_review?: boolean; reason?: string; entropy?: number | null; margin?: number | null }
  outputs?: { explanation_url?: string | null; report_url?: string | null }
  error?: string
}

function percent(value?: number | null) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "n/a"
}

export function AnalysisUploader() {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [result, setResult] = useState<ScreeningResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function runAnalysis(file?: File, demo = false) {
    setLoading(true)
    setError(null)
    const form = new FormData()
    if (file) form.append("image", file)
    if (demo) form.append("demo", "true")

    try {
      const response = await fetch("/api/screen", { method: "POST", body: form })
      const payload = (await response.json()) as ScreeningResult
      if (!response.ok) throw new Error(payload.error || "Screening failed")
      window.localStorage.setItem("retinaai-last-screening", JSON.stringify(payload))
      window.dispatchEvent(new Event("retinaai-last-screening-updated"))
      setResult(payload)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Screening failed")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="w-full max-w-2xl">
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/tiff"
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) void runAnalysis(file)
        }}
      />
      <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-lg bg-primary/10 text-primary">
        {loading ? <Loader2 className="h-10 w-10 animate-spin" /> : <UploadCloud className="h-10 w-10" />}
      </div>
      <h2 className="mt-6 text-2xl font-semibold text-foreground">Drop a retinal fundus image</h2>
      <p className="mt-3 text-sm leading-6 text-muted-foreground">
        Upload a scan or run demo mode. The pipeline performs quality gating before prediction and routes uncertain cases for review.
      </p>
      <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
        <Button onClick={() => inputRef.current?.click()} disabled={loading}>
          <UploadCloud className="h-4 w-4" /> Select image
        </Button>
        <Button variant="outline" onClick={() => void runAnalysis(undefined, true)} disabled={loading}>
          <FileImage className="h-4 w-4" /> Use sample case
        </Button>
      </div>

      {error ? (
        <div className="mt-6 rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-left text-sm text-destructive">
          {error}
        </div>
      ) : null}

      {result ? (
        <Card className="mt-6 text-left">
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <CardTitle>Live screening result</CardTitle>
                <CardDescription>Quality, prediction, routing, and artifacts from the current run.</CardDescription>
              </div>
              <Badge variant={result.uncertainty?.manual_review ? "danger" : "teal"}>
                {result.uncertainty?.manual_review ? "Manual review" : "AI support ready"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-lg border bg-card p-3">
                <p className="text-xs text-muted-foreground">Quality</p>
                <p className="mt-1 font-semibold text-foreground">{result.quality?.status || "unknown"}</p>
              </div>
              <div className="rounded-lg border bg-card p-3">
                <p className="text-xs text-muted-foreground">Severity</p>
                <p className="mt-1 font-semibold text-foreground">{result.prediction?.class_name || result.prediction?.status || "not available"}</p>
              </div>
              <div className="rounded-lg border bg-card p-3">
                <p className="text-xs text-muted-foreground">Confidence</p>
                <p className="mt-1 font-semibold text-foreground">{percent(result.prediction?.confidence)}</p>
              </div>
            </div>
            <div className="rounded-lg border bg-muted/50 p-4 text-sm text-muted-foreground">
              <div className="flex items-start gap-2">
                {result.uncertainty?.manual_review ? <AlertTriangle className="mt-0.5 h-4 w-4 text-destructive" /> : <CheckCircle2 className="mt-0.5 h-4 w-4 text-secondary" />}
                <p>Routing reason: <span className="font-medium text-foreground">{result.uncertainty?.reason || "n/a"}</span></p>
              </div>
              <p className="mt-2">Entropy: {result.uncertainty?.entropy ?? "n/a"} | Margin: {result.uncertainty?.margin ?? "n/a"}</p>
            </div>
            <div className="flex flex-wrap gap-3">
              {result.outputs?.explanation_url ? <Button asChild variant="outline"><a href={result.outputs.explanation_url} target="_blank">Open explanation</a></Button> : null}
              {result.outputs?.report_url ? <Button asChild><a href={result.outputs.report_url} target="_blank">Open report</a></Button> : null}
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}


