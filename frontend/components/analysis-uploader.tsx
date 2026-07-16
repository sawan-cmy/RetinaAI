"use client"

import { AlertTriangle, CheckCircle2, Loader2, UploadCloud } from "lucide-react"
import { type ReactNode, useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { storeScreeningResult } from "@/lib/screening-result"

type ScreeningResult = {
  quality?: { status?: string; reasons?: string[]; blur_score?: number; brightness_score?: number; contrast_score?: number }
  prediction?: { status?: string; class_name?: string | null; confidence?: number | null; probabilities?: number[] | null }
  uncertainty?: { manual_review?: boolean; reason?: string; entropy?: number | null; margin?: number | null }
  referral?: { category?: string; result?: string; action?: string; manual_review_required?: boolean }
  acquisition?: { eye_laterality?: string; screening_site?: string }
  outputs?: { explanation_url?: string | null; report_url?: string | null; json_url?: string | null }
  error?: string
}

type MetadataState = {
  patient_id: string
  age: string
  sex: string
  eye_laterality: string
  diabetes_type: string
  known_duration_of_diabetes: string
  latest_hba1c: string
  blood_pressure: string
  previous_dr_history: string
  current_visual_symptoms: string
  capture_device: string
  screening_site: string
  operator_id: string
}

const initialMetadata: MetadataState = {
  patient_id: "",
  age: "",
  sex: "",
  eye_laterality: "unknown",
  diabetes_type: "",
  known_duration_of_diabetes: "",
  latest_hba1c: "",
  blood_pressure: "",
  previous_dr_history: "",
  current_visual_symptoms: "",
  capture_device: "",
  screening_site: "",
  operator_id: "",
}

const inputClass = "mt-1 h-10 w-full rounded-lg border bg-card px-3 text-sm text-foreground outline-none focus:border-primary"
const textareaClass = "mt-1 min-h-20 w-full rounded-lg border bg-card px-3 py-2 text-sm text-foreground outline-none focus:border-primary"

function percent(value?: number | null) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "n/a"
}

function fieldLabel(label: string, children: ReactNode) {
  return (
    <label className="block text-xs font-medium text-muted-foreground">
      {label}
      {children}
    </label>
  )
}

export function AnalysisUploader() {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [metadata, setMetadata] = useState<MetadataState>(initialMetadata)
  const [result, setResult] = useState<ScreeningResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function updateMetadata(field: keyof MetadataState, value: string) {
    setMetadata((current) => ({ ...current, [field]: value }))
  }

  function appendMetadata(form: FormData) {
    Object.entries(metadata).forEach(([key, value]) => {
      const trimmed = value.trim()
      if (trimmed) form.append(key, trimmed)
    })
  }

  async function runAnalysis(file: File) {
    setLoading(true)
    setError(null)
    setResult(null)
    const form = new FormData()
    form.append("image", file)
    appendMetadata(form)

    try {
      const response = await fetch("/api/screen", { method: "POST", body: form })
      const payload = (await response.json()) as ScreeningResult
      if (!response.ok) throw new Error(payload.error || "Screening failed")
      storeScreeningResult(payload)
      window.dispatchEvent(new Event("retinaai-last-screening-updated"))
      setResult(payload)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Screening failed")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="w-full max-w-3xl">
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
      <h2 className="mt-6 text-2xl font-semibold text-foreground">Upload a retinal fundus image</h2>
      <p className="mt-3 text-sm leading-6 text-muted-foreground">
        The frontend only displays results returned by the connected RetinaAI API. No prediction is generated in the browser.
      </p>

      <div className="mt-6 rounded-lg border bg-card p-4 text-left">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-foreground">Optional report metadata</p>
            <p className="mt-1 text-xs text-muted-foreground">Blank fields are recorded as not provided in the PDF.</p>
          </div>
          <Badge variant="slate">Optional</Badge>
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {fieldLabel("Patient ID", <input className={inputClass} value={metadata.patient_id} onChange={(event) => updateMetadata("patient_id", event.target.value)} maxLength={120} />)}
          {fieldLabel("Age", <input className={inputClass} value={metadata.age} onChange={(event) => updateMetadata("age", event.target.value)} inputMode="numeric" />)}
          {fieldLabel("Sex", (
            <select className={inputClass} value={metadata.sex} onChange={(event) => updateMetadata("sex", event.target.value)}>
              <option value="">Not provided</option>
              <option value="female">Female</option>
              <option value="male">Male</option>
              <option value="intersex">Intersex</option>
              <option value="other">Other</option>
              <option value="prefer_not_to_say">Prefer not to say</option>
              <option value="unknown">Unknown</option>
            </select>
          ))}
          {fieldLabel("Eye laterality", (
            <select className={inputClass} value={metadata.eye_laterality} onChange={(event) => updateMetadata("eye_laterality", event.target.value)}>
              <option value="unknown">Unknown</option>
              <option value="left">Left</option>
              <option value="right">Right</option>
            </select>
          ))}
          {fieldLabel("Diabetes type", (
            <select className={inputClass} value={metadata.diabetes_type} onChange={(event) => updateMetadata("diabetes_type", event.target.value)}>
              <option value="">Not provided</option>
              <option value="type_1">Type 1</option>
              <option value="type_2">Type 2</option>
              <option value="gestational">Gestational</option>
              <option value="other">Other</option>
              <option value="unknown">Unknown</option>
            </select>
          ))}
          {fieldLabel("Known duration of diabetes", <input className={inputClass} value={metadata.known_duration_of_diabetes} onChange={(event) => updateMetadata("known_duration_of_diabetes", event.target.value)} maxLength={80} />)}
          {fieldLabel("Latest HbA1c", <input className={inputClass} value={metadata.latest_hba1c} onChange={(event) => updateMetadata("latest_hba1c", event.target.value)} inputMode="decimal" />)}
          {fieldLabel("Blood pressure", <input className={inputClass} value={metadata.blood_pressure} onChange={(event) => updateMetadata("blood_pressure", event.target.value)} maxLength={40} />)}
          {fieldLabel("Capture device/camera", <input className={inputClass} value={metadata.capture_device} onChange={(event) => updateMetadata("capture_device", event.target.value)} maxLength={120} />)}
          {fieldLabel("Screening site", <input className={inputClass} value={metadata.screening_site} onChange={(event) => updateMetadata("screening_site", event.target.value)} maxLength={160} />)}
          {fieldLabel("Operator ID", <input className={inputClass} value={metadata.operator_id} onChange={(event) => updateMetadata("operator_id", event.target.value)} maxLength={120} />)}
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {fieldLabel("Previous DR history", <textarea className={textareaClass} value={metadata.previous_dr_history} onChange={(event) => updateMetadata("previous_dr_history", event.target.value)} maxLength={400} />)}
          {fieldLabel("Current visual symptoms", <textarea className={textareaClass} value={metadata.current_visual_symptoms} onChange={(event) => updateMetadata("current_visual_symptoms", event.target.value)} maxLength={400} />)}
        </div>
      </div>

      <div className="mt-6 flex justify-center">
        <Button onClick={() => inputRef.current?.click()} disabled={loading}>
          <UploadCloud className="h-4 w-4" /> Select image
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
                <CardDescription>Quality, prediction, routing, and artifacts from the current API run.</CardDescription>
              </div>
              <Badge variant={result.uncertainty?.manual_review ? "danger" : "teal"}>
                {result.uncertainty?.manual_review ? "Manual review" : "Pipeline output"}
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
                <p className="text-xs text-muted-foreground">Screening class</p>
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
                <div>
                  <p>Routing reason: <span className="font-medium text-foreground">{result.uncertainty?.reason || "n/a"}</span></p>
                  <p className="mt-2">Referral category: <span className="font-medium text-foreground">{result.referral?.category || "not available"}</span></p>
                </div>
              </div>
              <p className="mt-2">Entropy: {result.uncertainty?.entropy ?? "n/a"} | Margin: {result.uncertainty?.margin ?? "n/a"}</p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg border bg-card p-3">
                <p className="text-xs text-muted-foreground">Eye</p>
                <p className="mt-1 font-semibold text-foreground">{result.acquisition?.eye_laterality || "unknown"}</p>
              </div>
              <div className="rounded-lg border bg-card p-3">
                <p className="text-xs text-muted-foreground">Next action</p>
                <p className="mt-1 text-sm font-medium leading-5 text-foreground">{result.referral?.action || "Clinical review before care decisions."}</p>
              </div>
            </div>
            <div className="flex flex-wrap gap-3">
              {result.outputs?.explanation_url ? <Button asChild variant="outline"><a href={result.outputs.explanation_url} target="_blank">Open explanation</a></Button> : null}
              {result.outputs?.report_url ? <Button asChild><a href={result.outputs.report_url} target="_blank">Open report</a></Button> : null}
              {result.outputs?.json_url ? <Button asChild variant="outline"><a href={result.outputs.json_url} target="_blank">Open JSON</a></Button> : null}
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}
