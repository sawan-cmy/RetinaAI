import { NextRequest, NextResponse } from "next/server"

export const runtime = "nodejs"
export const dynamic = "force-dynamic"

type ScreeningResult = {
  outputs?: {
    gradcam_url?: string | null
    explanation_url?: string | null
    report_url?: string | null
  }
}

const configuredApiBase = process.env.RETINAAI_API_URL
const apiBase = configuredApiBase || (process.env.VERCEL ? "" : "http://127.0.0.1:8000")
const apiKey = process.env.RETINAAI_API_KEY

function absolutize(url?: string | null) {
  if (!url) return null
  if (url.startsWith("http://") || url.startsWith("https://")) return url
  return `${apiBase}${url}`
}

export async function POST(request: NextRequest) {
  const incoming = await request.formData()
  const uploaded = incoming.get("image")
  const upstream = new FormData()

  if (uploaded instanceof File) {
    upstream.append("image", uploaded)
  } else {
    return NextResponse.json({ error: "Upload a retinal image." }, { status: 400 })
  }

  const patientId = incoming.get("patient_id")
  const patientIdValue = typeof patientId === "string" && patientId.trim() ? patientId.trim() : undefined
  if (patientIdValue) upstream.append("patient_id", patientIdValue)

  if (!apiBase) {
    return NextResponse.json(
      { error: "RETINAAI_API_URL is not configured for this deployment." },
      { status: 503 },
    )
  }

  try {
    const response = await fetch(`${apiBase}/predict`, {
      method: "POST",
      body: upstream,
      headers: apiKey ? { "X-API-Key": apiKey } : undefined,
      cache: "no-store",
    })
    const result = (await response.json()) as ScreeningResult & { error?: string; detail?: string }
    if (!response.ok) {
      return NextResponse.json({ error: result.error || result.detail || "screening failed" }, { status: response.status })
    }
    result.outputs = {
      ...(result.outputs || {}),
      gradcam_url: absolutize(result.outputs?.gradcam_url),
      explanation_url: absolutize(result.outputs?.explanation_url),
      report_url: absolutize(result.outputs?.report_url),
    }
    return NextResponse.json(result)
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "RetinaAI API is unavailable" },
      { status: 503 },
    )
  }
}