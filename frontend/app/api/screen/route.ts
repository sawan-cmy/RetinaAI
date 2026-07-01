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

const apiBase = process.env.RETINAAI_API_URL || "http://127.0.0.1:8000"
const demoPngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAW0lEQVR4nO3PQQ0AIBDAMMC/5yFjRxMFfXpn5Qbq2wHgW4CZAJgJgJkAmAmAmQCYCYCZAJgJgJkAmAmAmQCYCYCZAJgJgJkAmAmAmQCYCYCZAJgJgJkAmAmAmQCYCYCZALwWAi79v0adAAAAAElFTkSuQmCC"

function absolutize(url?: string | null) {
  if (!url) return null
  if (url.startsWith("http://") || url.startsWith("https://")) return url
  return `${apiBase}${url}`
}

export async function POST(request: NextRequest) {
  const incoming = await request.formData()
  const demo = incoming.get("demo") === "true"
  const uploaded = incoming.get("image")
  const upstream = new FormData()

  if (demo) {
    upstream.append("image", new Blob([Buffer.from(demoPngBase64, "base64")], { type: "image/png" }), "demo.png")
  } else if (uploaded instanceof File) {
    upstream.append("image", uploaded)
  } else {
    return NextResponse.json({ error: "Upload a retinal image or choose demo mode." }, { status: 400 })
  }

  const patientId = incoming.get("patient_id")
  if (typeof patientId === "string" && patientId.trim()) upstream.append("patient_id", patientId.trim())

  try {
    const response = await fetch(`${apiBase}/predict`, {
      method: "POST",
      body: upstream,
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