import { NextResponse } from "next/server"

export const runtime = "nodejs"
export const dynamic = "force-dynamic"

const apiBase = process.env.RETINAAI_API_URL || (process.env.VERCEL ? "" : "http://127.0.0.1:8000")
const apiKey = process.env.RETINAAI_API_KEY

export async function GET() {
  if (!apiBase) {
    return NextResponse.json(
      { error: "RETINAAI_API_URL is not configured for this deployment." },
      { status: 503 },
    )
  }

  const headers: HeadersInit = apiKey ? { "X-API-Key": apiKey } : {}
  try {
    const response = await fetch(`${apiBase}/metrics`, {
      headers,
      cache: "no-store",
    })
    const payload = await response.json()
    return NextResponse.json(payload, { status: response.status })
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "RetinaAI API is unavailable" },
      { status: 503 },
    )
  }
}