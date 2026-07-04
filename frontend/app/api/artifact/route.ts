import { NextRequest, NextResponse } from "next/server"
import { readFile } from "node:fs/promises"
import path from "node:path"

export const runtime = "nodejs"

const reportsRoot = path.resolve(process.env.RETINAAI_REPORTS_DIR || path.join(/*turbopackIgnore: true*/ process.cwd(), "reports"))
const mimeTypes: Record<string, string> = {
  ".pdf": "application/pdf",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
}

export async function GET(request: NextRequest) {
  const file = request.nextUrl.searchParams.get("file")
  if (!file) return NextResponse.json({ error: "Missing file parameter." }, { status: 400 })

  const resolved = path.resolve(reportsRoot, file)
  if (!resolved.startsWith(reportsRoot + path.sep)) {
    return NextResponse.json({ error: "Artifact path is not allowed." }, { status: 403 })
  }

  try {
    const body = await readFile(resolved)
    return new NextResponse(body, {
      headers: {
        "content-type": mimeTypes[path.extname(resolved).toLowerCase()] || "application/octet-stream",
        "cache-control": "no-store",
      },
    })
  } catch {
    return NextResponse.json({ error: "Artifact not found." }, { status: 404 })
  }
}