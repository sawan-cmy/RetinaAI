"use client"

import { useEffect, useState } from "react"
import { Clock, FileText, History } from "lucide-react"
import { AppShell } from "@/components/app-shell"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

type ScreeningResult = {
  metadata?: { run_id?: string; patient_id?: string | null; generated_at?: string }
  quality?: { status?: string }
  prediction?: { class_name?: string | null; status?: string; confidence?: number | null }
  uncertainty?: { manual_review?: boolean; reason?: string }
  outputs?: { report_url?: string | null }
}

function readLast(): ScreeningResult | null {
  try {
    const stored = window.localStorage.getItem("retinaai-last-screening")
    return stored ? JSON.parse(stored) as ScreeningResult : null
  } catch {
    return null
  }
}

export default function HistoryPage() {
  const [latest, setLatest] = useState<ScreeningResult | null>(null)

  useEffect(() => {
    const update = () => setLatest(readLast())
    update()
    window.addEventListener("storage", update)
    window.addEventListener("retinaai-last-screening-updated", update)
    return () => {
      window.removeEventListener("storage", update)
      window.removeEventListener("retinaai-last-screening-updated", update)
    }
  }, [])

  const rows = latest ? [latest] : []

  return (
    <AppShell>
      <PageHeader
        eyebrow="History"
        title="Screening history"
        description="Review locally retained screening results and report links generated through the connected upload workflow."
      />

      <Card className="medical-glass overflow-hidden">
        <CardHeader>
          <CardTitle>Recent cases</CardTitle>
          <CardDescription>Local browser history is intentionally minimal; production deployments should persist audit records server-side.</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto p-0">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead className="border-y bg-muted/60 text-muted-foreground">
              <tr>
                <th className="px-6 py-3 font-medium">Run</th>
                <th className="px-6 py-3 font-medium">Patient</th>
                <th className="px-6 py-3 font-medium">Prediction</th>
                <th className="px-6 py-3 font-medium">Quality</th>
                <th className="px-6 py-3 font-medium">Routing</th>
                <th className="px-6 py-3 font-medium">Report</th>
              </tr>
            </thead>
            <tbody>
              {rows.length ? rows.map((row) => (
                <tr key={row.metadata?.run_id || "latest"} className="border-b last:border-b-0">
                  <td className="px-6 py-4 font-medium text-foreground">{row.metadata?.run_id || "latest"}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.metadata?.patient_id || "Unspecified"}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.prediction?.class_name || row.prediction?.status || "not available"}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.quality?.status || "unknown"}</td>
                  <td className="px-6 py-4"><Badge variant={row.uncertainty?.manual_review ? "danger" : "teal"}>{row.uncertainty?.manual_review ? "Manual review" : "AI support"}</Badge></td>
                  <td className="px-6 py-4">{row.outputs?.report_url ? <Button asChild size="sm" variant="outline"><a href={row.outputs.report_url} target="_blank"><FileText className="h-4 w-4" /> Open</a></Button> : "not available"}</td>
                </tr>
              )) : (
                <tr>
                  <td className="px-6 py-8 text-muted-foreground" colSpan={6}>
                    <div className="flex items-center gap-2"><History className="h-4 w-4" /> No local screening history yet. Run a scan from Upload.</div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        {["Quality gate", "Model fallback", "PDF report"].map((item) => (
          <Card key={item} className="medical-glass">
            <CardContent className="p-5">
              <Clock className="h-5 w-5 text-secondary" />
              <p className="mt-4 text-sm font-medium text-foreground">{item}</p>
              <p className="mt-1 text-sm leading-6 text-muted-foreground">Captured in each connected screening result.</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </AppShell>
  )
}