"use client"

import { Database, KeyRound, ShieldCheck, SlidersHorizontal } from "lucide-react"
import { AppShell } from "@/components/app-shell"
import { PageHeader } from "@/components/page-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

const safetyControls = [
  "Quality thresholds are loaded by the FastAPI backend from config files.",
  "Uncertainty routing is computed by the shared inference pipeline.",
  "Grad-CAM and PDF report availability depends on backend artifacts.",
  "Reviewer notification is not implemented in this frontend.",
]

export default function SettingsPage() {
  return (
    <AppShell>
      <PageHeader
        eyebrow="Settings page"
        title="Platform controls"
        description="This frontend does not persist platform settings. Configure safety thresholds, auth, storage, and reviewer workflow in the backend deployment."
        action={<Button disabled><ShieldCheck className="h-4 w-4" /> Read-only</Button>}
      />

      <div className="grid gap-6 xl:grid-cols-[1fr_360px]">
        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Safety workflow</CardTitle>
            <CardDescription>Displayed as backend requirements, not editable frontend state.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {safetyControls.map((item) => (
              <div key={item} className="rounded-lg border bg-card p-4 text-sm leading-6 text-muted-foreground">
                {item}
              </div>
            ))}
          </CardContent>
        </Card>

        <aside className="space-y-6">
          <Card className="medical-glass">
            <CardHeader>
              <CardTitle>Backend configuration</CardTitle>
              <CardDescription>Required before deployed uploads can run.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <div className="flex gap-3 rounded-lg border bg-card p-3">
                <KeyRound className="mt-0.5 h-4 w-4 text-secondary" />
                <p>Set `RETINAAI_API_KEY` only if the API has key auth enabled.</p>
              </div>
              <div className="flex gap-3 rounded-lg border bg-card p-3">
                <Database className="mt-0.5 h-4 w-4 text-secondary" />
                <p>Set `RETINAAI_API_URL` to a reachable FastAPI deployment for Vercel previews.</p>
              </div>
              <Button variant="outline" className="mt-2" disabled><SlidersHorizontal className="h-4 w-4" /> Backend-managed</Button>
            </CardContent>
          </Card>
        </aside>
      </div>
    </AppShell>
  )
}