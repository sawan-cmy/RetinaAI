"use client"

import { Bell, Database, KeyRound, ShieldCheck, SlidersHorizontal, UserRound } from "lucide-react"
import { AppShell } from "@/components/app-shell"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

const settings = [
  { label: "Route low-confidence cases", description: "Send uncertainty above 0.25 to manual review.", checked: true },
  { label: "Require quality gate pass", description: "Hide model output when image quality is rejected.", checked: true },
  { label: "Include Grad-CAM in reports", description: "Attach explainability panel to exported reports.", checked: true },
  { label: "Auto-notify reviewer", description: "Notify assigned clinician after screening completion.", checked: false },
]

export default function SettingsPage() {
  return (
    <AppShell>
      <PageHeader
        eyebrow="Settings page"
        title="Platform controls"
        description="Configure safety thresholds, reviewer workflow, report defaults, and model governance settings for the RetinaAI workspace."
        action={<Button><ShieldCheck className="h-4 w-4" /> Save controls</Button>}
      />

      <div className="grid gap-6 xl:grid-cols-[1fr_360px]">
        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Safety workflow</CardTitle>
            <CardDescription>Defaults are conservative because screening output must never bypass clinical review.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {settings.map((item) => (
              <label key={item.label} className="flex cursor-pointer items-start justify-between gap-4 rounded-lg border bg-card p-4 transition hover:bg-muted/50">
                <span>
                  <span className="block text-sm font-medium text-foreground">{item.label}</span>
                  <span className="mt-1 block text-sm leading-6 text-muted-foreground">{item.description}</span>
                </span>
                <input type="checkbox" defaultChecked={item.checked} className="mt-1 h-5 w-5 accent-blue-600" aria-label={item.label} />
              </label>
            ))}
          </CardContent>
        </Card>

        <aside className="space-y-6">
          <Card className="medical-glass">
            <CardHeader>
              <CardTitle>Thresholds</CardTitle>
              <CardDescription>Current production guardrails.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              {[["Quality minimum", 80], ["Confidence minimum", 72], ["Review uncertainty", 25]].map(([label, value]) => (
                <div key={String(label)}>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{label}</span>
                    <span className="font-medium text-foreground">{value}%</span>
                  </div>
                  <Progress value={Number(value)} />
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="medical-glass">
            <CardHeader>
              <CardTitle>Workspace</CardTitle>
              <CardDescription>Access and data plane configuration.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              {[
                { icon: UserRound, label: "Reviewer roles", value: "4 active" },
                { icon: KeyRound, label: "API keys", value: "2 scoped" },
                { icon: Database, label: "Storage region", value: "US-East" },
                { icon: Bell, label: "Alerts", value: "Enabled" },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between rounded-lg border bg-card p-3">
                  <div className="flex items-center gap-3">
                    <item.icon className="h-4 w-4 text-secondary" />
                    <span className="text-sm text-muted-foreground">{item.label}</span>
                  </div>
                  <Badge variant="slate">{item.value}</Badge>
                </div>
              ))}
              <Button variant="outline" className="mt-2"><SlidersHorizontal className="h-4 w-4" /> Advanced settings</Button>
            </CardContent>
          </Card>
        </aside>
      </div>
    </AppShell>
  )
}
