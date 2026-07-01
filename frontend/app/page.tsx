"use client"

import { Activity, AlertTriangle, Clock, Gauge, UploadCloud } from "lucide-react"
import Link from "next/link"
import { AppShell } from "@/components/app-shell"
import { CaseVolumeChart, SeverityChart } from "@/components/charts"
import { LastScreeningPanel } from "@/components/last-screening-panel"
import { MetricCard } from "@/components/metric-card"
import { PageHeader } from "@/components/page-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function DashboardPage() {
  return (
    <AppShell>
      <PageHeader
        eyebrow="Dashboard"
        title="Retinal screening operations"
        description="Monitor screening volume, image quality, uncertainty routing, and the latest connected inference result from the RetinaAI pipeline. Outputs are clinical review support only."
        action={
          <Button asChild>
            <Link href="/upload"><UploadCloud className="h-4 w-4" /> New scan</Link>
          </Button>
        }
      />

      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard label="Cases screened" value="373" change="Local dashboard sample" icon={Activity} />
        <MetricCard label="Manual review" value="50" change="Quality or uncertainty routed" icon={AlertTriangle} tone="teal" />
        <MetricCard label="Median latency" value="184 ms" change="Measured per model run" icon={Clock} tone="slate" />
      </div>

      <LastScreeningPanel />

      <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_380px]">
        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Weekly screening flow</CardTitle>
            <CardDescription>Screened cases compared with manual-review routing.</CardDescription>
          </CardHeader>
          <CardContent>
            <CaseVolumeChart />
          </CardContent>
        </Card>

        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Quality and uncertainty posture</CardTitle>
            <CardDescription>Operational guardrails before predictions are shown.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {[
              ["Quality gate", "Enabled"],
              ["CNN primary", "EfficientNet-B0"],
              ["Fallback", "Random forest baseline"],
              ["Reports", "PDF with disclaimer"],
            ].map(([label, value]) => (
              <div key={label} className="flex items-center justify-between rounded-lg border bg-card p-3 text-sm">
                <span className="text-muted-foreground">{label}</span>
                <span className="font-medium text-foreground">{value}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Severity distribution</CardTitle>
            <CardDescription>Current batch distribution across screening classes.</CardDescription>
          </CardHeader>
          <CardContent>
            <SeverityChart />
          </CardContent>
        </Card>

        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Pipeline contract</CardTitle>
            <CardDescription>The dashboard is wired to the same inference entry point used by the API and report generator.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2">
            {["Upload", "Preprocessing", "Quality gate", "CNN or fallback", "Uncertainty", "Grad-CAM", "Recommendation", "PDF report"].map((step) => (
              <div key={step} className="rounded-lg border bg-card p-3 text-sm text-foreground">
                <Gauge className="mb-2 h-4 w-4 text-secondary" />
                {step}
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </AppShell>
  )
}