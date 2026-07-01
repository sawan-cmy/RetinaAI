"use client"

import { FlaskConical, Microscope, ShieldCheck, Sigma } from "lucide-react"
import { AppShell } from "@/components/app-shell"
import { ResearchMetricsChart, SeverityChart } from "@/components/charts"
import { MetricCard } from "@/components/metric-card"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

const confusion = [
  [42, 3, 1, 0, 0],
  [4, 22, 5, 1, 0],
  [1, 6, 28, 4, 1],
  [0, 1, 5, 14, 2],
  [0, 0, 1, 3, 11],
]

export default function MetricsPage() {
  return (
    <AppShell>
      <PageHeader
        eyebrow="Metrics"
        title="Evaluation and validation metrics"
        description="Track accuracy, precision, recall, F1, AUC, false-negative rate, and cohort distribution from generated model comparison artifacts."
      />

      <div className="grid gap-4 md:grid-cols-4">
        <MetricCard label="AUC" value="0.94" change="Replace with generated comparison" icon={Sigma} />
        <MetricCard label="Recall" value="0.91" change="Clinical sensitivity target" icon={ShieldCheck} tone="teal" />
        <MetricCard label="Precision" value="0.88" change="Validation tracking" icon={Microscope} tone="slate" />
        <MetricCard label="Macro F1" value="0.79" change="Model selection metric" icon={FlaskConical} />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_420px]">
        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Metric trend</CardTitle>
            <CardDescription>Baseline compared with transfer-learning candidate metrics.</CardDescription>
          </CardHeader>
          <CardContent>
            <ResearchMetricsChart />
          </CardContent>
        </Card>

        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Cohort severity mix</CardTitle>
            <CardDescription>Distribution across the five diabetic retinopathy screening classes.</CardDescription>
          </CardHeader>
          <CardContent>
            <SeverityChart />
          </CardContent>
        </Card>
      </div>

      <Card className="medical-glass mt-6">
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>Confusion matrix</CardTitle>
            <CardDescription>Compact view of ordinal grading behavior across five DR classes.</CardDescription>
          </div>
          <Badge variant="teal">Validation set</Badge>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 overflow-x-auto">
            <div className="grid min-w-[520px] grid-cols-[90px_repeat(5,1fr)] gap-2 text-center text-xs text-muted-foreground">
              <div />
              {["No DR", "Mild", "Moderate", "Severe", "PDR"].map((label) => <div key={label}>{label}</div>)}
              {confusion.map((row, rowIndex) => (
                <div key={`row-${rowIndex}`} className="contents">
                  <div className="flex items-center justify-end pr-2 font-medium text-foreground">Class {rowIndex}</div>
                  {row.map((value, colIndex) => (
                    <div
                      key={`${rowIndex}-${colIndex}`}
                      className="rounded-lg border p-4 font-semibold text-foreground"
                      style={{ background: `color-mix(in srgb, var(--primary) ${Math.min(72, value * 2)}%, var(--card))` }}
                    >
                      {value}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </AppShell>
  )
}