"use client"

import { FlaskConical, Microscope, ShieldCheck, Sigma } from "lucide-react"
import { useMemo } from "react"
import { AppShell } from "@/components/app-shell"
import { ResearchMetricsChart, SeverityChart } from "@/components/charts"
import { MetricCard } from "@/components/metric-card"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { chartDataFromMetrics, confusionFromMetrics, severityDataFromMetrics, summaryFromMetrics } from "@/lib/data"
import { useMetrics } from "@/lib/use-metrics"

function sourceLabel(source: "loading" | "generated" | "unavailable") {
  if (source === "generated") return "Generated metrics"
  if (source === "loading") return "Loading metrics"
  return "No generated metrics"
}

export default function MetricsPage() {
  const { metrics, source } = useMetrics()
  const summary = useMemo(() => summaryFromMetrics(metrics), [metrics])
  const chartData = useMemo(() => chartDataFromMetrics(metrics), [metrics])
  const severityData = useMemo(() => severityDataFromMetrics(metrics), [metrics])
  const confusion = useMemo(() => confusionFromMetrics(metrics), [metrics])
  const label = sourceLabel(source)

  return (
    <AppShell>
      <PageHeader
        eyebrow="Metrics"
        title="Evaluation and validation metrics"
        description="Track accuracy, precision, recall, F1, AUC, false-negative rate, and cohort distribution from generated model comparison artifacts."
      />

      <div className="grid gap-4 md:grid-cols-4">
        <MetricCard label="AUC" value={summary.auc} change={label} icon={Sigma} />
        <MetricCard label="Recall" value={summary.recall} change="Backend metric" icon={ShieldCheck} tone="teal" />
        <MetricCard label="Precision" value={summary.precision} change="Backend metric" icon={Microscope} tone="slate" />
        <MetricCard label="Macro F1" value={summary.f1} change="Backend metric" icon={FlaskConical} />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_420px]">
        <Card className="medical-glass">
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <CardTitle>Metric trend</CardTitle>
              <CardDescription>Baseline compared with transfer-learning candidate metrics.</CardDescription>
            </div>
            <Badge variant={source === "generated" ? "teal" : "slate"}>{label}</Badge>
          </CardHeader>
          <CardContent>
            <ResearchMetricsChart data={chartData} />
          </CardContent>
        </Card>

        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Cohort severity mix</CardTitle>
            <CardDescription>Distribution across the five diabetic retinopathy screening classes.</CardDescription>
          </CardHeader>
          <CardContent>
            <SeverityChart data={severityData} />
          </CardContent>
        </Card>
      </div>

      <Card className="medical-glass mt-6">
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>Confusion matrix</CardTitle>
            <CardDescription>Compact view of ordinal grading behavior across five DR classes.</CardDescription>
          </div>
          <Badge variant={source === "generated" && confusion ? "teal" : "slate"}>{confusion ? "Generated artifact" : "No generated matrix"}</Badge>
        </CardHeader>
        <CardContent>
          {confusion ? (
            <div className="grid gap-2 overflow-x-auto">
              <div className="grid min-w-[520px] grid-cols-[90px_repeat(5,1fr)] gap-2 text-center text-xs text-muted-foreground">
                <div />
                {["No DR", "Mild", "Moderate", "Severe", "PDR"].map((classLabel) => <div key={classLabel}>{classLabel}</div>)}
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
          ) : (
            <div className="rounded-lg border bg-card p-6 text-sm text-muted-foreground">
              Run model comparison and expose `/metrics` from the backend to show a confusion matrix.
            </div>
          )}
        </CardContent>
      </Card>
    </AppShell>
  )
}