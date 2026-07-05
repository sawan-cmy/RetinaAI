"use client"

import { BrainCircuit, Cpu, GitCompareArrows, Rocket } from "lucide-react"
import { useMemo } from "react"
import { AppShell } from "@/components/app-shell"
import { ResearchMetricsChart } from "@/components/charts"
import { MetricCard } from "@/components/metric-card"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { chartDataFromMetrics, displayModelName, rowsFromMetrics } from "@/lib/data"
import { useMetrics } from "@/lib/use-metrics"

function sourceLabel(source: "loading" | "generated" | "unavailable") {
  if (source === "generated") return "Generated metrics"
  if (source === "loading") return "Loading metrics"
  return "No generated metrics"
}

const promotionGates = [
  "Validation metrics generated",
  "Latency measured by backend",
  "False-negative review completed",
  "Model card and report artifacts produced",
]

export default function ModelComparisonPage() {
  const { metrics, source } = useMetrics()
  const rows = useMemo(() => rowsFromMetrics(metrics), [metrics])
  const chartData = useMemo(() => chartDataFromMetrics(metrics), [metrics])
  const bestModel = metrics?.best_model || metrics?.model_name ? displayModelName(metrics.best_model || metrics.model_name || undefined) : "pending"
  const label = sourceLabel(source)

  return (
    <AppShell>
      <PageHeader
        eyebrow="Model comparison"
        title="Compare candidate screening models"
        description="Train and evaluate EfficientNet-B0, EfficientNet-B3, ResNet50, and the random-forest baseline using the shared pipeline outputs."
        action={<Button disabled={source !== "generated"}><Rocket className="h-4 w-4" /> Promote reviewed model</Button>}
      />

      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard label="Primary model" value={bestModel} change={label} icon={BrainCircuit} />
        <MetricCard label="Fallback model" value="RF baseline" change="Configured in backend" icon={Cpu} tone="teal" />
        <MetricCard label="Models tracked" value={String(metrics?.models?.length ?? rows.length)} change="Configured candidates" icon={GitCompareArrows} tone="slate" />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_360px]">
        <Card className="medical-glass">
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <CardTitle>Comparison chart</CardTitle>
              <CardDescription>Macro F1 and safety metrics determine model selection.</CardDescription>
            </div>
            <Badge variant={source === "generated" ? "teal" : "slate"}>{label}</Badge>
          </CardHeader>
          <CardContent>
            <ResearchMetricsChart data={chartData} />
          </CardContent>
        </Card>

        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Promotion gates</CardTitle>
            <CardDescription>Model artifacts must pass quality, latency, calibration, and documentation checks.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {promotionGates.map((gate) => (
              <div key={gate} className="rounded-lg border bg-card p-3 text-sm text-muted-foreground">
                {gate}
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <Card className="medical-glass mt-6 overflow-hidden">
        <CardHeader>
          <CardTitle>Leaderboard</CardTitle>
          <CardDescription>Uses `reports/comparison.csv` and `reports/comparison.json` when the backend has generated them.</CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto p-0">
          <table className="w-full min-w-[820px] text-left text-sm">
            <thead className="border-y bg-muted/60 text-muted-foreground">
              <tr>
                <th className="px-6 py-3 font-medium">Model</th>
                <th className="px-6 py-3 font-medium">Accuracy</th>
                <th className="px-6 py-3 font-medium">Precision</th>
                <th className="px-6 py-3 font-medium">Recall</th>
                <th className="px-6 py-3 font-medium">F1</th>
                <th className="px-6 py-3 font-medium">AUC</th>
                <th className="px-6 py-3 font-medium">Latency</th>
                <th className="px-6 py-3 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.name} className="border-b last:border-b-0">
                  <td className="px-6 py-4 font-medium text-foreground">{row.name}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.accuracy}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.precision}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.recall}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.f1}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.auc}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.latency}</td>
                  <td className="px-6 py-4"><Badge variant={row.status === "trained" || row.status === "Configured" ? "teal" : "slate"}>{row.status}</Badge></td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </AppShell>
  )
}