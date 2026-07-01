"use client"

import { BrainCircuit, Cpu, GitCompareArrows, Rocket } from "lucide-react"
import { AppShell } from "@/components/app-shell"
import { ResearchMetricsChart } from "@/components/charts"
import { MetricCard } from "@/components/metric-card"
import { TremorProgress } from "@/components/tremor-progress"
import { PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { modelRows } from "@/lib/data"

export default function ModelComparisonPage() {
  return (
    <AppShell>
      <PageHeader
        eyebrow="Model comparison"
        title="Compare candidate screening models"
        description="Train and evaluate EfficientNet-B0, EfficientNet-B3, ResNet50, and the random-forest baseline using the shared pipeline outputs."
        action={<Button><Rocket className="h-4 w-4" /> Promote reviewed model</Button>}
      />

      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard label="Primary CNN" value="EffNet-B0" change="Configured default" icon={BrainCircuit} />
        <MetricCard label="Fast fallback" value="RF baseline" change="Used when CNN is missing" icon={Cpu} tone="teal" />
        <MetricCard label="Models tracked" value="4" change="Three CNNs plus baseline" icon={GitCompareArrows} tone="slate" />
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_360px]">
        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Comparison chart</CardTitle>
            <CardDescription>Macro F1 and safety metrics determine model selection.</CardDescription>
          </CardHeader>
          <CardContent>
            <ResearchMetricsChart />
          </CardContent>
        </Card>

        <Card className="medical-glass">
          <CardHeader>
            <CardTitle>Promotion gates</CardTitle>
            <CardDescription>Model artifacts must pass quality, latency, calibration, and documentation checks.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            {[["Validation complete", 94], ["Latency budget", 82], ["False-negative review", 76], ["Documentation", 88]].map(([label, value]) => (
              <div key={String(label)}>
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">{label}</span>
                  <span className="font-medium text-foreground">{value}%</span>
                </div>
                <TremorProgress value={Number(value)} />
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <Card className="medical-glass mt-6 overflow-hidden">
        <CardHeader>
          <CardTitle>Leaderboard</CardTitle>
          <CardDescription>Metrics map to `reports/comparison.csv` and `reports/comparison.json`.</CardDescription>
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
              {modelRows.map((row) => (
                <tr key={row.name} className="border-b last:border-b-0">
                  <td className="px-6 py-4 font-medium text-foreground">{row.name}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.accuracy}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.precision}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.recall}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.f1}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.auc}</td>
                  <td className="px-6 py-4 text-muted-foreground">{row.latency}</td>
                  <td className="px-6 py-4"><Badge variant={row.status === "Configured" ? "teal" : "slate"}>{row.status}</Badge></td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </AppShell>
  )
}