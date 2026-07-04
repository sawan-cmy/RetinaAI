export const caseTimeline: { day: string; screened: number; review: number }[] = []

export const modelRows = [
  { name: "EfficientNet-B0", accuracy: "pending", precision: "pending", recall: "pending", f1: "pending", auc: "pending", latency: "pending", status: "Configured" },
  { name: "EfficientNet-B3", accuracy: "pending", precision: "pending", recall: "pending", f1: "pending", auc: "pending", latency: "pending", status: "Candidate" },
  { name: "ResNet50", accuracy: "pending", precision: "pending", recall: "pending", f1: "pending", auc: "pending", latency: "pending", status: "Candidate" },
  { name: "Random Forest Baseline", accuracy: "pending", precision: "pending", recall: "pending", f1: "pending", auc: "pending", latency: "pending", status: "Baseline" },
]

export const severityData: { label: string; value: number }[] = []

export const researchMetrics: { metric: string; baseline: number; current: number }[] = []

export const historyRows: { id: string; patient: string; prediction: string; status: string; date: string }[] = []

export type ModelMetric = {
  name?: string
  status?: string
  accuracy?: number | null
  precision?: number | null
  recall?: number | null
  f1?: number | null
  auc?: number | null
  latency_ms?: number | null
  false_negative_rate?: number | null
  confusion_matrix?: number[][] | null
}

export type MetricsPayload = {
  status?: string
  best_model?: string | null
  models?: ModelMetric[]
  accuracy?: number | null
  macro_precision?: number | null
  macro_recall?: number | null
  macro_f1?: number | null
  auc_ovr_macro?: number | null
  false_negative_rate_any_dr?: number | null
  confusion_matrix?: number[][] | null
  test_distribution?: Record<string, number> | null
}

export function formatMetric(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "pending"
}

export function formatLatency(value?: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value)} ms` : "pending"
}

export function displayModelName(name?: string) {
  const names: Record<string, string> = {
    baseline_sklearn: "Random Forest Baseline",
    efficientnet_b0: "EfficientNet-B0",
    efficientnet_b3: "EfficientNet-B3",
    resnet50: "ResNet50",
  }
  return names[name || ""] || name || "Unknown model"
}

export function rowsFromMetrics(metrics: MetricsPayload | null) {
  if (!metrics || metrics.status === "no_metrics") return modelRows
  if (metrics.models?.length) {
    return metrics.models.map((row) => ({
      name: displayModelName(row.name),
      accuracy: formatMetric(row.accuracy),
      precision: formatMetric(row.precision),
      recall: formatMetric(row.recall),
      f1: formatMetric(row.f1),
      auc: formatMetric(row.auc),
      latency: formatLatency(row.latency_ms),
      status: row.status || "trained",
    }))
  }
  return [{
    name: "Random Forest Baseline",
    accuracy: formatMetric(metrics.accuracy),
    precision: formatMetric(metrics.macro_precision),
    recall: formatMetric(metrics.macro_recall),
    f1: formatMetric(metrics.macro_f1),
    auc: formatMetric(metrics.auc_ovr_macro),
    latency: "pending",
    status: "trained",
  }]
}

export function summaryFromMetrics(metrics: MetricsPayload | null) {
  const best = metrics?.models?.find((row) => row.name === metrics.best_model) || metrics?.models?.find((row) => row.status === "trained")
  return {
    auc: formatMetric(best?.auc ?? metrics?.auc_ovr_macro),
    recall: formatMetric(best?.recall ?? metrics?.macro_recall),
    precision: formatMetric(best?.precision ?? metrics?.macro_precision),
    f1: formatMetric(best?.f1 ?? metrics?.macro_f1),
  }
}

export function chartDataFromMetrics(metrics: MetricsPayload | null) {
  const baseline = metrics?.models?.find((row) => row.name === "baseline_sklearn")
  const best = metrics?.models?.find((row) => row.name === metrics.best_model) || metrics?.models?.find((row) => row.status === "trained")
  if (!baseline && !best && !metrics?.macro_f1) return []
  return [
    { metric: "Accuracy", baseline: baseline?.accuracy ?? metrics?.accuracy ?? 0, current: best?.accuracy ?? metrics?.accuracy ?? 0 },
    { metric: "Precision", baseline: baseline?.precision ?? metrics?.macro_precision ?? 0, current: best?.precision ?? metrics?.macro_precision ?? 0 },
    { metric: "Recall", baseline: baseline?.recall ?? metrics?.macro_recall ?? 0, current: best?.recall ?? metrics?.macro_recall ?? 0 },
    { metric: "F1", baseline: baseline?.f1 ?? metrics?.macro_f1 ?? 0, current: best?.f1 ?? metrics?.macro_f1 ?? 0 },
  ]
}

export function severityDataFromMetrics(metrics: MetricsPayload | null) {
  const distribution = metrics?.test_distribution
  if (!distribution) return []
  const labels = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
  return labels.map((label, index) => ({ label, value: Number(distribution[String(index)] || 0) }))
}

export function confusionFromMetrics(metrics: MetricsPayload | null) {
  return metrics?.models?.find((row) => row.name === metrics.best_model)?.confusion_matrix || metrics?.confusion_matrix || null
}