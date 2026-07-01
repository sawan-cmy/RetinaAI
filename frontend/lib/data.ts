export const caseTimeline = [
  { day: "Mon", screened: 42, review: 7 },
  { day: "Tue", screened: 58, review: 9 },
  { day: "Wed", screened: 51, review: 6 },
  { day: "Thu", screened: 74, review: 11 },
  { day: "Fri", screened: 68, review: 8 },
  { day: "Sat", screened: 36, review: 5 },
  { day: "Sun", screened: 44, review: 4 },
]

export const modelRows = [
  { name: "EfficientNet-B0", accuracy: "from run", precision: "from run", recall: "from run", f1: "from run", auc: "from run", latency: "from run", status: "Configured" },
  { name: "EfficientNet-B3", accuracy: "from run", precision: "from run", recall: "from run", f1: "from run", auc: "from run", latency: "from run", status: "Candidate" },
  { name: "ResNet50", accuracy: "from run", precision: "from run", recall: "from run", f1: "from run", auc: "from run", latency: "from run", status: "Candidate" },
  { name: "Random Forest Baseline", accuracy: "available", precision: "available", recall: "available", f1: "available", auc: "available", latency: "fast", status: "Baseline" },
]

export const severityData = [
  { label: "No DR", value: 44 },
  { label: "Mild", value: 23 },
  { label: "Moderate", value: 19 },
  { label: "Severe", value: 9 },
  { label: "PDR", value: 5 },
]

export const researchMetrics = [
  { metric: "Accuracy", baseline: 0.82, current: 0.9 },
  { metric: "Precision", baseline: 0.78, current: 0.88 },
  { metric: "Recall", baseline: 0.76, current: 0.91 },
  { metric: "F1", baseline: 0.77, current: 0.89 },
]

export const historyRows = [
  { id: "local-demo", patient: "Demo", prediction: "Awaiting connected run", status: "Ready", date: "Local" },
]