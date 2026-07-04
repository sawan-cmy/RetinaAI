"use client"

import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { caseTimeline, researchMetrics, severityData } from "@/lib/data"

const colors = ["#2563EB", "#14B8A6", "#60A5FA", "#F59E0B", "#F43F5E"]

type CaseTimelinePoint = typeof caseTimeline[number]
type SeverityPoint = typeof severityData[number]
type ResearchMetricPoint = typeof researchMetrics[number]

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="grid h-[250px] place-items-center rounded-lg border bg-card p-6 text-center text-sm text-muted-foreground">
      {message}
    </div>
  )
}

export function CaseVolumeChart({ data = caseTimeline }: { data?: CaseTimelinePoint[] }) {
  if (!data.length) return <EmptyChart message="No generated case-volume data is loaded." />

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={{ left: -20, right: 10, top: 10, bottom: 0 }}>
        <defs>
          <linearGradient id="screened" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#2563EB" stopOpacity={0.36} />
            <stop offset="95%" stopColor="#2563EB" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="review" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#14B8A6" stopOpacity={0.34} />
            <stop offset="95%" stopColor="#14B8A6" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="day" tickLine={false} axisLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
        <YAxis tickLine={false} axisLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
        <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 12 }} />
        <Area type="monotone" dataKey="screened" stroke="#2563EB" strokeWidth={2} fill="url(#screened)" />
        <Area type="monotone" dataKey="review" stroke="#14B8A6" strokeWidth={2} fill="url(#review)" />
      </AreaChart>
    </ResponsiveContainer>
  )
}

export function SeverityChart({ data = severityData }: { data?: SeverityPoint[] }) {
  if (!data.length) return <EmptyChart message="No generated severity distribution is loaded." />

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={data} margin={{ left: -20, right: 10, top: 10, bottom: 0 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
        <YAxis tickLine={false} axisLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
        <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 12 }} />
        <Bar dataKey="value" radius={[10, 10, 4, 4]}>
          {data.map((entry, index) => <Cell key={entry.label} fill={colors[index % colors.length]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function ResearchMetricsChart({ data = researchMetrics }: { data?: ResearchMetricPoint[] }) {
  if (!data.length) return <EmptyChart message="No generated comparison metrics are loaded." />

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ left: -20, right: 10, top: 10, bottom: 0 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey="metric" tickLine={false} axisLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
        <YAxis domain={[0.6, 1]} tickLine={false} axisLine={false} tick={{ fill: "var(--muted-foreground)", fontSize: 12 }} />
        <Tooltip contentStyle={{ background: "var(--card)", border: "1px solid var(--border)", borderRadius: 12 }} />
        <Line type="monotone" dataKey="baseline" stroke="#94A3B8" strokeWidth={2} dot={{ r: 4 }} />
        <Line type="monotone" dataKey="current" stroke="#14B8A6" strokeWidth={3} dot={{ r: 4 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}