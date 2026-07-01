import { cn } from "@/lib/utils"

export function TremorProgress({ value, tone = "blue", className }: { value: number; tone?: "blue" | "teal"; className?: string }) {
  const color = tone === "teal" ? "bg-secondary" : "bg-primary"

  return (
    <div className={cn("h-2.5 overflow-hidden rounded-full bg-muted", className)} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={value}>
      <div className={cn("h-full rounded-full transition-all", color)} style={{ width: `${Math.min(100, Math.max(0, value))}%` }} />
    </div>
  )
}
