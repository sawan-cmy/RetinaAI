import { cn } from "@/lib/utils"

export function ConfidenceGauge({ label, value, sublabel, className }: { label: string; value: number; sublabel: string; className?: string }) {
  const angle = Math.min(100, Math.max(0, value)) * 3.6

  return (
    <div className={cn("rounded-lg border bg-card p-5 shadow-sm", className)}>
      <div className="mx-auto grid h-36 w-36 place-items-center rounded-full" style={{ background: `conic-gradient(var(--primary) ${angle}deg, color-mix(in srgb, var(--muted) 88%, transparent) ${angle}deg)` }}>
        <div className="grid h-28 w-28 place-items-center rounded-full bg-card text-center shadow-inner">
          <div>
            <p className="text-3xl font-semibold text-foreground">{value}%</p>
            <p className="text-xs text-muted-foreground">confidence</p>
          </div>
        </div>
      </div>
      <div className="mt-4 text-center">
        <p className="font-medium text-foreground">{label}</p>
        <p className="mt-1 text-sm text-muted-foreground">{sublabel}</p>
      </div>
    </div>
  )
}
