import { ArrowUpRight, type LucideIcon } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

export function MetricCard({
  label,
  value,
  change,
  icon: Icon,
  tone = "blue",
}: {
  label: string
  value: string
  change: string
  icon: LucideIcon
  tone?: "blue" | "teal" | "slate"
}) {
  return (
    <Card className="medical-glass">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm text-muted-foreground">{label}</p>
            <p className="mt-2 text-2xl font-semibold text-foreground">{value}</p>
          </div>
          <span
            className={cn(
              "flex h-11 w-11 items-center justify-center rounded-lg",
              tone === "blue" && "bg-primary/10 text-primary",
              tone === "teal" && "bg-secondary/10 text-secondary",
              tone === "slate" && "bg-muted text-muted-foreground",
            )}
          >
            <Icon className="h-5 w-5" />
          </span>
        </div>
        <div className="mt-5 flex items-center gap-1 text-xs font-medium text-secondary">
          <ArrowUpRight className="h-3.5 w-3.5" />
          {change}
        </div>
      </CardContent>
    </Card>
  )
}
