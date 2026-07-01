import { cn } from "@/lib/utils"

export function AnimatedGridPattern({ className }: { className?: string }) {
  return (
    <div className={cn("pointer-events-none absolute inset-0 overflow-hidden", className)} aria-hidden="true">
      <div className="retina-grid absolute inset-0 opacity-45" />
    </div>
  )
}