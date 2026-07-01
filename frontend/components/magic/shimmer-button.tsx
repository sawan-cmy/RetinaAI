import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { cn } from "@/lib/utils"

export function ShimmerButton({ href, children, className }: { href: string; children: React.ReactNode; className?: string }) {
  return (
    <Link
      href={href}
      className={cn(
        "group relative inline-flex h-12 items-center justify-center overflow-hidden rounded-lg bg-primary px-6 text-sm font-medium text-primary-foreground shadow-sm transition hover:bg-primary/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2",
        className,
      )}
    >
      <span className="absolute inset-y-0 -left-10 w-10 rotate-12 bg-white/30 blur-md transition-all duration-700 group-hover:left-full" aria-hidden="true" />
      <span className="relative z-10 flex items-center gap-2">{children}<ArrowRight className="h-4 w-4" /></span>
    </Link>
  )
}
