"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  Activity,
  BarChart3,
  Brain,
  ClipboardList,
  FileText,
  Gauge,
  History,
  LayoutDashboard,
  Settings,
  UploadCloud,
} from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const navItems = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "Upload", icon: UploadCloud },
  { href: "/prediction", label: "Prediction", icon: Gauge },
  { href: "/gradcam", label: "Grad-CAM", icon: Brain },
  { href: "/metrics", label: "Metrics", icon: BarChart3 },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/model-comparison", label: "Comparison", icon: ClipboardList },
  { href: "/history", label: "History", icon: History },
  { href: "/settings", label: "Settings", icon: Settings },
]

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[280px_1fr]">
      <aside className="medical-glass sticky top-0 z-40 border-b px-4 py-4 lg:h-screen lg:border-b-0 lg:border-r lg:px-5">
        <div className="mx-auto flex max-w-7xl items-center justify-between lg:block">
          <Link href="/" className="flex items-center gap-3" aria-label="RetinaAI dashboard">
            <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
              <Activity className="h-5 w-5" />
            </span>
            <span>
              <span className="block text-sm font-semibold text-foreground">RetinaAI</span>
              <span className="block text-xs text-muted-foreground">Clinical screening</span>
            </span>
          </Link>
          <div className="flex items-center gap-2 lg:hidden">
            <ThemeToggle />
          </div>
        </div>

        <div className="mt-5 hidden rounded-lg border bg-card p-4 lg:block">
          <div className="flex items-center justify-between gap-3">
            <Badge variant="teal">Safe mode</Badge>
            <span className="text-xs text-muted-foreground">v1.0</span>
          </div>
          <p className="mt-3 text-sm font-medium text-foreground">Manual-review routing is enabled.</p>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">Low-quality, missing-model, and uncertain cases do not produce unsupported claims.</p>
        </div>

        <nav className="mt-4 flex gap-2 overflow-x-auto pb-2 lg:mt-6 lg:block lg:space-y-1 lg:overflow-visible lg:pb-0" aria-label="Primary">
          {navItems.map((item) => {
            const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href))
            const Icon = item.icon
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex shrink-0 items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition hover:bg-muted hover:text-foreground",
                  active && "bg-primary text-primary-foreground shadow-sm hover:bg-primary hover:text-primary-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="mt-6 hidden items-center justify-between rounded-lg border bg-card p-3 lg:flex">
          <div>
            <p className="text-xs font-medium text-foreground">Appearance</p>
            <p className="text-xs text-muted-foreground">Dark and light mode</p>
          </div>
          <ThemeToggle />
        </div>

        <Button asChild className="mt-6 hidden w-full lg:inline-flex" variant="secondary">
          <Link href="/upload">New retinal scan</Link>
        </Button>
      </aside>
      <main className="min-w-0 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">{children}</main>
    </div>
  )
}