"use client"

import { Moon, Sun } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"

export function ThemeToggle() {
  const [dark, setDark] = useState(false)

  function toggleTheme() {
    const nextDark = !document.documentElement.classList.contains("dark")
    document.documentElement.classList.toggle("dark", nextDark)
    window.localStorage.setItem("retinaai-theme", nextDark ? "dark" : "light")
    setDark(nextDark)
  }

  return (
    <Button variant="outline" size="icon" onClick={toggleTheme} aria-label="Toggle color theme">
      {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  )
}
