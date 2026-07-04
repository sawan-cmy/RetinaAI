import { expect, test } from "@playwright/test"
import { mkdir, rm, writeFile } from "node:fs/promises"
import path from "node:path"

test("artifact API serves reports files and blocks traversal", async ({ request }) => {
  const reportsDir = path.resolve(process.cwd(), "reports")
  const artifact = path.join(reportsDir, "playwright-artifact.txt")
  await mkdir(reportsDir, { recursive: true })
  await writeFile(artifact, "artifact-ok", "utf-8")

  try {
    const allowed = await request.get("/api/artifact?file=playwright-artifact.txt")
    expect(allowed.status()).toBe(200)
    expect(await allowed.text()).toBe("artifact-ok")

    const blocked = await request.get("/api/artifact?file=../README.md")
    expect(blocked.status()).toBe(403)
  } finally {
    await rm(artifact, { force: true })
  }
})

test("metrics page shows unavailable state when FastAPI metrics are unavailable", async ({ page }) => {
  await page.route("**/api/metrics", (route) => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: "unavailable" }) }))
  await page.goto("/metrics")
  await expect(page.getByText("No generated metrics").first()).toBeVisible()
})