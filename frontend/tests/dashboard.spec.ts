import { expect, test } from "@playwright/test"

test("dashboard navigation exposes clinical workflow pages", async ({ page }) => {
  await page.goto("/")
  await expect(page.getByRole("heading", { name: "Retinal screening operations" })).toBeVisible()

  await page.getByRole("link", { name: /Upload/ }).click()
  await expect(page.getByRole("heading", { name: "Start a retinal screening review" })).toBeVisible()

  await page.getByRole("link", { name: /Prediction/ }).click()
  await expect(page.getByRole("heading", { name: "Latest screening prediction" })).toBeVisible()

  await page.getByRole("link", { name: /Comparison/ }).click()
  await expect(page.getByRole("heading", { name: "Compare candidate screening models" })).toBeVisible()
})