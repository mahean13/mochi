// Renders docs/figures/src/*.html to PNG at 3x scale with headless Chromium (Playwright).
// Usage: node scripts/screenshot_figures.cjs [docs/figures]
const path = require("path");
const fs = require("fs");
const { chromium } = require("playwright");

(async () => {
  const outDir = process.argv[2] || "docs/figures";
  const srcDir = path.join(outDir, "src");
  const browser = await chromium.launch();
  for (const f of fs.readdirSync(srcDir).filter((x) => x.endsWith(".html"))) {
    const page = await browser.newPage({ viewport: { width: 1040, height: 600 }, deviceScaleFactor: 3 });
    await page.goto("file://" + path.resolve(srcDir, f));
    await (await page.$("#fig")).screenshot({ path: path.join(outDir, f.replace(".html", ".png")) });
    console.log("wrote", f.replace(".html", ".png"));
  }
  await browser.close();
})();
