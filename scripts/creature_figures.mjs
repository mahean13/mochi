// Gene figure for the write-up: what each gene does (low vs high value with every other gene held at 0.5).
// Usage: node scripts/creature_figures.mjs docs/figures
//        node scripts/screenshot_figures.cjs   (turns the HTML into PNG; needs Playwright)
import fs from "node:fs";
import path from "node:path";
import { renderSVG, GENES, D } from "../web/engine.js";

const [outDir = "docs/figures"] = process.argv.slice(2);
const font = "font-family:'DejaVu Sans',Inter,sans-serif";

function page(body, w) {
  return `<!doctype html><html><body style="margin:0;background:#fff"><div id="fig" style="width:${w}px;padding:8px;${font};color:#2A231E">${body}</div></body></html>`;
}

// 2. Gene glossary: each gene at 0.05 and 0.95, all others at 0.5.
const rows = GENES.map((g, k) => {
  const lo = new Array(D).fill(0.5); lo[k] = 0.05;
  const hi = new Array(D).fill(0.5); hi[k] = 0.95;
  return `<div style="display:flex;align-items:center;gap:2px;border:1px solid #E4D8C8;border-radius:8px;padding:2px 6px">${renderSVG(lo, { size: 66 })}<span style="font-size:14px;color:#9A8B7C">→</span>${renderSVG(hi, { size: 66 })}<div style="font-size:11.5px;width:74px;line-height:1.2">${k + 1}. ${g.label}</div></div>`;
}).join("");
const glossHtml = page(`<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:6px">${rows}</div>`, 1000);

const srcDir = path.join(outDir, "src");
fs.mkdirSync(srcDir, { recursive: true });
fs.writeFileSync(path.join(srcDir, "fig_genes.html"), glossHtml);
console.log(`HTML written to ${srcDir}`);
