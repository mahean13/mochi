// Builds web/index.html: the game as one self-contained file, with the engine inlined.
// The engine stays a normal ES module (web/engine.js) for the simulation and the tests.
import fs from "node:fs";

const engine = fs.readFileSync("web/engine.js", "utf8").replace(/^export /gm, "");
const src = fs.readFileSync("web/index.src.html", "utf8");
if (!src.includes("/*__ENGINE__*/")) throw new Error("Engine placeholder missing from web/index.src.html");
const out = src.replace("/*__ENGINE__*/", () => `// ---- engine.js v${/ENGINE_VERSION = "([^"]+)"/.exec(engine)[1]} (inlined by scripts/build.mjs)\n${engine}`);
fs.writeFileSync("web/index.html", out);
console.log(`web/index.html written (${(out.length / 1024).toFixed(1)} KB)`);
