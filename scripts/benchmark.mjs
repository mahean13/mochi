// Timing of each engine step at study scale, and how the costs grow with the number of picks.
// Usage: node scripts/benchmark.mjs > docs/benchmark.json
import { performance } from "node:perf_hooks";
import os from "node:os";
import { nextPair, advanceGeneration, fitFeatureBT, fitItemBT, picksToRows, replay, recordPick, newState } from "../web/engine.js";
import { runSimulation } from "../sim/simulate.mjs";

function time(fn, reps) {
  for (let i = 0; i < Math.min(5, reps); i++) fn(); // warm-up
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) fn();
  return (performance.now() - t0) / reps;
}
const out = { node: process.version, cpu: os.cpus()[0]?.model ?? "unknown", note: "Single-threaded; mean milliseconds per call after warm-up.", rows: [] };
for (const sessions of [5, 10, 20, 40, 80]) {
  const { state } = runSimulation({ raters: 3, sessionsPerRater: sessions, seed: 1 });
  const P = state.picks.length;
  const clone = () => JSON.parse(JSON.stringify(state));
  const home = state.picks.filter((p) => p.lineage === "home");
  const items = new Set(home.flatMap((p) => [p.left, p.right]));
  const pairs = home.map((p) => ({ winner: p.chosen, loser: p.chosen === p.left ? p.right : p.left }));
  const rows = picksToRows(state, "home");
  out.rows.push({
    picks: P,
    picks_per_lineage: home.length,
    creatures: Object.keys(state.creatures).length,
    nextPair_ms: +time(() => nextPair(state, "home"), 200).toFixed(4),
    itemBT_ms: +time(() => fitItemBT(items, pairs), 10).toFixed(3),
    advanceGeneration_ms: +time(() => advanceGeneration(clone(), "home"), 5).toFixed(3),
    featureBT_ms: +time(() => fitFeatureBT(rows, { lambda: 1 }), 5).toFixed(3),
    replay_ms: +time(() => replay(state.seed, state.picks), 2).toFixed(2),
    // recordPick on a fresh copy, timing only picks that do not end a generation.
    recordPick_ms: (() => {
      const s = clone();
      const L = s.lineages.home;
      const t0 = performance.now();
      let n = 0;
      for (let i = 0; i < 39 - (s.picks.filter((p) => p.lineage === "home" && p.gen === L.gen).length); i++) {
        const { left, right } = nextPair(s, "home");
        recordPick(s, { lineage: "home", left, right, chosen: left, session: "b", participant: "B", round: i + 1 });
        n++;
      }
      return n ? +(((performance.now() - t0) / n)).toFixed(4) : null;
    })(),
  });
}
console.log(JSON.stringify(out, null, 2));
