// Engine tests. Run with: node --test tests/
import test from "node:test";
import assert from "node:assert/strict";
import {
  D, GENES, GRID, CONFIG, renderSVG, randomGenome, rngFor, descriptors, cellOf,
  fitItemBT, fitFeatureBT, predictLeft, sigmoid, newState, nextPair, recordPick,
  replay, stateFingerprint, picksCSV, creaturesCSV, normal,
} from "../web/engine.js";
import { createHash } from "node:crypto";
import { runSimulation } from "../sim/simulate.mjs";

const sha = (s) => createHash("sha256").update(s).digest("hex");

test("genome has 20 genes with valid ranges", () => {
  assert.equal(D, 20);
  for (const g of GENES) assert.ok(g.max > g.min, g.key);
  assert.equal(new Set(GENES.map((g) => g.key)).size, D);
});

test("rendering is deterministic: same genome, same picture", () => {
  const g = randomGenome(rngFor(7));
  assert.equal(renderSVG(g), renderSVG([...g]));
  assert.notEqual(renderSVG(g), renderSVG(randomGenome(rngFor(8))));
  assert.match(renderSVG(g), /^<svg[^>]*viewBox="0 0 200 200"/);
});

test("descriptors stay on [0,1] and map to a valid archive cell", () => {
  const rng = rngFor(3);
  for (let i = 0; i < 500; i++) {
    const g = randomGenome(rng);
    for (const v of descriptors(g)) assert.ok(v >= 0 && v <= 1);
    const [a, b] = cellOf(g).split(",").map(Number);
    assert.ok(a >= 0 && a < GRID && b >= 0 && b < GRID);
  }
});

test("item-level Bradley-Terry ranks a clearly stronger item first", () => {
  const strengths = { a: 3, b: 1, c: 0.3 };
  const rng = rngFor(11);
  const picks = [];
  const ids = Object.keys(strengths);
  for (let i = 0; i < 3000; i++) {
    const x = ids[Math.floor(rng() * 3)];
    let y = ids[Math.floor(rng() * 3)];
    if (x === y) continue;
    const p = strengths[x] / (strengths[x] + strengths[y]);
    picks.push(rng() < p ? { winner: x, loser: y } : { winner: y, loser: x });
  }
  const s = fitItemBT(ids, picks);
  assert.ok(s.a > s.b && s.b > s.c);
  // The log-strength gap a-b should be near log(3).
  assert.ok(Math.abs(s.a - s.b - Math.log(3)) < 0.2);
});

test("feature-level Bradley-Terry recovers known weights from synthetic picks", () => {
  const rng = rngFor(5);
  const w = new Array(D).fill(0);
  w[0] = 2; w[6] = -1.5; w[10] = 1;
  const rows = [];
  for (let i = 0; i < 6000; i++) {
    const xl = randomGenome(rng).map((v) => v - 0.5);
    const xr = randomGenome(rng).map((v) => v - 0.5);
    let eta = 0.2;
    for (let k = 0; k < D; k++) eta += w[k] * (xl[k] - xr[k]);
    rows.push({ xl, xr, y: rng() < sigmoid(eta) ? 1 : 0 });
  }
  const m = fitFeatureBT(rows, { lambda: 0.1 });
  assert.ok(Math.abs(m.weights[0] - 2) < 0.3);
  assert.ok(Math.abs(m.weights[6] + 1.5) < 0.3);
  assert.ok(Math.abs(m.weights[10] - 1) < 0.3);
  assert.ok(Math.abs(m.bias - 0.2) < 0.15);
  const p = predictLeft(m, rows[0].xl, rows[0].xr);
  assert.ok(p > 0 && p < 1);
});

test("pair scheduler balances exposure and avoids repeats within a generation", () => {
  const state = newState(99);
  const seen = new Set();
  const count = {};
  for (let r = 0; r < CONFIG.picksPerGeneration; r++) {
    const { left, right } = nextPair(state, "home");
    assert.notEqual(left, right);
    const key = [left, right].sort().join("~");
    assert.ok(!seen.has(key), "repeated pair " + key);
    seen.add(key);
    count[left] = (count[left] || 0) + 1;
    count[right] = (count[right] || 0) + 1;
    recordPick(state, { lineage: "home", left, right, chosen: left, session: "t", participant: "T", round: r + 1 });
  }
  const c = Object.values(count);
  assert.ok(Math.max(...c) - Math.min(...c) <= 1, "exposure differs by more than one");
});

test("a generation advances after 40 picks and keeps the population valid", () => {
  const state = newState(4);
  const rng = rngFor(4, "t");
  for (let r = 0; r < CONFIG.picksPerGeneration; r++) {
    const { left, right } = nextPair(state, "trust");
    recordPick(state, { lineage: "trust", left, right, chosen: rng() < 0.5 ? left : right, session: "t", participant: "T", round: r + 1 });
  }
  const L = state.lineages.trust;
  assert.equal(L.gen, 1);
  assert.equal(L.active.length, CONFIG.activeSize);
  assert.ok(Object.keys(L.archive).length <= GRID * GRID);
  for (const id of L.active) for (const v of state.creatures[id].genes) assert.ok(v >= 0 && v <= 1);
  const kids = L.active.filter((id) => state.creatures[id].gen === 1);
  assert.ok(kids.length >= CONFIG.activeSize - CONFIG.eliteCount);
  for (const id of kids) assert.equal(state.creatures[id].parents.length, 2);
  assert.equal(state.lineages.home.gen, 0, "the other lineage is untouched");
});

test("replay of the pick log rebuilds the identical study", () => {
  const { state } = runSimulation({ raters: 3, sessionsPerRater: 4, seed: 123 });
  const rebuilt = replay(state.seed, state.picks);
  assert.equal(stateFingerprint(rebuilt), stateFingerprint(state));
  assert.deepEqual(Object.keys(rebuilt.creatures).sort(), Object.keys(state.creatures).sort());
});

test("replay rejects a tampered log", () => {
  const { state } = runSimulation({ raters: 2, sessionsPerRater: 2, seed: 9 });
  const picks = state.picks.map((p) => ({ ...p }));
  picks[50].gen = 99;
  assert.throws(() => replay(state.seed, picks));
});

test("simulation is reproducible for a fixed seed", () => {
  const a = runSimulation({ raters: 2, sessionsPerRater: 2, seed: 77 }).state;
  const b = runSimulation({ raters: 2, sessionsPerRater: 2, seed: 77 }).state;
  assert.equal(picksCSV(a), picksCSV(b));
});

test("CSV exports have one row per pick and per creature", () => {
  const { state } = runSimulation({ raters: 1, sessionsPerRater: 2, seed: 1 });
  assert.equal(picksCSV(state).split("\n").length, state.picks.length + 1);
  assert.equal(creaturesCSV(state).split("\n").length, Object.keys(state.creatures).length + 1);
  assert.equal(picksCSV(state).split("\n")[0].split(",").length, 12 + 2 * D);
});

test("normal draws have mean near 0 and sd near 1", () => {
  const rng = rngFor(2);
  const xs = Array.from({ length: 20000 }, () => normal(rng));
  const m = xs.reduce((a, b) => a + b) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  assert.ok(Math.abs(m) < 0.03 && Math.abs(sd - 1) < 0.03);
});

// ---------------------------------------------------------------- golden tests
// These pin the frozen study design (engine 1.1.0). Any change to a gene's range, the
// drawing, the breeding settings or the scheduler changes one of these values and fails here.
test("golden: the 20 gene ranges are frozen", () => {
  const frozen = [["headWidth",0.72,1.15],["headHeight",0.72,1.08],["headRoundness",0.15,1],["earShape",0,1],["earSize",0.45,1.2],["earSpread",-0.15,0.35],["eyeSize",5,17],["eyeSpacing",17,33],["eyeHeight",-14,12],["eyeShine",0,1],["mouth",0,1],["mouthWidth",6,22],["blush",0,1],["bodySize",0.62,1.18],["limbLength",0,1],["antenna",0,1],["facePlate",0,1],["hue",14,214],["saturation",0.14,0.6],["lightness",0.36,0.7]];
  assert.deepEqual(GENES.map((g) => [g.key, g.min, g.max]), frozen);
});

test("golden: breeding settings are frozen", () => {
  assert.deepEqual(CONFIG, { activeSize: 16, eliteCount: 8, picksPerGeneration: 40, roundsPerSession: 40, mutationRate: 0.3, mutationSigma: 0.1 });
});

test("golden: the renderer draws the same pictures as when the design was frozen", () => {
  assert.equal(sha(renderSVG(new Array(D).fill(0.5))), "29d9f4674814cb3a3687dd0e7afbedbfa7a563a9f8ee8fbc30e3a0bfeba2d9ba");
  assert.equal(sha(renderSVG(randomGenome(rngFor(7)))), "296d2012d11db5dc0738dc9948e1ff02f676c6cfcdd1d6599922c84458831fb5");
});

test("golden: the test-pick generator output is frozen", () => {
  const { state } = runSimulation({ raters: 3, sessionsPerRater: 10, seed: 20261101 });
  assert.equal(stateFingerprint(state), 2711672101);
  assert.equal(sha(picksCSV(state)), "9c240685b06e02b70264df1914659939e76ab4b5170dc0f087fb6351ba833836");
});
