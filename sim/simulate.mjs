// Simulated players for Mochi.
// Each simulated rater has known feature weights (the "true" preference). They play the
// real engine exactly as a volunteer would, through nextPair and recordPick. The analysis
// then checks whether the preference model recovers those weights from the picks alone.
//
// Usage: node sim/simulate.mjs [outDir] [raters] [sessionsPerRater] [seed]
import fs from "node:fs";
import path from "node:path";
import {
  GENES, GENE_INDEX, D, LINEAGES, newState, nextPair, recordPick, rngFor, normal, sigmoid,
  picksCSV, creaturesCSV,
} from "../web/engine.js";

// True preference weights per question, on genes centred at 0.5. Genes not listed have weight 0.
// These are assumptions for the simulation only, chosen to resemble the baby-schema literature.
export const TRUE_WEIGHTS = {
  home: { eyeSize: 2.4, eyeHeight: 1.6, headRoundness: 2.0, blush: 1.2, mouth: 1.4, bodySize: -1.0, lightness: 0.8, saturation: 0.6 },
  trust: { mouth: 2.2, eyeSize: 0.8, facePlate: 1.2, headRoundness: 0.8, lightness: 1.0, saturation: -1.0, antenna: -0.8, eyeHeight: 0.6 },
};
export const RATER_SPREAD = 0.4; // standard deviation of each rater's deviation from the shared weights
export const SIDE_BIAS = 0.15; // small preference for the left creature, in logits

export function weightVector(spec) {
  const w = new Array(D).fill(0);
  for (const [k, v] of Object.entries(spec)) w[GENE_INDEX[k]] = v;
  return w;
}

export function runSimulation({ raters = 3, sessionsPerRater = 10, seed = 20261101 } = {}) {
  const state = newState(seed);
  state.created = "simulation";
  const people = [];
  for (let r = 0; r < raters; r++) {
    const rng = rngFor(seed, "rater", r);
    const w = {};
    for (const lin of Object.keys(LINEAGES)) {
      w[lin] = weightVector(TRUE_WEIGHTS[lin]).map((v) => v + RATER_SPREAD * normal(rng));
    }
    people.push({ code: `S${String(r + 1).padStart(2, "0")}`, w });
  }
  // Sessions alternate lineages so each simulated rater plays both questions, like the study plan.
  let sessionNo = 0;
  for (let s = 0; s < sessionsPerRater; s++) {
    for (const person of people) {
      sessionNo++;
      const lineage = sessionNo % 2 === 1 ? "home" : "trust";
      const sessionId = `sim-${String(sessionNo).padStart(3, "0")}`;
      state.sessions.push({ id: sessionId, participant: person.code, lineage, simulated: true });
      const rng = rngFor(seed, "choices", sessionId);
      for (let round = 1; round <= state.config.roundsPerSession; round++) {
        const { left, right } = nextPair(state, lineage);
        const gl = state.creatures[left].genes;
        const gr = state.creatures[right].genes;
        let eta = SIDE_BIAS;
        for (let k = 0; k < D; k++) eta += person.w[lineage][k] * (gl[k] - gr[k]);
        const chosen = rng() < sigmoid(eta) ? left : right;
        const ms = Math.round(900 + 1800 * rng());
        recordPick(state, { lineage, left, right, chosen, session: sessionId, participant: person.code, round, ms, at: null });
      }
    }
  }
  return { state, people };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const out = process.argv[2] || "data/sim";
  const raters = Number(process.argv[3] || 3);
  const sessions = Number(process.argv[4] || 10);
  const seed = Number(process.argv[5] || 20261101);
  const { state } = runSimulation({ raters, sessionsPerRater: sessions, seed });
  fs.mkdirSync(out, { recursive: true });
  fs.writeFileSync(path.join(out, "state.json"), JSON.stringify(state));
  fs.writeFileSync(path.join(out, "picks.csv"), picksCSV(state));
  fs.writeFileSync(path.join(out, "creatures.csv"), creaturesCSV(state));
  const truth = { genes: GENES.map((g) => g.key), side_bias: SIDE_BIAS, rater_spread: RATER_SPREAD, weights: {} };
  for (const lin of Object.keys(LINEAGES)) truth.weights[lin] = weightVector(TRUE_WEIGHTS[lin]);
  fs.writeFileSync(path.join(out, "truth.json"), JSON.stringify(truth, null, 2));
  const gens = Object.fromEntries(Object.values(state.lineages).map((L) => [L.id, L.gen]));
  console.log(`picks=${state.picks.length} sessions=${state.sessions.length} generations=${JSON.stringify(gens)} creatures=${Object.keys(state.creatures).length}`);
}
