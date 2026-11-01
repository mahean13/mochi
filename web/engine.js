// Mochi engine: genome, rendering, pair scheduling, preference models and MAP-Elites breeding.
// Pure functions over a plain JSON state, so the same code runs in the browser game
// and in the Node simulation (sim/simulate.mjs). No DOM access in this file.

export const ENGINE_VERSION = "1.1.0";

// ---------------------------------------------------------------- random numbers
// mulberry32: small, fast, deterministic 32-bit generator.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// FNV-1a hash of any list of values, used to derive one independent seed per decision.
export function hashSeed(...parts) {
  let h = 0x811c9dc5;
  const s = parts.join("|");
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function rngFor(...parts) {
  return mulberry32(hashSeed(...parts));
}

export function normal(rng) {
  // Box-Muller transform.
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const lerp = (a, b, t) => a + (b - a) * t;

// ---------------------------------------------------------------- genome
// Every gene is stored on [0, 1] and mapped to a drawing range at render time.
export const GENES = [
  { key: "headWidth", label: "Head width", group: "head", min: 0.72, max: 1.15, desc: "Width of the head" },
  { key: "headHeight", label: "Head height", group: "head", min: 0.72, max: 1.08, desc: "Height of the head" },
  { key: "headRoundness", label: "Head roundness", group: "head", min: 0.15, max: 1.0, desc: "Corner rounding, from boxy to fully round" },
  { key: "earShape", label: "Ear shape", group: "ears", min: 0, max: 1, desc: "Pointed animal ears (0) to rounded machine panels (1)" },
  { key: "earSize", label: "Ear size", group: "ears", min: 0.45, max: 1.2, desc: "Length of the ears" },
  { key: "earSpread", label: "Ear spread", group: "ears", min: -0.15, max: 0.35, desc: "How far the ears tilt outward" },
  { key: "eyeSize", label: "Eye size", group: "eyes", min: 5, max: 17, desc: "Radius of each eye" },
  { key: "eyeSpacing", label: "Eye spacing", group: "eyes", min: 17, max: 33, desc: "Distance from the centre line to each eye" },
  { key: "eyeHeight", label: "Eye height", group: "eyes", min: -14, max: 12, desc: "Eye position, from high on the face (0) to low (1)" },
  { key: "eyeShine", label: "Eye shine", group: "eyes", min: 0, max: 1, desc: "Size of the highlight in each eye" },
  { key: "mouth", label: "Mouth", group: "face", min: 0, max: 1, desc: "No mouth below 0.2, then an increasingly curved smile" },
  { key: "mouthWidth", label: "Mouth width", group: "face", min: 6, max: 22, desc: "Width of the mouth" },
  { key: "blush", label: "Blush", group: "face", min: 0, max: 1, desc: "Strength of the cheek blush" },
  { key: "bodySize", label: "Body size", group: "body", min: 0.62, max: 1.18, desc: "Body size relative to the head" },
  { key: "limbLength", label: "Leg length", group: "body", min: 0, max: 1, desc: "Stubby feet (0) to visible legs (1)" },
  { key: "antenna", label: "Antenna", group: "machine", min: 0, max: 1, desc: "No antenna below 0.3, then a longer antenna" },
  { key: "facePlate", label: "Face plate", group: "machine", min: 0, max: 1, desc: "Size of the light visor around the face" },
  { key: "hue", label: "Hue", group: "colour", min: 14, max: 214, desc: "Body hue in degrees, from warm orange (0) to cool blue (1)" },
  { key: "saturation", label: "Saturation", group: "colour", min: 0.14, max: 0.6, desc: "Colour intensity" },
  { key: "lightness", label: "Lightness", group: "colour", min: 0.36, max: 0.7, desc: "Colour lightness" },
];
export const D = GENES.length; // 20
export const GENE_INDEX = Object.fromEntries(GENES.map((g, i) => [g.key, i]));

export function decode(genes) {
  const o = {};
  GENES.forEach((g, i) => (o[g.key] = lerp(g.min, g.max, genes[i])));
  o._raw = genes;
  return o;
}

export function randomGenome(rng) {
  return Array.from({ length: D }, () => rng());
}

// ---------------------------------------------------------------- rendering
function hsl(h, s, l) {
  return `hsl(${h.toFixed(1)} ${(s * 100).toFixed(1)}% ${(l * 100).toFixed(1)}%)`;
}

// Draw a creature as an SVG string from its genome alone. Same genome, same picture.
export function renderSVG(genes, { size = 200, title = "" } = {}) {
  const p = decode(genes);
  const g = genes;
  const body = hsl(p.hue, p.saturation, p.lightness);
  const bodyDark = hsl(p.hue, p.saturation, Math.max(0.2, p.lightness - 0.14));
  const plate = hsl(p.hue, Math.min(0.35, p.saturation * 0.5), 0.93);
  const blushCol = `rgba(231,140,120,${(0.15 + 0.7 * g[GENE_INDEX.blush]).toFixed(3)})`;
  const ink = "#2A231E";
  const cx = 100;

  // Head geometry.
  const hw = 62 * p.headWidth;
  const hh = 50 * p.headHeight;
  const hcy = 92;
  const rx = Math.max(6, p.headRoundness * Math.min(hw, hh));
  const headTop = hcy - hh;

  // Body and legs.
  const bw = 48 * p.bodySize;
  const bh = 31 * p.bodySize;
  const legLen = 4 + 18 * p.limbLength;
  const footY = Math.min(186, hcy + hh + 2 + bh * 1.2 + legLen);
  const bcy = footY - legLen - bh * 0.55;

  const parts = [];
  if (title) parts.push(`<title>${title}</title>`);

  // Antenna.
  if (g[GENE_INDEX.antenna] >= 0.3) {
    const len = Math.min(headTop - 14, 8 + 30 * ((g[GENE_INDEX.antenna] - 0.3) / 0.7));
    parts.push(`<path d="M${cx} ${headTop + 4} L${cx} ${headTop - len}" stroke="${bodyDark}" stroke-width="5" stroke-linecap="round"/>`);
    parts.push(`<circle cx="${cx}" cy="${headTop - len - 3}" r="6.5" fill="#E7967F"/>`);
  }

  // Ears: blend from a pointed triangle to a rounded panel.
  const es = p.earSize;
  const spread = p.earSpread;
  const earShape = g[GENE_INDEX.earShape];
  for (const side of [-1, 1]) {
    const baseX = cx + side * hw * 0.62;
    const baseY = headTop + hh * 0.45;
    if (earShape < 0.5) {
      const tipX = baseX + side * (spread * 40 + 4);
      const tipY = headTop - 30 * es;
      const innerX = cx + side * hw * 0.12;
      const t = earShape * 2; // 0..1 softening of the tip
      const ctrl = `${lerp(tipX, (baseX + innerX) / 2, t * 0.35)} ${lerp(tipY, (baseY + tipY) / 2, t * 0.35)}`;
      parts.push(`<path d="M${baseX + side * 6} ${baseY} Q${ctrl} ${tipX} ${tipY} Q${ctrl} ${innerX} ${headTop + 6} Z" fill="${body}" stroke="${body}" stroke-width="${2 + 10 * t}" stroke-linejoin="round"/>`);
      parts.push(`<path d="M${lerp(baseX, innerX, 0.2)} ${baseY - 8} L${lerp(tipX, innerX, 0.18)} ${lerp(tipY, baseY, 0.25)} L${lerp(innerX, baseX, 0.15)} ${headTop + 10} Z" fill="${plate}" opacity="${(0.8 - t * 0.5).toFixed(2)}"/>`);
    } else {
      const pw = 22 * es + 6;
      const ph = 46 * es + 10;
      const px = cx + side * (hw + 2 + spread * 18) - pw / 2;
      const py = hcy - ph / 2 - 6;
      const r = lerp(4, pw / 2, (earShape - 0.5) * 2);
      parts.push(`<rect x="${px}" y="${py}" width="${pw}" height="${ph}" rx="${r}" fill="${body}"/>`);
    }
  }

  // Legs and feet.
  for (const side of [-1, 1]) {
    const fx = cx + side * bw * 0.5;
    if (legLen > 7) parts.push(`<rect x="${fx - 6}" y="${footY - legLen}" width="12" height="${legLen}" rx="6" fill="${bodyDark}"/>`);
    parts.push(`<ellipse cx="${fx}" cy="${footY}" rx="15" ry="8" fill="${body}"/>`);
  }

  // Body with a belly panel.
  parts.push(`<ellipse cx="${cx}" cy="${bcy}" rx="${bw}" ry="${bh}" fill="${body}"/>`);
  parts.push(`<ellipse cx="${cx}" cy="${bcy + bh * 0.22}" rx="${bw * 0.55}" ry="${bh * 0.5}" fill="${plate}" opacity="0.85"/>`);

  // Head.
  parts.push(`<rect x="${cx - hw}" y="${hcy - hh}" width="${hw * 2}" height="${hh * 2}" rx="${rx}" fill="${body}"/>`);

  // Face plate (visor).
  const fp = g[GENE_INDEX.facePlate];
  if (fp > 0.08) {
    const fw = hw * lerp(0.45, 0.82, fp);
    const fh = hh * lerp(0.4, 0.78, fp);
    const frx = Math.min(fw, fh) * lerp(0.4, 1, p.headRoundness);
    parts.push(`<rect x="${cx - fw}" y="${hcy + 4 - fh}" width="${fw * 2}" height="${fh * 2}" rx="${frx}" fill="${plate}"/>`);
  }

  // Eyes.
  const er = p.eyeSize;
  const ex = Math.min(p.eyeSpacing, hw - er - 4);
  const ey = hcy + p.eyeHeight;
  const shine = g[GENE_INDEX.eyeShine];
  for (const side of [-1, 1]) {
    const x = cx + side * ex;
    parts.push(`<ellipse cx="${x}" cy="${ey}" rx="${er}" ry="${er * 1.08}" fill="${ink}"/>`);
    if (shine > 0.05) {
      parts.push(`<circle cx="${x + er * 0.35}" cy="${ey - er * 0.4}" r="${(0.12 + 0.28 * shine) * er + 0.6}" fill="#FFFDFA"/>`);
      parts.push(`<circle cx="${x - er * 0.3}" cy="${ey + er * 0.35}" r="${(0.05 + 0.1 * shine) * er}" fill="#FFFDFA" opacity="0.8"/>`);
    }
  }

  // Cheeks.
  const cheekY = ey + er + 6;
  for (const side of [-1, 1]) {
    parts.push(`<ellipse cx="${cx + side * (ex + er * 0.6 + 4)}" cy="${cheekY}" rx="10" ry="6" fill="${blushCol}"/>`);
  }

  // Mouth.
  const m = g[GENE_INDEX.mouth];
  if (m >= 0.2) {
    const curve = 1 + 9 * ((m - 0.2) / 0.8);
    const mw = p.mouthWidth / 2;
    const my = Math.min(hcy + hh - 8, ey + er + 9);
    parts.push(`<path d="M${cx - mw} ${my} Q${cx} ${my + curve} ${cx + mw} ${my}" stroke="${ink}" stroke-width="2.6" fill="none" stroke-linecap="round"/>`);
  }

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="${size}" height="${size}" role="img"${title ? ` aria-label="${title}"` : ""}>${parts.join("")}</svg>`;
}

// ---------------------------------------------------------------- descriptors for MAP-Elites
// Two behaviour descriptors on [0, 1]. They define the archive grid that keeps variety.
//   babyness: large low eyes, a round head and a small body (Lorenz's baby schema)
//   machineness: antenna, face plate and panel ears
export function descriptors(genes) {
  const G = GENE_INDEX;
  const babyness = (genes[G.eyeSize] + genes[G.eyeHeight] + genes[G.headRoundness] + (1 - genes[G.bodySize])) / 4;
  const machineness = (genes[G.antenna] + genes[G.facePlate] + genes[G.earShape]) / 3;
  return [babyness, machineness];
}

export const GRID = 5;
export function cellOf(genes) {
  const [a, b] = descriptors(genes);
  const i = Math.min(GRID - 1, Math.floor(a * GRID));
  const j = Math.min(GRID - 1, Math.floor(b * GRID));
  return `${i},${j}`;
}

// ---------------------------------------------------------------- creature-level Bradley-Terry
// Minorisation-maximisation (Hunter, 2004) with a weak prior: every item gets one virtual
// win and one virtual loss against a reference item of strength 1, so scores stay finite.
// picks: [{winner, loser}]. Returns {id: log strength}. Cost O(iterations * picks).
export function fitItemBT(items, picks, { iterations = 200, tol = 1e-9 } = {}) {
  const ids = [...items];
  const idx = new Map(ids.map((id, i) => [id, i]));
  const n = ids.length;
  const wins = new Float64Array(n).fill(1); // prior virtual win
  const pairs = [];
  for (const p of picks) {
    const w = idx.get(p.winner);
    const l = idx.get(p.loser);
    if (w === undefined || l === undefined) continue;
    wins[w] += 1;
    pairs.push([w, l]);
  }
  let s = new Float64Array(n).fill(1);
  for (let it = 0; it < iterations; it++) {
    const denom = new Float64Array(n);
    for (const [a, b] of pairs) {
      const inv = 1 / (s[a] + s[b]);
      denom[a] += inv;
      denom[b] += inv;
    }
    // Prior comparisons against the reference item of strength 1 (two per item).
    const next = new Float64Array(n);
    let change = 0;
    let logSum = 0;
    for (let i = 0; i < n; i++) {
      next[i] = wins[i] / (denom[i] + 2 / (s[i] + 1));
      logSum += Math.log(next[i]);
    }
    const gm = Math.exp(logSum / n);
    for (let i = 0; i < n; i++) {
      next[i] /= gm;
      change = Math.max(change, Math.abs(Math.log(next[i]) - Math.log(s[i])));
    }
    s = next;
    if (change < tol) break;
  }
  const out = {};
  ids.forEach((id, i) => (out[id] = Math.log(s[i])));
  return out;
}

// ---------------------------------------------------------------- feature-level Bradley-Terry
// P(left chosen) = sigmoid(b + w . (x_left - x_right)), with x = genes - 0.5.
// This is a Bradley-Terry model whose strengths are built from features, fitted as an
// L2-regularised logistic regression by Newton's method. b captures any left/right bias.
// rows: [{xl: genes, xr: genes, y: 1 if left chosen else 0}]. Cost per iteration O(m d^2 + d^3).
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export function designRow(xl, xr) {
  const z = new Float64Array(D + 1);
  z[0] = 1;
  for (let k = 0; k < D; k++) z[k + 1] = xl[k] - xr[k];
  return z;
}

function solve(A, b) {
  // Gaussian elimination with partial pivoting; A is small (21 x 21).
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let acc = M[r][n];
    for (let k = r + 1; k < n; k++) acc -= M[r][k] * x[k];
    x[r] = acc / M[r][r];
  }
  return x;
}

export function fitFeatureBT(rows, { lambda = 1.0, iterations = 50, tol = 1e-8 } = {}) {
  const P = D + 1;
  let w = new Array(P).fill(0);
  const Z = rows.map((r) => designRow(r.xl, r.xr));
  const y = rows.map((r) => r.y);
  for (let it = 0; it < iterations; it++) {
    const H = Array.from({ length: P }, () => new Array(P).fill(0));
    const grad = new Array(P).fill(0);
    for (let i = 0; i < Z.length; i++) {
      const z = Z[i];
      let eta = 0;
      for (let k = 0; k < P; k++) eta += w[k] * z[k];
      const p = sigmoid(eta);
      const r = y[i] - p;
      const v = p * (1 - p);
      for (let a = 0; a < P; a++) {
        grad[a] += r * z[a];
        const va = v * z[a];
        for (let b = a; b < P; b++) H[a][b] += va * z[b];
      }
    }
    for (let a = 0; a < P; a++) {
      for (let b = 0; b < a; b++) H[a][b] = H[b][a];
      if (a > 0) {
        // No penalty on the side-bias intercept.
        H[a][a] += lambda;
        grad[a] -= lambda * w[a];
      }
      H[a][a] += 1e-9;
    }
    const step = solve(H, grad);
    let maxStep = 0;
    w = w.map((wk, k) => {
      maxStep = Math.max(maxStep, Math.abs(step[k]));
      return wk + step[k];
    });
    if (maxStep < tol) break;
  }
  return { bias: w[0], weights: w.slice(1) };
}

export function predictLeft(model, xl, xr) {
  let eta = model.bias;
  for (let k = 0; k < D; k++) eta += model.weights[k] * (xl[k] - xr[k]);
  return sigmoid(eta);
}

// ---------------------------------------------------------------- study configuration
export const LINEAGES = {
  home: { id: "home", question: "Which would you rather have at home?", short: "Home" },
  trust: { id: "trust", question: "Which would you trust?", short: "Trust" },
};
export const CONFIG = {
  activeSize: 16, // creatures shown in one generation
  eliteCount: 8, // carried over from the archive each generation
  picksPerGeneration: 40, // one ten-minute session
  roundsPerSession: 40,
  mutationRate: 0.3, // chance that each gene mutates
  mutationSigma: 0.1, // standard deviation of a mutation on [0, 1]
};

// ---------------------------------------------------------------- state
export function newState(seed = 20261101) {
  const state = {
    engine: ENGINE_VERSION,
    seed,
    created: null,
    config: { ...CONFIG },
    creatures: {},
    counts: {},
    lineages: {},
    picks: [],
    sessions: [],
  };
  for (const id of Object.keys(LINEAGES)) {
    const rng = rngFor(seed, id, "gen", 0);
    const active = [];
    for (let i = 0; i < CONFIG.activeSize; i++) {
      const c = makeCreature(state, id, 0, randomGenome(rng), null);
      active.push(c.id);
    }
    state.lineages[id] = { id, gen: 0, active, archive: {}, scores: {}, history: [] };
  }
  return state;
}

// Creature ids are "<H|T><generation>-<index>". A counter per lineage and generation keeps
// creation O(1); older saved states without the counter fall back to counting once.
function makeCreature(state, lineage, gen, genes, parents) {
  state.counts = state.counts || {};
  const key = `${lineage}:${gen}`;
  if (state.counts[key] === undefined) {
    state.counts[key] = Object.values(state.creatures).filter((c) => c.lineage === lineage && c.gen === gen).length;
  }
  const n = state.counts[key]++;
  const id = `${lineage === "home" ? "H" : "T"}${gen}-${String(n + 1).padStart(2, "0")}`;
  const c = { id, lineage, gen, genes: genes.map((v) => Math.round(clamp01(v) * 1e6) / 1e6), parents, cell: null };
  c.cell = cellOf(c.genes);
  state.creatures[id] = c;
  return c;
}

export function lineagePicks(state, lineageId) {
  return state.picks.filter((p) => p.lineage === lineageId);
}

// Picks of the lineage's current generation, oldest first. Picks are appended in time order,
// so the current generation is a suffix of the lineage's picks: scan backwards and stop at the
// first earlier-generation pick. Cost O(picks since this generation began), not O(all picks).
export function currentGenerationPicks(state, lineageId) {
  const L = state.lineages[lineageId];
  const out = [];
  for (let i = state.picks.length - 1; i >= 0; i--) {
    const p = state.picks[i];
    if (p.lineage !== lineageId) continue;
    if (p.gen !== L.gen) break;
    out.push(p);
  }
  return out.reverse();
}

export function picksThisGeneration(state, lineageId) {
  return currentGenerationPicks(state, lineageId).length;
}

// ---------------------------------------------------------------- pair scheduling
// Balanced design: show the least-compared creature next, against the least-compared
// partner it has not yet met this generation. Ties are broken by a seeded draw, and
// the screen side is randomised to counterbalance any left/right bias.
// Cost O(g + n log n): g picks so far in this generation (at most 40), n = 16 creatures.
export function nextPair(state, lineageId) {
  const L = state.lineages[lineageId];
  const genPicks = currentGenerationPicks(state, lineageId);
  const rng = rngFor(state.seed, lineageId, L.gen, "pair", genPicks.length);
  const count = Object.fromEntries(L.active.map((id) => [id, 0]));
  const met = new Set();
  for (const p of genPicks) {
    count[p.left] = (count[p.left] || 0) + 1;
    count[p.right] = (count[p.right] || 0) + 1;
    met.add([p.left, p.right].sort().join("~"));
  }
  const jitter = Object.fromEntries(L.active.map((id) => [id, rng()]));
  const order = [...L.active].sort((a, b) => count[a] - count[b] || jitter[a] - jitter[b]);
  const a = order[0];
  let partners = order.slice(1).filter((b) => !met.has([a, b].sort().join("~")));
  if (partners.length === 0) partners = order.slice(1);
  const b = partners[0];
  return rng() < 0.5 ? { left: a, right: b } : { left: b, right: a };
}

// ---------------------------------------------------------------- recording a pick
export function recordPick(state, { lineage, left, right, chosen, session, participant, round, ms, at }) {
  const L = state.lineages[lineage];
  const pick = {
    n: state.picks.length + 1,
    lineage,
    gen: L.gen,
    session,
    participant,
    round,
    left,
    right,
    chosen,
    side: chosen === left ? "left" : "right",
    ms: ms ?? null,
    at: at ?? null,
  };
  state.picks.push(pick);
  let advanced = false;
  if (picksThisGeneration(state, lineage) >= state.config.picksPerGeneration) {
    advanceGeneration(state, lineage);
    advanced = true;
  }
  return { pick, advanced };
}

// ---------------------------------------------------------------- MAP-Elites generation step
// 1. Refit creature-level Bradley-Terry on every pick in the lineage so far.
// 2. Place each scored creature in its descriptor cell, keeping the higher score.
// 3. Next generation = up to eliteCount elites drawn from the archive + offspring
//    from two archive parents (uniform crossover, Gaussian mutation).
export function advanceGeneration(state, lineageId) {
  const L = state.lineages[lineageId];
  const picks = lineagePicks(state, lineageId).map((p) => ({
    winner: p.chosen,
    loser: p.chosen === p.left ? p.right : p.left,
  }));
  const seen = new Set();
  picks.forEach((p) => {
    seen.add(p.winner);
    seen.add(p.loser);
  });
  const scores = fitItemBT(seen, picks);
  L.scores = scores;

  // Rebuild the archive from all compared creatures with their refreshed scores.
  const archive = {};
  for (const id of seen) {
    const c = state.creatures[id];
    const cur = archive[c.cell];
    if (!cur || scores[id] > scores[cur]) archive[c.cell] = id;
  }
  L.archive = archive;

  const rng = rngFor(state.seed, lineageId, "gen", L.gen + 1);
  const elites = Object.values(archive);
  // Elites: the best-scoring elites, so strong creatures keep being tested against newcomers.
  const keep = [...elites].sort((a, b) => scores[b] - scores[a]).slice(0, state.config.eliteCount);
  const nextGen = L.gen + 1;
  const offspring = [];
  const parentsPool = elites.length >= 2 ? elites : [...seen];
  while (keep.length + offspring.length < state.config.activeSize) {
    const pa = parentsPool[Math.floor(rng() * parentsPool.length)];
    let pb = parentsPool[Math.floor(rng() * parentsPool.length)];
    if (pb === pa && parentsPool.length > 1) pb = parentsPool[(parentsPool.indexOf(pa) + 1) % parentsPool.length];
    const ga = state.creatures[pa].genes;
    const gb = state.creatures[pb].genes;
    const child = ga.map((v, k) => {
      let x = rng() < 0.5 ? v : gb[k];
      if (rng() < state.config.mutationRate) x += state.config.mutationSigma * normal(rng);
      return clamp01(x);
    });
    offspring.push(makeCreature(state, lineageId, nextGen, child, [pa, pb]).id);
  }
  L.history.push({
    gen: L.gen,
    picks: picksThisGeneration(state, lineageId),
    archiveCells: Object.keys(archive).length,
    best: keep[0] ?? null,
    bestScore: keep[0] ? scores[keep[0]] : null,
    meanGenes: meanGenes(L.active.map((id) => state.creatures[id].genes)),
  });
  L.gen = nextGen;
  L.active = [...keep, ...offspring];
  return L;
}

export function meanGenes(list) {
  const m = new Array(D).fill(0);
  if (!list.length) return m;
  for (const g of list) for (let k = 0; k < D; k++) m[k] += g[k] / list.length;
  return m.map((v) => Math.round(v * 1e4) / 1e4);
}

// ---------------------------------------------------------------- export helpers
export function picksToRows(state, lineageId = null) {
  return state.picks
    .filter((p) => !lineageId || p.lineage === lineageId)
    .map((p) => ({
      xl: state.creatures[p.left].genes.map((v) => v - 0.5),
      xr: state.creatures[p.right].genes.map((v) => v - 0.5),
      y: p.chosen === p.left ? 1 : 0,
      pick: p,
    }));
}

export function picksCSV(state) {
  const head = ["n", "lineage", "gen", "session", "participant", "round", "left", "right", "chosen", "side", "ms", "at"];
  const geneHead = [];
  for (const s of ["left", "right"]) for (const g of GENES) geneHead.push(`${s}_${g.key}`);
  const lines = [[...head, ...geneHead].join(",")];
  for (const p of state.picks) {
    const row = head.map((h) => (p[h] === null || p[h] === undefined ? "" : String(p[h]).replace(/,/g, ";")));
    for (const s of ["left", "right"]) for (const v of state.creatures[p[s]].genes) row.push(v);
    lines.push(row.join(","));
  }
  return lines.join("\n");
}

export function creaturesCSV(state) {
  const lines = [["id", "lineage", "gen", "parent_a", "parent_b", "cell", ...GENES.map((g) => g.key)].join(",")];
  for (const c of Object.values(state.creatures)) {
    lines.push([c.id, c.lineage, c.gen, c.parents?.[0] ?? "", c.parents?.[1] ?? "", `"${c.cell}"`, ...c.genes].join(","));
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------- replay
// The whole study is a deterministic function of the seed and the ordered pick log.
// Replaying the log rebuilds every generation, archive and creature exactly.
export function replay(seed, picks) {
  const state = newState(seed);
  for (const p of [...picks].sort((a, b) => a.n - b.n)) {
    const L = state.lineages[p.lineage];
    if (!state.creatures[p.left] || !state.creatures[p.right]) throw new Error(`Pick ${p.n} names a creature that does not exist at replay`);
    if (p.gen !== L.gen) throw new Error(`Pick ${p.n} expected generation ${p.gen}, replay is at ${L.gen}`);
    recordPick(state, p);
  }
  return state;
}

// ---------------------------------------------------------------- consistency check
export function stateFingerprint(state) {
  return hashSeed(JSON.stringify(Object.values(state.lineages).map((L) => [L.id, L.gen, L.active, L.archive])), Object.keys(state.creatures).length);
}
