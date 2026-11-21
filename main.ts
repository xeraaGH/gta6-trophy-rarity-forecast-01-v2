// gta6-trophy-rarity-forecast-01 entry #2: "Cohort Rarity Forecaster v2" (Node 22, stdlib only).
//
// Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its Exophase
// earn rate (percent of tracked players) at the case's target, as five quantiles at
// levels [0.05, 0.25, 0.5, 0.75, 0.95]. Scored by mean pinball loss on the logit scale.
//
// MEDIAN — cohort-delay model. Players arrive over time (N only grows); a player who
// bought at time s earns a trophy by t with probability R*(1-exp(-(t-s)/tau)).
// Earners at t are the cohorts summed with that delay; rate = earners / players tracked.
// R and tau are fitted per trophy: tau by grid search over 12 values on full-history
// fit error, R by least squares. The median saturates at R. Future arrivals between the
// last read and the target are projected at the recent arrival rate (constant), split
// into sub-cohorts, so the forecast denominator is the projected players tracked.
// Story-chain medians (chapters 1-4, then the final mission) are forced non-increasing
// by isotonic regression; the platinum median is capped at the smallest median.
//
// SPREAD — calibrated, per trophy: holdout error (fit on the first 70% of reads, scored
// on the last 30%) + horizon term + biggest refresh-jump term + disagreement term
// (range of the target forecast across tau values that fit almost as well), combined by
// root-sum-of-squares, floored at 0.35 logit. A FLAT history (rates barely moving) gets
// a wide band (>= 1.0 logit) and its median shifted DOWN 0.3 logit — a flat page is a
// page that has not refreshed yet, and the refresh drops rates.
//
// ENTRY #2 CHANGE: the calibrated spread is multiplied by SPREAD_SCALE = 0.85. Entry #1's
// practice grade over-covered (coverage_90 0.9636 vs nominal 0.90) — its band was
// systematically too wide — so #2 tightens it toward the nominal 90%.
//
// Edge cases: rates clamped to 0.01..99.99 before any logit; null rates skipped; 0%
// trophies get a small prior R so they can start earning; 100% trophies saturate at
// 99.99; histories with fewer than 3 reads fall back to persistence with a wide band.
//
// Runs offline, no network, deterministic (no randomness anywhere).
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in
// for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Tau grid (hours) for the delay fit: 12 values, fast to very slow. */
const TAU_GRID_H = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
/** Entry #2: narrow the calibrated spread by this factor. */
const SPREAD_SCALE = 0.85;
/** Spread floor, in logit units. */
const SPREAD_FLOOR = 0.35;
/** Flat-history band floor, in logit units. */
const FLAT_SPREAD = 1.0;
/** Flat-history median shift, in logit units (down). */
const FLAT_SHIFT = -0.3;
/** Horizon term coefficient, logit per sqrt(day). */
const HORIZON_COEF = 0.15;
/** Default holdout error when the history is too short to hold out. */
const DEFAULT_HOLDOUT = 0.5;
/** Prior R for a trophy never earned in the history (so it can start earning). */
const NEVER_EARNED_R = 0.005;
/** Prior tau for a never-earned trophy (hours). */
const NEVER_EARNED_TAU_H = 512;
/** Story chain, in order: chapters 1-4, then the final mission. */
const STORY_CHAIN = [3, 4, 5, 6, 7];
/** The platinum trophy. */
const PLATINUM_ID = 1;

interface Trophy { id: number; name: string; description: string; tier: string; points: number; secret: boolean }
interface Read { slot: string; at: string; players_tracked: number; trophies_earned: number; rates: Array<number | null> }
interface Case {
  id: string; at: string; horizon_minutes: number; target: string;
  levels: number[]; trophies: Trophy[]; history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => { const p = clampPct(pct) / 100; return Math.log(p / (1 - p)); };
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const parseTime = (s: string): number => Date.parse(s);

/** Pool-adjacent-violators isotonic regression, non-increasing, on values (logits). */
function isotonicDecreasing(vals: number[]): number[] {
  const out = vals.slice();
  const w = new Array<number>(vals.length).fill(1);
  let i = 0;
  while (i < out.length - 1) {
    if (out[i] < out[i + 1]) {
      // Violation: pool i and i+1.
      const pooled = (out[i] * w[i] + out[i + 1] * w[i + 1]) / (w[i] + w[i + 1]);
      out[i] = pooled; out[i + 1] = pooled;
      w[i] = w[i] + w[i + 1]; w[i + 1] = w[i];
      if (i > 0) i--;
    } else i++;
  }
  return out;
}

interface Fit { tauH: number; R: number; sse: number }

/**
 * Fit the cohort-delay model for one trophy on the given reads.
 * timesH: read times in hours since the first read. N: players tracked per read.
 * E: earners per read (null where the rate was null).
 * Returns the best (tau, R) by least-squares fit error, plus all fits for disagreement.
 */
function fitCohort(timesH: number[], N: number[], E: Array<number | null>): { best: Fit; fits: Fit[] } {
  const K = timesH.length;
  // Cohort sizes: c_0 = N_0, c_j = N_j - N_{j-1} (clamped >= 0).
  const c: number[] = [];
  for (let j = 0; j < K; j++) c.push(j === 0 ? Math.max(0, N[0]) : Math.max(0, N[j] - N[j - 1]));
  const fits: Fit[] = [];
  for (const tauH of TAU_GRID_H) {
    // A_k(tau) = N_k - B_k, B_k = sum_j c_j exp(-(t_k - t_j)/tau), computed incrementally.
    const A: number[] = [];
    let B = 0;
    for (let k = 0; k < K; k++) {
      if (k > 0) B *= Math.exp(-(timesH[k] - timesH[k - 1]) / tauH);
      B += c[k];
      A.push(N[k] - B);
    }
    // Least-squares R over reads with a known rate.
    let num = 0, den = 0;
    for (let k = 0; k < K; k++) {
      if (E[k] === null) continue;
      num += (E[k] as number) * A[k];
      den += A[k] * A[k];
    }
    let R = den > 0 ? num / den : 0;
    R = Math.min(1, Math.max(0, R));
    let sse = 0;
    for (let k = 0; k < K; k++) {
      if (E[k] === null) continue;
      const d = (E[k] as number) - R * A[k];
      sse += d * d;
    }
    fits.push({ tauH, R, sse });
  }
  fits.sort((a, b) => a.sse - b.sse);
  return { best: fits[0], fits };
}

/** Predicted earners at time tH (hours) for a fitted (tau, R), given cohorts. */
function predictEarners(tH: number, timesH: number[], c: number[], tauH: number, R: number): number {
  let B = 0;
  let N = 0;
  for (let j = 0; j < timesH.length; j++) {
    if (timesH[j] > tH) break;
    B += c[j] * Math.exp(-(tH - timesH[j]) / tauH);
    N += c[j];
  }
  return R * (N - B);
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  // Validate the case before forecasting: a malformed case must fail loudly
  // here (and be caught in testing) rather than silently produce a bad forecast.
  if (c.history.length === 0) throw new Error(`case ${c.id}: history is empty`);
  if (c.trophies.length === 0) throw new Error(`case ${c.id}: no trophies`);
  for (let k = 0; k < c.history.length; k++) {
    const r = c.history[k];
    if (!Number.isFinite(r.players_tracked) || r.players_tracked <= 0) {
      throw new Error(`case ${c.id}: read ${k} has invalid players_tracked ${r.players_tracked}`);
    }
    if (!Array.isArray(r.rates) || r.rates.length !== c.trophies.length) {
      throw new Error(`case ${c.id}: read ${k} has ${Array.isArray(r.rates) ? r.rates.length : 'no'} rates for ${c.trophies.length} trophies`);
    }
    if (!Number.isFinite(parseTime(r.at))) throw new Error(`case ${c.id}: read ${k} has an unparseable time ${r.at}`);
    for (const v of r.rates) {
      if (v !== null && (!Number.isFinite(v) || v < 0 || v > 100)) {
        throw new Error(`case ${c.id}: read ${k} has an out-of-range rate ${v}`);
      }
    }
  }
  if (!Number.isFinite(parseTime(c.target))) throw new Error(`case ${c.id}: unparseable target ${c.target}`);
  if (parseTime(c.target) < parseTime(c.history[c.history.length - 1].at)) {
    throw new Error(`case ${c.id}: target is before the last read`);
  }

  const K = c.history.length;
  const timesH = c.history.map((r) => parseTime(r.at) / 3600000);
  const t0 = timesH[0];
  const relH = timesH.map((t) => t - t0);
  const N = c.history.map((r) => r.players_tracked);
  const targetH = parseTime(c.target) / 3600000 - t0;
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;

  // Cohorts (shared across trophies).
  const cohorts: number[] = [];
  for (let j = 0; j < K; j++) cohorts.push(j === 0 ? Math.max(0, N[0]) : Math.max(0, N[j] - N[j - 1]));

  // Future arrivals: recent arrival rate (last 24 h of reads, or all reads),
  // projected flat to the target, split into 12 sub-cohorts.
  let futureTimes: number[] = [];
  let futureSizes: number[] = [];
  const lastH = relH[K - 1];
  if (targetH > lastH && K >= 1) {
    let wStart = 0;
    for (let j = 0; j < K; j++) if (relH[j] >= lastH - 24) { wStart = j; break; }
    const spanH = Math.max(0.001, lastH - relH[wStart]);
    const lambda = Math.max(0, (N[K - 1] - N[wStart]) / spanH); // players per hour
    const dt = targetH - lastH;
    const projected = lambda * dt;
    const M = 12;
    for (let m = 0; m < M; m++) {
      futureTimes.push(lastH + ((m + 0.5) / M) * dt);
      futureSizes.push(projected / M);
    }
  }
  const allTimes = relH.concat(futureTimes);
  const allCohorts = cohorts.concat(futureSizes);
  const nTarget = N[K - 1] + futureSizes.reduce((a, b) => a + b, 0);

  // First pass: per-trophy median (logit) and spread, before the chain constraints.
  const medLogit = new Map<number, number>();
  const spread = new Map<number, number>();

  for (const tr of c.trophies) {
    const tid = tr.id;
    const rates: Array<number | null> = c.history.map((r) => r.rates[tid - 1] ?? null);
    const known = rates.filter((r): r is number => r !== null);
    const E: Array<number | null> = rates.map((r, k) => (r === null ? null : (r / 100) * N[k]));

    const flat = known.length > 0 && Math.max(...known) - Math.min(...known) < 0.05;
    const neverEarned = known.length > 0 && known.every((r) => r <= 0);
    const allHundred = known.length > 0 && known.every((r) => r >= 100);

    let med: number; // median rate in percent
    let sigma: number;

    if (K < 3 || allHundred) {
      // Too little history to fit, or saturated: persistence, wide band.
      med = known.length > 0 ? known[known.length - 1] : 0.01;
      sigma = Math.max(SPREAD_FLOOR, 1.0);
    } else if (neverEarned) {
      // Never earned yet: a small prior so it can start earning.
      const eT = NEVER_EARNED_R * (nTarget - allCohorts.reduce((acc, cj, j) => acc + cj * Math.exp(-(targetH - allTimes[j]) / NEVER_EARNED_TAU_H), 0));
      med = clampPct((100 * eT) / Math.max(1, nTarget));
      sigma = Math.max(SPREAD_FLOOR, 1.0);
    } else {
      const { best, fits } = fitCohort(relH, N, E);
      // Forecast earners at the target with the fitted tau/R.
      let B = 0;
      for (let j = 0; j < allTimes.length; j++) B += allCohorts[j] * Math.exp(-(targetH - allTimes[j]) / best.tauH);
      const nAll = allCohorts.reduce((a, b) => a + b, 0);
      const eT = best.R * (nAll - B);
      med = clampPct((100 * eT) / Math.max(1, nTarget));

      // --- spread components (logit units) ---
      // (a) Holdout error: fit on the first 70% of reads, score on the last 30%.
      // The prediction at a held-out read uses every cohort that arrived by then.
      let holdout = DEFAULT_HOLDOUT;
      const split = Math.floor(K * 0.7);
      if (split >= 2 && K - split >= 1) {
        const sub = fitCohort(relH.slice(0, split), N.slice(0, split), E.slice(0, split));
        const errs: number[] = [];
        for (let k = split; k < K; k++) {
          if (E[k] === null) continue;
          const pred = predictEarners(relH[k], relH, cohorts, sub.best.tauH, sub.best.R);
          const predRate = clampPct((100 * pred) / Math.max(1, N[k]));
          errs.push(Math.abs(logit(rates[k] as number) - logit(predRate)));
        }
        if (errs.length > 0) holdout = errs.reduce((a, b) => a + b, 0) / errs.length;
      }
      // (b) Horizon term: uncertainty grows with the horizon.
      const horizon = HORIZON_COEF * Math.sqrt(horizonDays);
      // (c) Biggest refresh-jump term: the largest UNEXPECTED consecutive-read move —
      // the observed logit change minus the smooth model's predicted logit change.
      // Smooth growth cancels out; a page refresh (a level shift the model cannot
      // predict) shows up here. Only reads where both rates are established (>= 0.1%).
      let jump = 0;
      {
        const predLogit: number[] = [];
        for (let k = 0; k < K; k++) {
          const pred = predictEarners(relH[k], relH, cohorts, best.tauH, best.R);
          predLogit.push(logit(clampPct((100 * pred) / Math.max(1, N[k]))));
        }
        for (let k = 1; k < K; k++) {
          if (rates[k] === null || rates[k - 1] === null) continue;
          if ((rates[k] as number) < 0.1 || (rates[k - 1] as number) < 0.1) continue;
          const dObs = logit(rates[k] as number) - logit(rates[k - 1] as number);
          const dPred = predLogit[k] - predLogit[k - 1];
          jump = Math.max(jump, Math.abs(dObs - dPred));
        }
      }
      // (d) Disagreement: target-forecast logit range across tau values that fit
      // almost as well as the best (SSE within 1.5x).
      let lo = Infinity, hi = -Infinity;
      for (const f of fits) {
        if (f.sse > 1.5 * best.sse + 1e-12) continue;
        let Bj = 0;
        for (let j = 0; j < allTimes.length; j++) Bj += allCohorts[j] * Math.exp(-(targetH - allTimes[j]) / f.tauH);
        const ej = f.R * (nAll - Bj);
        const lj = logit(clampPct((100 * ej) / Math.max(1, nTarget)));
        if (lj < lo) lo = lj;
        if (lj > hi) hi = lj;
      }
      const disagreement = hi > lo ? (hi - lo) / 2 : 0;

      sigma = Math.sqrt(holdout * holdout + horizon * horizon + jump * jump + disagreement * disagreement);
      sigma = Math.max(sigma, SPREAD_FLOOR);
    }

    let ml = logit(med);
    if (flat) {
      sigma = Math.max(sigma, FLAT_SPREAD);
      ml += FLAT_SHIFT;
    }
    medLogit.set(tid, ml);
    spread.set(tid, sigma * SPREAD_SCALE);
  }

  // Story chain: medians non-increasing (isotonic regression on logits).
  const chainIdx = STORY_CHAIN.filter((id) => medLogit.has(id));
  if (chainIdx.length > 1) {
    const fixed = isotonicDecreasing(chainIdx.map((id) => medLogit.get(id) as number));
    chainIdx.forEach((id, i) => medLogit.set(id, fixed[i]));
  }
  // Platinum: cannot exceed any trophy's rate.
  if (medLogit.has(PLATINUM_ID)) {
    let minOther = Infinity;
    for (const [id, ml] of medLogit) if (id !== PLATINUM_ID && ml < minOther) minOther = ml;
    if (minOther < Infinity) medLogit.set(PLATINUM_ID, Math.min(medLogit.get(PLATINUM_ID) as number, minOther));
  }

  // Quantiles.
  const rows: Array<{ id: number; q: number[] }> = [];
  for (const tr of c.trophies) {
    const ml = medLogit.get(tr.id) as number;
    const s = spread.get(tr.id) as number;
    const q = Z.map((z) => round4(clampPct(invLogit(ml + z * s))));
    // Enforce non-decreasing (rounding can only tie, never invert, but be safe).
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
    rows.push({ id: tr.id, q });
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const rows = forecastCase(c);
  const out = JSON.stringify({ trophies: rows });
  if (Buffer.byteLength(out, 'utf8') > 65536) throw new Error(`${id}.out would exceed 65536 bytes`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), out);
}
