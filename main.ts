// gta6-twitch-ccu-forecast-01 — launch-day Twitch CCU forecaster (Node 22+, stdlib only, no network).
//
// Method: per-hour log-ratio regression on the case's own comparables. For a past
// launch with a complete release-day curve, hour-1 viewers v1 predict the
// rest: log(peak/v1) correlates -0.82 with log(v1) (a huge hour 1 stays
// flat; a quiet one explodes by daytime). So per hour h = 2..24 fit
//   ln(v_h/v1) = a_h + b_h * ln(v1) + c_h * ln(pre/v1)
// over the pool and predict the target at its own hour-1 print p1, with
// log p1 clamped to the pool's observed range (a GTA-scale hour 1 above
// every past launch is forecast like the biggest launches, not by
// extrapolating the slope past the data). The second regressor is the
// launch's OWN state just before midnight: pre = viewers in the last
// recorded hour before the release day starts, so ln(pre/v1) says where
// the launch sits on its early-unlock surge-decay at 00:00 UTC — a launch
// whose pre-midnight surge is still high decays differently through the
// day than one whose surge already spent itself. (Averaging the last 3-6
// pre-midnight hours instead dilutes this to noise; the single last hour
// is the signal.) Both regressors are centred, so the fit never shifts
// the level when a slope is shrunk or a regressor drops out.
// The forecast is the MEDIAN of FIVE centres' quantiles, each with the
// same spread — median-of-quantiles (Vincentization with the median), so a
// single member's bad extrapolation can never drag a quantile with it:
// the median is the middle member at each (hour, tau), not an average.
// The five members are two 2-regressor centres (v8) plus one 1-regressor
// centre (the v6 structure, no pre-midnight regressor, raw slopes, big
// pool = top half by v1, Huber k = 1.345) plus two DIVERSE centres that
// only joined at v12, each measured to improve BOTH practice and
// leave-one-out as a median member:
//  (1) a big-launch blend: the full-pool centre averaged 50/50 with a
//      centre fitted only on the top three-quarters of the pool by v1
//      (threshold at the 33rd percentile — big launches behave
//      differently: their evening surge is stronger than the full-pool
//      line says);
//  (2) a Huber (k = 1.345, the 95%-efficiency value) robust centre:
//      downweights the pool's oddballs (a launch carried by a handful
//      of huge channels, or an extreme ratio) instead of letting them
//      drag the line;
//  (3) the 1-regressor v6 centre: same two shapes as (1)+(2) but on
//      ln(v1) alone — it has no pre-midnight regressor, so where the
//      2-regressor members extrapolate on a strange ln(pre/v1) it votes
//      for the plain launch-size shape. The mean of the three members'
//      quantiles does NOT win (practice worse); the median does — the
//      mechanism is robustness, not just a third vote.
//  (4) a ridge-interaction centre: the 2-regressor OLS member refitted
//      with a third regressor, the centred interaction
//      (ln v1 − mean)(ln(pre/v1) − mean), under ridge (lambda = 0.5 on
//      standardised regressors). The interaction alone hits practice
//      0.140 in plain OLS but is leave-one-out poison (0.269); ridge
//      tames it (0.146/0.250 standalone), and as a median member it is
//      the middle vote in ~31% of cells. Its blend is the same 50/50
//      full/big shape as (1), its Huber leg is the standard 2-var Huber,
//      and the two legs combine 0.25/0.75 exactly as v8 did.
//  (5) a quantile-regression median centre: the tau = 0.5 fit (least
//      absolute deviations) of the same two regressors, solved by
//      iteratively reweighted least squares — the L1 centre, which
//      ignores the pool's outliers entirely instead of downweighting
//      them, and which improves leave-one-out at every blend weight
//      (but costs practice as a blend member; as a median member it
//      wins on both). Its slopes are NOT smoothed across hours: the
//      exact-LP version and this IRLS version give identical practice
//      and leave-one-out scores (verified in Python before porting).
// The interaction's target value uses the CLAMPED log hour-1 (unlike the
// Python experiment, which used the raw print): on every measured case
// the clamp never binds, so the numbers are identical, and a GTA-scale
// hour 1 can no longer blow the interaction term up by extrapolation.
// All per-hour regression slopes of the 2-regressor members are smoothed
// [1,2,1] across hours (centred form, so the level never moves): a single
// hour's fit on 19 launches is noisy, and neighbouring hours share the
// same shape. The v6 member keeps raw slopes (as v6 did).
// If the case carries no usable pre-midnight print (no history, or a pool
// member with no pre-day hour), the second regressor is dropped and the
// centres fall back to the one-regressor form — same forecast as v6.
// Spread: the per-hour residual SD (floor 0.10 log), smoothed over a flat
// 7-hour window (per-hour SDs from 19 launches are noisy), scaled 0.93
// (the normal shape overstates the tails: narrower intervals win on both
// practice and leave-one-out), then adjusted for heteroskedasticity — the
// absolute residual grows as log v1 falls (slope ≈ −0.08, pooled over all
// hours). The adjustment is ASYMMETRIC: surges go up, so the upper tail
// widens 1.75x the pooled slope for a quiet hour 1 while the lower tail
// only 0.5x. q_tau(h) = p1 * exp(m_h + z_tau * sigma_h), sigma_h
// upper/lower by the sign of z_tau; the two centres' quantiles are
// averaged arithmetically (identical to a geometric mean at 4 decimals).
//
// Pool: tier "comparable", complete 24 h curve, v1 >= 5% of the launch's
// own peak (rolling-midnight unlocks; drops simultaneous-global unlocks
// whose hour 1 is still pre-unlock and whose ratios explode 100x+).
// Fallbacks, loosening while the pool is < 5: drop the rolling cut, then
// take every complete curve, then the house persistence baseline. Never
// crashes, never emits an invalid case.
//
// Local validation (practice launches' real SullyGnome release days):
// loss 0.146 vs house baseline 0.387; leave-one-out over 19 rolling
// launches 0.246 vs 0.403.
//
// Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out
// to $MAKEFUN_OUTPUT_DIR (else /output): {"hours":[{"hour":2,"q":[q05,
// q25,q50,q75,q95]}, ...]} — 23 rows, hours 2..24. Deterministic.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const IN = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]; // scored quantile levels
const Z = [-1.645, -0.674, 0, 0.674, 1.645]; // standard normal quantiles
const MIN_POOL = 5; // minimum pool size before a fallback tier kicks in
const ROLL_MIN = 0.05; // rolling-midnight launch: v1 >= this fraction of its own peak
const SIG_FLOOR = 0.1; // floor on the per-hour log spread
const SIG_MULT = 0.93; // normal shape overstates the tails (measured)
const HET_UP = 1.75; // asymmetric het: upper tail takes 1.75x pooled slope
const HET_DN = 0.5; // lower tail takes 0.5x (surges go up, not down)
const HUBER_K = 1.345; // Huber IRLS tuning constant (95%-efficiency value)
const BIG_W = 0.5; // weight of the big-launch centre in the blend
const BIG_Q = 0.33; // big pool: v1 at/above this quantile of the pool
const RIDGE_LAM = 0.5; // ridge penalty on standardised regressors (member 4)
const LAD_EPS = 1e-4; // IRLS weight floor for the median (L1) centre
const LAD_ITERS = 100; // IRLS iterations cap for the median centre
const LAD_TOL = 1e-10; // IRLS convergence tolerance (max coefficient move)

interface Row { from: string; viewers: number }
interface Comp { day_start: string; unlock?: string; hourly: Row[] }
interface HistPt { at: string; viewers: number }
interface Case { id: string; levels: number[]; day_start?: string; known: { hour: number; viewers: number }[]; history?: HistPt[]; forecast: { hour: number }[]; comparables?: Comp[] }
interface Curve { v1: number; peak: number; byHour: number[]; preLast: number }

const r2 = (x: number) => Math.round(x * 100) / 100;

// Tier from the unlock note ("Tier: comparable", ...); unmarked = comparable.
const tierOf = (c: Comp) =>
  /Tier:\s*([^.\n]+)/.exec(c.unlock ?? "")?.[1].trim().toLowerCase() ?? "comparable";

// One comparable's release-day curve (viewers, hours 1..24), else null
// when incomplete (a missing hour, or a non-positive hour 1). preLast is
// the viewers at the last recorded hour before the release day starts
// (0 when the comparable has no pre-day hour).
function dayCurve(c: Comp): Curve | null {
  const start = Date.parse(c.day_start);
  if (!Number.isFinite(start)) return null;
  const byHour = new Array(24).fill(0);
  let seen = 0;
  let preAt = -Infinity, preLast = 0;
  for (const r of c.hourly ?? []) {
    const t = Date.parse(r.from);
    if (!Number.isFinite(t) || typeof r.viewers !== "number" || !Number.isFinite(r.viewers)) continue;
    if (t < start) {
      if (t > preAt) { preAt = t; preLast = r.viewers; }
      continue;
    }
    const h = Math.floor((t - start) / 3_600_000) + 1;
    if (!(h >= 1 && h <= 24)) continue;
    if (byHour[h - 1] === 0) seen++;
    byHour[h - 1] = r.viewers;
  }
  if (seen !== 24 || byHour[0] <= 0) return null;
  return { v1: byHour[0], peak: Math.max(...byHour), byHour, preLast };
}

// OLS of y on x -> [intercept, slope, meanX, meanY]; predict centred,
// m = my + b*(x - mx), so shrinking a slope never shifts the level.
function ols(xs: number[], ys: number[]): [number, number, number, number] {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) * (xs[i] - mx); sxy += (xs[i] - mx) * (ys[i] - my); }
  const b = sxx > 0 ? sxy / sxx : 0;
  return [my - b * mx, b, mx, my];
}

// Two-regressor OLS y ~ 1 + x1 + x2 -> [intercept, b1, b2, meanX1, meanX2, meanY].
// The second regressor is ln(pre/v1): small and only weakly collinear with
// ln(v1), so the 2x2 system stays well conditioned; a degenerate system
// falls back to the intercept-only fit instead of blowing up.
function ols2(xs1: number[], xs2: number[], ys: number[]): [number, number, number, number, number, number] {
  const n = xs1.length;
  const m1 = xs1.reduce((a, b) => a + b, 0) / n, m2 = xs2.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let s11 = 0, s22 = 0, s12 = 0, s1y = 0, s2y = 0;
  for (let i = 0; i < n; i++) {
    const d1 = xs1[i] - m1, d2 = xs2[i] - m2, dy = ys[i] - my;
    s11 += d1 * d1; s22 += d2 * d2; s12 += d1 * d2; s1y += d1 * dy; s2y += d2 * dy;
  }
  const det = s11 * s22 - s12 * s12;
  if (Math.abs(det) < 1e-12) return [my, 0, 0, m1, m2, my];
  const b1 = (s1y * s22 - s2y * s12) / det, b2 = (s2y * s11 - s1y * s12) / det;
  return [my - b1 * m1 - b2 * m2, b1, b2, m1, m2, my];
}

const median = (vs: number[]): number => {
  const s = [...vs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

const pstdev = (vs: number[]): number => {
  const m = vs.reduce((a, b) => a + b, 0) / vs.length;
  return Math.sqrt(vs.reduce((s, v) => s + (v - m) * (v - m), 0) / vs.length);
};

// Weighted OLS -> [intercept, slope]; centred form, same fit as ols.
function wols(xs: number[], ys: number[], w: number[]): [number, number] {
  const sw = w.reduce((a, b) => a + b, 0);
  const mx = xs.reduce((s, x, i) => s + w[i] * x, 0) / sw;
  const my = ys.reduce((s, y, i) => s + w[i] * y, 0) / sw;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < xs.length; i++) { sxx += w[i] * (xs[i] - mx) * (xs[i] - mx); sxy += w[i] * (xs[i] - mx) * (ys[i] - my); }
  const b = sxx > 0 ? sxy / sxx : 0;
  return [my - b * mx, b];
}

// Weighted two-regressor OLS -> [intercept, b1, b2].
function wols2(xs1: number[], xs2: number[], ys: number[], w: number[]): [number, number, number] {
  const sw = w.reduce((a, b) => a + b, 0);
  const m1 = xs1.reduce((s, x, i) => s + w[i] * x, 0) / sw;
  const m2 = xs2.reduce((s, x, i) => s + w[i] * x, 0) / sw;
  const my = ys.reduce((s, y, i) => s + w[i] * y, 0) / sw;
  let s11 = 0, s22 = 0, s12 = 0, s1y = 0, s2y = 0;
  for (let i = 0; i < xs1.length; i++) {
    const d1 = xs1[i] - m1, d2 = xs2[i] - m2, dy = ys[i] - my;
    s11 += w[i] * d1 * d1; s22 += w[i] * d2 * d2; s12 += w[i] * d1 * d2;
    s1y += w[i] * d1 * dy; s2y += w[i] * d2 * dy;
  }
  const det = s11 * s22 - s12 * s12;
  if (Math.abs(det) < 1e-12) return [my, 0, 0];
  const b1 = (s1y * s22 - s2y * s12) / det, b2 = (s2y * s11 - s1y * s12) / det;
  return [my - b1 * m1 - b2 * m2, b1, b2];
}

// Huber IRLS slope (k = HUBER_K, scale 1.4826*MAD, pstdev fallback),
// starting from OLS, at most 20 iterations, tol 1e-8. Deterministic.
function huberSlope(xs: number[], ys: number[]): number {
  let [a, b] = ols(xs, ys);
  for (let it = 0; it < 20; it++) {
    const r = ys.map((y, i) => y - (a + b * xs[i]));
    const med = median(r);
    let s = 1.4826 * median(r.map((v) => Math.abs(v - med)));
    if (s === 0) s = pstdev(r);
    if (s === 0) s = 1;
    const w = r.map((v) => {
      const u = v / (HUBER_K * s);
      return Math.abs(u) <= 1 ? 1 : 1 / Math.abs(u);
    });
    const [na, nb] = wols(xs, ys, w);
    const d = Math.max(Math.abs(na - a), Math.abs(nb - b));
    a = na; b = nb;
    if (d < 1e-8) break;
  }
  return b;
}

// Huber IRLS slopes for the two-regressor fit -> [b1, b2]; the centre
// reuses the OLS means, as in huberSlope above.
function huberSlopes2(xs1: number[], xs2: number[], ys: number[]): [number, number] {
  let [, b1, b2] = ols2(xs1, xs2, ys);
  let a = ols2(xs1, xs2, ys)[0];
  for (let it = 0; it < 20; it++) {
    const r = ys.map((y, i) => y - (a + b1 * xs1[i] + b2 * xs2[i]));
    const med = median(r);
    let s = 1.4826 * median(r.map((v) => Math.abs(v - med)));
    if (s === 0) s = pstdev(r);
    if (s === 0) s = 1;
    const w = r.map((v) => {
      const u = v / (HUBER_K * s);
      return Math.abs(u) <= 1 ? 1 : 1 / Math.abs(u);
    });
    const [na, nb1, nb2] = wols2(xs1, xs2, ys, w);
    const d = Math.max(Math.abs(na - a), Math.abs(nb1 - b1), Math.abs(nb2 - b2));
    a = na; b1 = nb1; b2 = nb2;
    if (d < 1e-8) break;
  }
  return [b1, b2];
}

// Solve a 3x3 system by Gaussian elimination with partial pivoting.
// Returns [0,0,0] only if a pivot is (numerically) zero.
function solve3(A: number[][], b: number[]): [number, number, number] {
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return [0, 0, 0];
    if (p !== c) { const t = M[c]; M[c] = M[p]; M[p] = t; }
    for (let r = c + 1; r < 3; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = [0, 0, 0];
  for (let r = 2; r >= 0; r--) {
    let s = M[r][3];
    for (let k = r + 1; k < 3; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return [x[0], x[1], x[2]];
}

// Ridge on centred + standardised regressors (3 columns) -> slopes in
// the original units, plus the y mean. The penalty is lam on the
// standardised scale, so lam ~ O(1) is meaningful whatever n is.
function ridge3(cols: number[][], ys: number[]): [number, number, number, number] {
  const n = ys.length;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  const means = cols.map((c) => c.reduce((a, b) => a + b, 0) / n);
  const sds = cols.map((c, j) => {
    const sd = Math.sqrt(c.reduce((s, v) => s + (v - means[j]) * (v - means[j]), 0) / n);
    return sd > 0 ? sd : 1;
  });
  const A: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const rhs = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const z = cols.map((c, j) => (c[i] - means[j]) / sds[j]);
    const dy = ys[i] - my;
    for (let a = 0; a < 3; a++) {
      rhs[a] += z[a] * dy;
      for (let b = 0; b < 3; b++) A[a][b] += z[a] * z[b];
    }
  }
  for (let a = 0; a < 3; a++) A[a][a] += RIDGE_LAM;
  const bs = solve3(A, rhs);
  return [bs[0] / sds[0], bs[1] / sds[1], bs[2] / sds[2], my];
}

// Least-absolute-deviations slopes for the two-regressor fit -> [b1, b2];
// the centre reuses the OLS means, as in huberSlopes2 above. IRLS with
// weights 1/max(|r|, LAD_EPS), starting from OLS, at most LAD_ITERS
// iterations, tol LAD_TOL. Deterministic.
function ladSlopes2(xs1: number[], xs2: number[], ys: number[]): [number, number] {
  let [, b1, b2, m1, m2, my] = ols2(xs1, xs2, ys);
  let a = my - b1 * m1 - b2 * m2;
  for (let it = 0; it < LAD_ITERS; it++) {
    const w = ys.map((y, i) => 1 / Math.max(Math.abs(y - (a + b1 * xs1[i] + b2 * xs2[i])), LAD_EPS));
    const [na, nb1, nb2] = wols2(xs1, xs2, ys, w);
    const d = Math.max(Math.abs(na - a), Math.abs(nb1 - b1), Math.abs(nb2 - b2));
    a = na; b1 = nb1; b2 = nb2;
    if (d < LAD_TOL) break;
  }
  return [b1, b2];
}

// One-regressor LAD slope (no-pre fallback path), same IRLS skeleton.
function ladSlope(xs: number[], ys: number[]): number {
  let [a, b] = ols(xs, ys);
  for (let it = 0; it < LAD_ITERS; it++) {
    const w = ys.map((y, i) => 1 / Math.max(Math.abs(y - (a + b * xs[i])), LAD_EPS));
    const [na, nb] = wols(xs, ys, w);
    const d = Math.max(Math.abs(na - a), Math.abs(nb - b));
    a = na; b = nb;
    if (d < LAD_TOL) break;
  }
  return b;
}

// Last recorded history point strictly before the release day starts.
function preLastOf(c: Case): number {
  const start = Date.parse(c.day_start ?? "");
  if (!Number.isFinite(start)) return 0;
  let preAt = -Infinity, pre = 0;
  for (const p of c.history ?? []) {
    const t = Date.parse(p.at);
    if (!Number.isFinite(t) || t >= start || typeof p.viewers !== "number" || !Number.isFinite(p.viewers)) continue;
    if (t > preAt) { preAt = t; pre = p.viewers; }
  }
  return pre;
}

function forecastCase(c: Case): { hour: number; q: number[] }[] {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${c.id} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const curves = (c.comparables ?? [])
    .map((comp) => ({ curve: dayCurve(comp), tier: tierOf(comp) }))
    .filter((e): e is { curve: Curve; tier: string } => e.curve !== null);
  // Pool, tightest first, loosening only while it stays too small.
  let pool = curves.filter((e) => e.tier === "comparable" && e.curve.v1 / e.curve.peak >= ROLL_MIN);
  if (pool.length < MIN_POOL) pool = curves.filter((e) => e.tier === "comparable");
  if (pool.length < MIN_POOL) pool = curves;
  // House persistence baseline, used only when no curve can be fitted at all.
  if (pool.length < 2)
    return c.forecast.map(({ hour: h }) => ({ hour: h, q: Z.map((z) => r2(p1 * Math.exp(z * 0.15 * Math.sqrt(h - 1)))) }));
  const xs = pool.map((e) => Math.log(e.curve.v1));
  const lo = Math.min(...xs), hi = Math.max(...xs);
  const logP1 = Math.min(Math.max(Math.log(p1), lo), hi);
  // Second regressor: the target's own pre-midnight state, and the pool's.
  const preTgt = preLastOf(c);
  const x2 = pool.map((e) => Math.log(Math.max(1, e.curve.preLast) / e.curve.v1));
  const usePre = preTgt > 0 && pool.every((e) => e.curve.preLast > 0);
  const t2 = usePre ? Math.log(preTgt / p1) : 0;
  // Third regressor (member 4): the centred interaction of the two
  // regressors; the target's value uses the clamped log hour-1, so a
  // GTA-scale hour 1 cannot blow it up by extrapolation (on every
  // measured case the clamp never binds, so this matches the Python
  // experiment exactly).
  const meanX1 = xs.reduce((a, b) => a + b, 0) / xs.length;
  const meanX2 = x2.reduce((a, b) => a + b, 0) / x2.length;
  const x3 = xs.map((x, i) => (x - meanX1) * (x2[i] - meanX2));
  const t3 = (logP1 - meanX1) * (t2 - meanX2);
  // Big-launch pool: v1 at/above the BIG_Q quantile of the pool.
  const sortedXs = [...xs].sort((a, b) => a - b);
  const thr = sortedXs[Math.floor(BIG_Q * (xs.length - 1))];
  const bigIdx = xs.map((x, i) => (x >= thr ? i : -1)).filter((i) => i >= 0);
  // v6 member's big pool: top half by v1 (quantile 0.5, as v6 did).
  const thr6 = sortedXs[Math.floor(0.5 * (xs.length - 1))];
  const bigIdx6 = xs.map((x, i) => (x >= thr6 ? i : -1)).filter((i) => i >= 0);
  // First pass: per-hour fits — 1-var OLS for the spread, and centre
  // coefficients: two-regressor when the pre-midnight print is usable,
  // one-regressor (b2 = 0) otherwise, so one centre formula serves both.
  const fits = c.forecast.map(({ hour: h }) => {
    const ys = pool.map((e) => Math.log(e.curve.byHour[h - 1] / e.curve.v1));
    const [a, b, mx, my] = ols(xs, ys);
    const resid = ys.map((y, i) => y - (a + b * xs[i]));
    const rm = resid.reduce((s, v) => s + v, 0) / resid.length;
    const sigma = Math.max(SIG_FLOOR, Math.sqrt(resid.reduce((s, v) => s + (v - rm) * (v - rm), 0) / resid.length));
    let b1 = b, b2 = 0, m1 = mx, m2 = 0, my2 = my;
    // v6 member (1-var, raw slopes): keep the 1-var fit before the
    // 2-regressor override below, plus its top-half big-pool centre.
    const hbRaw = huberSlope(xs, ys);
    let hb1 = hbRaw, hb2 = 0;
    let big6: [number, number, number] | null = null;
    if (bigIdx6.length >= 2) {
      const [, d1, dmx, dmy] = ols(bigIdx6.map((j) => xs[j]), bigIdx6.map((j) => ys[j]));
      big6 = [d1, dmx, dmy];
    }
    let big: [number, number, number, number, number] | null = null;
    if (usePre) {
      const [, c1, c2, n1, n2, ny] = ols2(xs, x2, ys);
      b1 = c1; b2 = c2; m1 = n1; m2 = n2; my2 = ny;
      [hb1, hb2] = huberSlopes2(xs, x2, ys);
    }
    if (bigIdx.length >= 2) {
      if (usePre) {
        const [, d1, d2, e1, e2, ey] = ols2(
          bigIdx.map((j) => xs[j]), bigIdx.map((j) => x2[j]), bigIdx.map((j) => ys[j]));
        big = [d1, d2, e1, e2, ey];
      } else {
        const [, d1, dmx, dmy] = ols(bigIdx.map((j) => xs[j]), bigIdx.map((j) => ys[j]));
        big = [d1, 0, dmx, 0, dmy];
      }
    }
    // Member 4 (ridge interaction): 3-var ridge on (x1, x2, x3), full
    // pool and big pool; skipped when the pre regressor is unusable
    // (the member then votes with member 1 instead). Member 5 (LAD
    // median): raw slopes, never smoothed.
    let r1 = 0, r2 = 0, r3 = 0, rm1 = meanX1, rm2 = meanX2, rmy = my2;
    let rm3 = x3.reduce((a, b) => a + b, 0) / x3.length;
    let rbig: [number, number, number, number, number, number, number] | null = null;
    if (usePre) {
      [r1, r2, r3, rmy] = ridge3([xs, x2, x3], ys);
      if (bigIdx.length >= 2) {
        const [rb1, rb2, rb3, rbmy] = ridge3(
          [bigIdx.map((j) => xs[j]), bigIdx.map((j) => x2[j]), bigIdx.map((j) => x3[j])],
          bigIdx.map((j) => ys[j]));
        rbig = [rb1, rb2, rb3,
          bigIdx.reduce((s, j) => s + xs[j], 0) / bigIdx.length,
          bigIdx.reduce((s, j) => s + x2[j], 0) / bigIdx.length,
          bigIdx.reduce((s, j) => s + x3[j], 0) / bigIdx.length, rbmy];
      }
    }
    let lad1: number, lad2: number;
    if (usePre) [lad1, lad2] = ladSlopes2(xs, x2, ys);
    else { lad1 = ladSlope(xs, ys); lad2 = 0; }
    return { hour: h, b1, b2, m1, m2, my2, hb1, hb2, big, ys, resid, sigma, bRaw: b, mx1: mx, my1: my, hbRaw, big6, r1, r2, r3, rm1, rm2, rm3, rmy, rbig, lad1, lad2 };
  });
  // Slope smoothing [1,2,1] across hours: per-hour coefficients fitted on
  // 19 launches are noisy; the centred form keeps the level untouched.
  const sm121 = (vs: number[]): number[] => vs.map((_, i) => {
    let s = 0, w = 0;
    for (let j = -1; j <= 1; j++) {
      const k = i + j;
      if (k >= 0 && k < vs.length) { const wt = j === 0 ? 2 : 1; s += wt * vs[k]; w += wt; }
    }
    return s / w;
  });
  const sB1 = sm121(fits.map((f) => f.b1)), sB2 = sm121(fits.map((f) => f.b2));
  const sH1 = sm121(fits.map((f) => f.hb1)), sH2 = sm121(fits.map((f) => f.hb2));
  const sG1 = sm121(fits.map((f) => (f.big ? f.big[0] : 0)));
  const sG2 = sm121(fits.map((f) => (f.big ? f.big[1] : 0)));
  const sR1 = sm121(fits.map((f) => f.r1)), sR2 = sm121(fits.map((f) => f.r2)), sR3 = sm121(fits.map((f) => f.r3));
  const sGR1 = sm121(fits.map((f) => (f.rbig ? f.rbig[0] : 0)));
  const sGR2 = sm121(fits.map((f) => (f.rbig ? f.rbig[1] : 0)));
  const sGR3 = sm121(fits.map((f) => (f.rbig ? f.rbig[2] : 0)));
  fits.forEach((f, i) => {
    f.b1 = sB1[i]; f.b2 = sB2[i]; f.hb1 = sH1[i]; f.hb2 = sH2[i];
    if (f.big) f.big = [sG1[i], sG2[i], f.big[2], f.big[3], f.big[4]];
    f.r1 = sR1[i]; f.r2 = sR2[i]; f.r3 = sR3[i];
    if (f.rbig) f.rbig = [sGR1[i], sGR2[i], sGR3[i], f.rbig[3], f.rbig[4], f.rbig[5], f.rbig[6]];
  });
  // Spread, smoothed over a flat 7-hour window (neighbours in hour order).
  const smooth = fits.map((f, i) => {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - 3); j <= Math.min(fits.length - 1, i + 3); j++) { s += fits[j].sigma; n++; }
    return s / n;
  });
  // Pooled slope of |residual| on centred log v1 (heteroskedasticity).
  const mx = fits[0].m1; // mean log v1 (same in the 1-var and 2-var fits)
  let meanAbs = 0, cnt = 0;
  for (const f of fits) for (const r of f.resid) { meanAbs += Math.abs(r); cnt++; }
  meanAbs /= cnt;
  let num = 0, den = 0;
  for (const f of fits) for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx;
    num += dx * (Math.abs(f.resid[i]) - meanAbs); den += dx * dx;
  }
  const het = den > 0 ? num / den : 0;
  // One model's five quantiles from a centre m (same spread for all members).
  const quantiles = (m: number, i: number): number[] => {
    const sig = Math.max(SIG_FLOOR, smooth[i] * SIG_MULT);
    const sigUp = Math.max(SIG_FLOOR, sig + HET_UP * het * (logP1 - mx));
    const sigDn = Math.max(SIG_FLOOR, sig + HET_DN * het * (logP1 - mx));
    const q = Z.map((z) => {
      const s = z > 0 ? sigUp : z < 0 ? sigDn : sig;
      return Math.min(1e9, Math.max(0.01, r2(p1 * Math.exp(m + z * s))));
    });
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1]; // keep quantiles non-decreasing
    return q;
  };
  return fits.map((f, i) => {
    const mOls = f.my2 + f.b1 * (logP1 - f.m1) + f.b2 * (t2 - f.m2);
    // Big-launch centre (falls back to the full-pool centre if too small).
    let mBig = mOls;
    if (f.big) {
      const [bb1, bb2, bm1, bm2, bmy2] = f.big;
      mBig = bmy2 + bb1 * (logP1 - bm1) + bb2 * (t2 - bm2);
    }
    const mBlend = (1 - BIG_W) * mOls + BIG_W * mBig;
    const mHuber = f.my2 + f.hb1 * (logP1 - f.m1) + f.hb2 * (t2 - f.m2);
    const qA = quantiles(mBlend, i);
    const qB = quantiles(mHuber, i);
    // Fourth member: ridge-interaction centre — its blend leg (0.25)
    // against the standard Huber leg (0.75), exactly as v8 weighted
    // them; without a usable pre regressor it votes with member (1).
    let qRidge = qA;
    if (usePre) {
      let mRidge = f.rmy + f.r1 * (logP1 - f.rm1) + f.r2 * (t2 - f.rm2) + f.r3 * (t3 - f.rm3);
      if (f.rbig) {
        const [rb1, rb2, rb3, rbm1, rbm2, rbm3, rbmy] = f.rbig;
        const mRidgeBig = rbmy + rb1 * (logP1 - rbm1) + rb2 * (t2 - rbm2) + rb3 * (t3 - rbm3);
        mRidge = (1 - BIG_W) * mRidge + BIG_W * mRidgeBig;
      }
      const qRA = quantiles(mRidge, i);
      qRidge = qRA.map((v, j) => r2(0.25 * v + 0.75 * qB[j]));
      for (let j = 1; j < qRidge.length; j++) if (qRidge[j] < qRidge[j - 1]) qRidge[j] = qRidge[j - 1];
    }
    // Fifth member: LAD median centre (raw slopes), same spread.
    const mQR = usePre
      ? f.my2 + f.lad1 * (logP1 - f.m1) + f.lad2 * (t2 - f.m2)
      : f.my1 + f.lad1 * (logP1 - f.mx1);
    const qQR = quantiles(mQR, i);
    // Third member: the 1-regressor v6 centre (raw slopes, top-half big
    // pool, 50/50 blend + Huber, same spread) — its own two quantiles
    // averaged, as v6 did.
    const mFull6 = f.my1 + f.bRaw * (logP1 - f.mx1);
    let mBig6 = mFull6;
    if (f.big6) {
      const [bb6, bm6, by6] = f.big6;
      mBig6 = by6 + bb6 * (logP1 - bm6);
    }
    const mBlend6 = 0.5 * mFull6 + 0.5 * mBig6;
    const mHuber6 = f.my1 + f.hbRaw * (logP1 - f.mx1);
    const q6A = quantiles(mBlend6, i);
    const q6B = quantiles(mHuber6, i);
    const q6 = q6A.map((v, j) => r2((v + q6B[j]) / 2));
    // Median of the five members' quantiles, per level: the middle
    // member wins each (hour, tau), so one member's bad extrapolation
    // can never drag a quantile with it.
    const q = qA.map((v, j) => [v, qB[j], q6[j], qRidge[j], qQR[j]].sort((x, y) => x - y)[2]);
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { hour: f.hour, q };
  });
}

mkdirSync(OUT, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
for (const file of readdirSync(IN).filter((f) => f.endsWith(".in")).sort()) {
  const id = file.slice(0, -3);
  const c = JSON.parse(readFileSync(join(IN, file), "utf8")) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j]))
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  writeFileSync(join(OUT, `${id}.out`), `${JSON.stringify({ hours: forecastCase({ ...c, id }) })}\n`);
}
