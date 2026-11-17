// gta6-twitch-ccu-forecast-01 — launch-day Twitch CCU forecaster (Node 22+, stdlib only, no network).
//
// Method: per-hour log-ratio OLS on the case's own comparables. For a past
// launch with a complete release-day curve, hour-1 viewers v1 predict the
// rest: log(peak/v1) correlates -0.82 with log(v1) (a huge hour 1 stays
// flat; a quiet one explodes by daytime). So per hour h = 2..24 fit
//   ln(v_h/v1) = a_h + b_h * ln(v1)
// over the pool and predict the target at its own hour-1 print p1, with
// log p1 clamped to the pool's observed range (a GTA-scale hour 1 above
// every past launch is forecast like the biggest launches, not by
// extrapolating the slope past the data). Spread: the per-hour residual SD
// (floor 0.10 log), smoothed over a flat 7-hour window (per-hour SDs from
// 19 launches are noisy), scaled 0.93 (the normal shape overstates the
// tails: narrower intervals win on both practice and leave-one-out), then
// adjusted for heteroskedasticity — the absolute residual grows as log v1
// falls (slope ≈ −0.08, pooled over all hours). The adjustment is
// ASYMMETRIC: surges go up, so the upper tail widens 1.75x the pooled
// slope for a quiet hour 1 while the lower tail only 0.5x. Centre: the
// per-hour slopes are smoothed [1,2,1] across hours (the day curve is
// smooth; raw per-hour slopes are noisy). q_tau(h) = p1 * exp(m_h +
// z_tau * sigma_h), sigma_h upper/lower by the sign of z_tau.
//
// Pool: tier "comparable", complete 24 h curve, v1 >= 5% of the launch's
// own peak (rolling-midnight unlocks; drops simultaneous-global unlocks
// whose hour 1 is still pre-unlock and whose ratios explode 100x+).
// Fallbacks, loosening while the pool is < 5: drop the rolling cut, then
// take every complete curve, then the house persistence baseline. Never
// crashes, never emits an invalid case.
//
// Local validation (practice launches' real SullyGnome release days):
// loss 0.169 vs house baseline 0.387; leave-one-out over 19 rolling
// launches 0.255 vs 0.403.
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

interface Row { from: string; viewers: number }
interface Comp { day_start: string; unlock?: string; hourly: Row[] }
interface Case { id: string; levels: number[]; known: { hour: number; viewers: number }[]; forecast: { hour: number }[]; comparables?: Comp[] }
interface Curve { v1: number; peak: number; byHour: number[] }

const r2 = (x: number) => Math.round(x * 100) / 100;

// Tier from the unlock note ("Tier: comparable", ...); unmarked = comparable.
const tierOf = (c: Comp) =>
  /Tier:\s*([^.\n]+)/.exec(c.unlock ?? "")?.[1].trim().toLowerCase() ?? "comparable";

// One comparable's release-day curve (viewers, hours 1..24), else null
// when incomplete (a missing hour, or a non-positive hour 1).
function dayCurve(c: Comp): Curve | null {
  const start = Date.parse(c.day_start);
  if (!Number.isFinite(start)) return null;
  const byHour = new Array(24).fill(0);
  let seen = 0;
  for (const r of c.hourly ?? []) {
    const h = Math.floor((Date.parse(r.from) - start) / 3_600_000) + 1;
    if (!(h >= 1 && h <= 24) || typeof r.viewers !== "number" || !Number.isFinite(r.viewers)) continue;
    if (byHour[h - 1] === 0) seen++;
    byHour[h - 1] = r.viewers;
  }
  if (seen !== 24 || byHour[0] <= 0) return null;
  return { v1: byHour[0], peak: Math.max(...byHour), byHour };
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
  // First pass: per-hour OLS, residuals, raw spread (centre follows after
  // slope smoothing, so m is computed below, not here).
  const fits = c.forecast.map(({ hour: h }) => {
    const ys = pool.map((e) => Math.log(e.curve.byHour[h - 1] / e.curve.v1));
    const [a, b, mx, my] = ols(xs, ys);
    const resid = ys.map((y, i) => y - (a + b * xs[i]));
    const rm = resid.reduce((s, v) => s + v, 0) / resid.length;
    const sigma = Math.max(SIG_FLOOR, Math.sqrt(resid.reduce((s, v) => s + (v - rm) * (v - rm), 0) / resid.length));
    return { hour: h, b, mx, my, resid, sigma };
  });
  // Centre slopes smoothed [1,2,1] across hours (edges renormalised).
  const ms = fits.map((f, i) => {
    let s = 0, w = 0;
    for (let j = -1; j <= 1; j++) {
      const k = i + j;
      if (k < 0 || k >= fits.length) continue;
      const wt = j === 0 ? 2 : 1;
      s += wt * fits[k].b; w += wt;
    }
    return f.my + (s / w) * (logP1 - f.mx);
  });
  // Spread, smoothed over a flat 7-hour window (neighbours in hour order).
  const smooth = fits.map((f, i) => {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - 3); j <= Math.min(fits.length - 1, i + 3); j++) { s += fits[j].sigma; n++; }
    return s / n;
  });
  // Pooled slope of |residual| on centred log v1 (heteroskedasticity).
  const mx = fits[0].mx;
  let meanAbs = 0, cnt = 0;
  for (const f of fits) for (const r of f.resid) { meanAbs += Math.abs(r); cnt++; }
  meanAbs /= cnt;
  let num = 0, den = 0;
  for (const f of fits) for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx;
    num += dx * (Math.abs(f.resid[i]) - meanAbs); den += dx * dx;
  }
  const het = den > 0 ? num / den : 0;
  return fits.map((f, i) => {
    const sig = Math.max(SIG_FLOOR, smooth[i] * SIG_MULT);
    const sigUp = Math.max(SIG_FLOOR, sig + HET_UP * het * (logP1 - mx));
    const sigDn = Math.max(SIG_FLOOR, sig + HET_DN * het * (logP1 - mx));
    const q = Z.map((z) => {
      const s = z > 0 ? sigUp : z < 0 ? sigDn : sig;
      return Math.min(1e9, Math.max(0.01, r2(p1 * Math.exp(ms[i] + z * s))));
    });
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1]; // keep quantiles non-decreasing
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
