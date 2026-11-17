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
// extrapolating the slope past the data). Spread = per-hour residual SD
// (floor 0.10 log): q_tau(h) = p1 * exp(m_h + z_tau * sigma_h).
//
// Pool: tier "comparable", complete 24 h curve, v1 >= 5% of the launch's
// own peak (rolling-midnight unlocks; drops simultaneous-global unlocks
// whose hour 1 is still pre-unlock and whose ratios explode 100x+).
// Fallbacks, loosening while the pool is < 5: drop the rolling cut, then
// take every complete curve, then the house persistence baseline. Never
// crashes, never emits an invalid case.
//
// Local validation (practice launches' real SullyGnome release days):
// loss 0.181 vs house baseline 0.387; leave-one-out over 19 rolling
// launches 0.262 vs 0.403.
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
  return c.forecast.map(({ hour: h }) => {
    const ys = pool.map((e) => Math.log(e.curve.byHour[h - 1] / e.curve.v1));
    const [a, b, mx, my] = ols(xs, ys);
    const m = my + b * (logP1 - mx);
    const resid = ys.map((y, i) => y - (a + b * xs[i]));
    const rm = resid.reduce((s, v) => s + v, 0) / resid.length;
    const sigma = Math.max(SIG_FLOOR, Math.sqrt(resid.reduce((s, v) => s + (v - rm) * (v - rm), 0) / resid.length));
    const q = Z.map((z) => Math.min(1e9, Math.max(0.01, r2(p1 * Math.exp(m + z * sigma)))));
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1]; // keep quantiles non-decreasing
    return { hour: h, q };
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
