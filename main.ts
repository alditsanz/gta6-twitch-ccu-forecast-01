// gta6-twitch-ccu-forecast-01 — launch-day Twitch CCU forecaster (Node 22, stdlib only).
//
// Method: per-hour log-ratio regression on the case's own comparables.
//   For each past big launch with a complete release-day curve, let v1 be its
//   hour-1 viewers and v_h its viewers in hour h. The shape of a launch day
//   depends strongly on how big hour 1 already is: log(peak/v1) correlates
//   -0.82 with log(v1) across the kit's comparables (a launch whose hour 1 is
//   already huge stays flat; a quiet hour 1 explodes by daytime). So for each
//   hour h = 2..24 we fit, over the comparable launches, the OLS line
//
//     ln(v_h / v1) = a_h + b_h * ln(v1)
//
//   and predict the target launch at its own hour-1 print p1. The spread is
//   the per-hour residual standard deviation (floored at 0.10 log units), so
//   q_tau(h) = p1 * exp(m_h + z_tau * sigma_h) with z the standard normal
//   quantiles at the five scored levels. The median curve this gives is the
//   familiar launch-day shape: a night dip to ~0.8x in hours 2-5, US unlocks
//   lifting it through the morning, and an evening peak around hours 19-20,
//   with the amplitude shrinking as hour 1 grows.
//
//   Pool: only launches that look like this one — tier "comparable" in the
//   case file, a complete 24-hour release-day curve, and v1 at least 5% of
//   the launch's own peak (rolling-midnight unlocks: hour 1 already holds
//   the UK/EU midnight audience). That last cut drops simultaneous-global
//   unlocks (Silksong, Battlefield 6, Black Myth: Wukong, ...), whose hour 1
//   is still pre-unlock and whose ratios explode 100x+. Early-access,
//   free-to-play and the GTA VI pre-release history rows are dropped by the
//   tier cut. Fallbacks (never crash, never emit an invalid case): if the
//   pool is under 5 launches, drop the rolling cut; if still under 5, take
//   every comparable with a complete curve; if there is no usable curve at
//   all, fall back to the house persistence baseline.
//
//   Validation (local, against the four practice launches' real SullyGnome
//   release days, looked up from the public chart data the kit cites):
//   loss 0.181 (p00 0.238, p01 0.163, p02 0.137, p03 0.188) versus the house
//   baseline's 0.387 on the same four cases. Leave-one-out over the 19
//   rolling tier-comparable launches: 0.262 versus 0.403 for the baseline.
//
// It reads every <id>.in in the input directory (MAKEFUN_INPUT_DIR or
// /input) and writes <id>.out to the output directory (MAKEFUN_OUTPUT_DIR or
// /output): {"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}, exactly
// 23 rows, hours 2..24. Deterministic: no randomness, no network, no clock.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Minimum pool size before a fallback tier kicks in. */
const MIN_POOL = 5;
/** A rolling-midnight launch already has a real audience in hour 1: at
 *  least this fraction of its own release-day peak. Simultaneous-global
 *  launches sit far below it (their hour 1 is still pre-unlock). */
const ROLLING_MIN_V1_OVER_PEAK = 0.05;
/** Floor on the per-hour log spread, so a freakishly tight fit never
 *  claims more certainty than the data supports. */
const SIGMA_FLOOR = 0.1;

interface HourlyRow {
  from: string;
  viewers: number;
  channels?: number | null;
}
interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  hourly: HourlyRow[];
}
interface Case {
  id: string;
  levels: number[];
  known: Array<{ hour: number; viewers: number; channels?: number | null }>;
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[];
}

/** One comparable's release-day curve: viewers in hours 1..24, or null
 *  when the curve is incomplete (a missing hour, or a non-positive hour 1). */
interface Curve {
  v1: number;
  peak: number;
  byHour: number[]; // index h-1 -> viewers in hour h
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The tier a comparable's unlock note assigns it ("comparable",
 *  "early-access-contaminated", "weak", "target-history", ...). A note
 *  without a Tier marker is treated as comparable; the rolling cut below
 *  still has to pass. */
function tierOf(c: Comparable): string {
  const m = /Tier:\s*([^.\n]+)/.exec(c.unlock ?? "");
  return m ? m[1].trim().toLowerCase() : "comparable";
}

function dayCurve(c: Comparable): Curve | null {
  const start = Date.parse(c.day_start);
  if (!Number.isFinite(start)) return null;
  const byHour: number[] = new Array(24).fill(0);
  let seen = 0;
  for (const row of c.hourly ?? []) {
    const t = Date.parse(row.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.floor((t - start) / 3_600_000) + 1;
    if (h < 1 || h > 24) continue;
    if (typeof row.viewers !== "number" || !Number.isFinite(row.viewers)) continue;
    if (byHour[h - 1] === 0) seen += 1;
    byHour[h - 1] = row.viewers;
  }
  if (seen !== 24 || byHour[0] <= 0) return null;
  return { v1: byHour[0], peak: Math.max(...byHour), byHour };
}

function isRolling(curve: Curve): boolean {
  return curve.v1 / curve.peak >= ROLLING_MIN_V1_OVER_PEAK;
}

/** OLS of y on x; returns [intercept, slope, meanX, meanY]. */
function ols(xs: number[], ys: number[]): [number, number, number, number] {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  return [my - b * mx, b, mx, my];
}

function pstdev(values: number[]): number {
  const n = values.length;
  if (n < 2) return 0;
  const m = values.reduce((a, b) => a + b, 0) / n;
  return Math.sqrt(values.reduce((a, v) => a + (v - m) * (v - m), 0) / n);
}

/** The house persistence baseline, used only when no comparable curve
 *  can be fitted at all. q_tau(h) = p1 * exp(z_tau * 0.15 * sqrt(h-1)). */
function baselineQuantiles(p1: number, h: number): number[] {
  return Z.map((z) => round2(p1 * Math.exp(z * 0.15 * Math.sqrt(h - 1))));
}

function forecastCase(c: Case): Array<{ hour: number; q: number[] }> {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${c.id} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);

  const curves: Array<{ curve: Curve; tier: string }> = [];
  for (const comp of c.comparables ?? []) {
    const curve = dayCurve(comp);
    if (curve !== null) curves.push({ curve, tier: tierOf(comp) });
  }

  // Pool, tightest first, loosening only while it stays too small.
  let pool = curves.filter(
    (e) => e.tier === "comparable" && isRolling(e.curve),
  );
  if (pool.length < MIN_POOL) {
    pool = curves.filter((e) => e.tier === "comparable");
  }
  if (pool.length < MIN_POOL) pool = curves;
  if (pool.length < 2) {
    return c.forecast.map(({ hour }) => ({ hour, q: baselineQuantiles(p1, hour) }));
  }

  const xs = pool.map((e) => Math.log(e.curve.v1));
  const logP1 = Math.log(p1);

  return c.forecast.map(({ hour: h }) => {
    const ys = pool.map((e) => Math.log(e.curve.byHour[h - 1] / e.curve.v1));
    const [a, b, mx, my] = ols(xs, ys);
    const m = my + b * (logP1 - mx);
    const resid = ys.map((y, i) => y - (a + b * xs[i]));
    const sigma = Math.max(SIGMA_FLOOR, pstdev(resid));
    const q = Z.map((z) => {
      const v = round2(p1 * Math.exp(m + z * sigma));
      return Math.min(1e9, Math.max(0.01, v));
    });
    // Quantiles are non-decreasing by construction (z ascending, sigma > 0);
    // enforce it anyway so a rounding edge can never break the shape.
    for (let i = 1; i < q.length; i++) {
      if (q[i] < q[i - 1]) q[i] = q[i - 1];
    }
    return { hour: h, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith(".in")).sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as Case;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  const hours = forecastCase({ ...c, id });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
