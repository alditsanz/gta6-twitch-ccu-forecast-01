# GTA VI Twitch CCU Forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/alditsanz/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/alditsanz/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast Grand Theft Auto VI's Twitch viewers hour by hour over its release
day, 2026-11-19 (UTC), from everything recorded up to the end of the first
hour — five quantiles (5%, 25%, 50%, 75%, 95%) for each of hours 2 to 24,
scored by pinball loss on the log scale.

## Program

`main.ts` (Node 22, standard library only, no network). It reads every
`<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`):

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Method

Per-hour log-ratio regression on the case's own comparables — the full write-up is
the header comment in `main.ts`. In short: for each past big launch with a
complete release-day curve, hour-1 viewers `v1` predict the rest of the day
(`log(peak/v1)` correlates −0.82 with `log(v1)`), so per hour `h = 2..24` the
program fits `ln(v_h/v1) = a_h + b_h * ln(v1) + c_h * ln(pre/v1)` and predicts
the target at its own hour-1 print, log-clamped to the pool's range. The
second regressor is the launch's own state just before midnight — `pre` is
the viewers in the last recorded hour before the release day — which says
where the launch sits on its early-unlock surge-decay at 00:00 UTC. The
forecast is the MEDIAN of five centres' quantiles (same spread for all):
two 2-regressor centres (a big-launch blend — the full-pool centre averaged
50/50 with a centre fitted only on the top three-quarters of the pool by
`v1`, threshold at the 33rd percentile, because big launches' evening
surge runs stronger than the full-pool line says — and a Huber, k = 1.345,
robust centre that downweights the pool's oddballs instead of letting them
drag the line) plus a third 1-regressor centre on `ln(v1)` alone (raw
slopes, top-half big pool, its own 50/50 blend + Huber) plus two diverse
centres: a ridge-interaction centre (the 2-regressor OLS member refitted
with the centred interaction `(ln v1 − mean)(ln(pre/v1) − mean)` under
ridge, λ = 0.5 — the interaction alone hits practice 0.140 in plain OLS
but is leave-one-out poison, ridge tames it, and as a median member it is
the middle vote in ~31% of cells) and a quantile-regression median centre
(least absolute deviations on the same two regressors, by iteratively
reweighted least squares — the L1 centre, which ignores outliers entirely
instead of downweighting them). The median — the middle member at each
(hour, quantile) — is what makes the extra members work: a mean of the
five does not improve anything, but the median is robust to whichever
member extrapolates worst on a given hour. All
per-hour regression slopes of the 2-regressor members are smoothed [1,2,1]
across hours (centred form, so the level never moves): a single hour's fit
on 19 launches is noisy; the median centre's slopes are not smoothed.
If a case carries no usable pre-midnight print the second regressor drops
out and the centres fall back to the one-regressor form (same smoothed
slopes, same weights). The spread is
the per-hour residual SD (floor 0.10 log), smoothed over a flat 7-hour
window, scaled 0.93 (the normal shape overstates the tails), and adjusted
for heteroskedasticity — small hour-1 launches surge unpredictably, huge
ones run a flat predictable day, so the spread widens for a quiet hour 1
and narrows for a huge one (pooled slope of |residual| on centred log v1
≈ −0.08) — asymmetrically: the upper tail takes 1.75x the slope, the lower
tail 0.5x, because surges go up, not down. Pool: tier `comparable`,
complete 24 h curve, `v1` ≥ 5% of the launch's own peak. Fallbacks loosen
the pool, then the house persistence baseline — never crashes, never emits
an invalid case.

## Local validation

Against the four practice launches' real release days (SullyGnome hourly
averages, the public source the kit cites):

| case | this entry | house baseline |
|---|---|---|
| Marvel's Wolverine | 0.165 | 0.744 |
| Red Dead Redemption 2 | 0.139 | 0.287 |
| God of War Ragnarök | 0.102 | 0.215 |
| Marvel's Spider-Man 2 | 0.179 | 0.303 |
| **mean pinball loss** | **0.146** | **0.387** |

90% coverage 0.946, typical miss 22.4%. Leave-one-out over the 19 rolling
tier-comparable launches: 0.246 versus 0.403 for the baseline (90% coverage
0.865). Two runs on
the practice cases are byte-identical; one run takes ~0.8 s for all four
cases (cap is 8 s per case).

## License

MIT — see [LICENSE](LICENSE).
