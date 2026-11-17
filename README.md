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

Per-hour log-ratio OLS on the case's own comparables — the full write-up is
the header comment in `main.ts`. In short: for each past big launch with a
complete release-day curve, hour-1 viewers `v1` predict the rest of the day
(`log(peak/v1)` correlates −0.82 with `log(v1)`), so per hour `h = 2..24` the
program fits `ln(v_h/v1) = a_h + b_h * ln(v1)` and predicts the target at its
own hour-1 print, log-clamped to the pool's range; the spread is the per-hour
residual SD (floor 0.10 log), smoothed over a flat 5-hour window and adjusted
for heteroskedasticity — small hour-1 launches surge unpredictably, huge ones
run a flat predictable day, so the spread widens for a quiet hour 1 and
narrows for a huge one (pooled slope of |residual| on centred log v1 ≈ −0.08).
Pool: tier `comparable`, complete 24 h curve,
`v1` ≥ 5% of the launch's own peak. Fallbacks loosen the pool, then the house
persistence baseline — never crashes, never emits an invalid case.

## Local validation

Against the four practice launches' real release days (SullyGnome hourly
averages, the public source the kit cites):

| case | this entry | house baseline |
|---|---|---|
| Marvel's Wolverine | 0.239 | 0.744 |
| Red Dead Redemption 2 | 0.158 | 0.287 |
| God of War Ragnarök | 0.124 | 0.215 |
| Marvel's Spider-Man 2 | 0.184 | 0.303 |
| **mean pinball loss** | **0.176** | **0.387** |

90% coverage 0.946, typical miss 30.1%. Leave-one-out over the 19 rolling
tier-comparable launches: 0.258 versus 0.403 for the baseline. Two runs on
the practice cases are byte-identical; one run takes ~0.45 s for all four
cases (cap is 8 s per case).

## License

MIT — see [LICENSE](LICENSE).
