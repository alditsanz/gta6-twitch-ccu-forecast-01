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

Per-hour log-ratio regression on the case's own comparables. For each past
big launch with a complete release-day curve, hour 1's viewers `v1` predict
the rest of the day: `log(peak/v1)` correlates −0.82 with `log(v1)` across
the kit's comparables — a launch whose hour 1 is already huge stays flat,
a quiet hour 1 explodes by daytime. So for each hour `h = 2..24` the program
fits the OLS line `ln(v_h / v1) = a_h + b_h * ln(v1)` over the comparable
launches and predicts the target at its own hour-1 print `p1` (log-clamped
to the pool's observed range, so a GTA-scale hour 1 above every past launch
is forecast like the biggest launches, not by extrapolating the slope past
the data). The spread is the per-hour residual standard deviation (floored
at 0.10 log units): `q_tau(h) = p1 * exp(m_h + z_tau * sigma_h)`.

The pool is only launches that look like this one: tier `comparable`,
a complete 24-hour release-day curve, and `v1` at least 5% of the launch's
own peak (rolling-midnight unlocks — hour 1 already holds the UK/EU
midnight audience). That cut drops simultaneous-global unlocks, whose hour 1
is still pre-unlock and whose ratios explode 100x+, and the tier cut drops
early-access, free-to-play and the GTA VI pre-release history rows.
Fallbacks loosen the pool, and with no usable curve at all the program falls
back to the house persistence baseline — it never crashes and never emits
an invalid case. Deterministic: no randomness, no network, no clock.

## Local validation

Against the four practice launches' real release days (SullyGnome hourly
averages, the public source the kit cites):

| case | this entry | house baseline |
|---|---|---|
| Marvel's Wolverine | 0.238 | 0.744 |
| Red Dead Redemption 2 | 0.163 | 0.287 |
| God of War Ragnarök | 0.137 | 0.215 |
| Marvel's Spider-Man 2 | 0.188 | 0.303 |
| **mean pinball loss** | **0.181** | **0.387** |

90% coverage 0.924, typical miss 30.1%. Leave-one-out over the 19 rolling
tier-comparable launches: 0.262 versus 0.403 for the baseline. Two runs on
the practice cases are byte-identical; one run takes ~0.45 s for all four
cases (cap is 8 s per case).

## License

MIT — see [LICENSE](LICENSE).
