# Cohort Rarity Forecaster v2

Forecast how many GTA VI players earn each trophy in week one — entry #2 for
make.fun's [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty.

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xeraaGH/gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xeraaGH/gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)

## The model

**Median — cohort-delay.** Players arrive over time (N only grows); a player who
bought at time `s` earns a trophy by time `t` with probability `R·(1−exp(−(t−s)/τ))`.
Earners are the cohorts summed with that delay; rate = earners / players tracked.
`R` and `τ` are fitted per trophy (`τ` by grid search over 12 values on full-history
fit error, `R` by least squares) and the median saturates at `R`. Future arrivals
between the last read and the target are projected at the recent arrival rate, so the
forecast denominator is the projected players tracked. Story-chain medians (chapters
1–4, then the final mission) are forced non-increasing by isotonic regression, and the
platinum median is capped at the smallest median.

**Spread — calibrated.** Holdout error (fit on the first 70% of reads, scored on the
last 30%) + horizon term + biggest refresh-jump term + disagreement term (target
forecasts across τ values that fit almost as well), combined by root-sum-of-squares,
floored at 0.35 logit. A flat history gets a wide band (≥ 1.0 logit) and its median
shifted down 0.3 logit — a flat page has not refreshed yet, and the refresh drops rates.

**Entry #2 change.** Entry #1's practice grade over-covered (coverage_90 0.9636 vs
nominal 0.90) — its band was systematically too wide. This entry multiplies the
calibrated spread by 0.85, tightening it toward the nominal 90%. It is submitted
only because it beats entry #1 on the same synthetic rich-history test.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` (Node 22, standard library only, no network) reads every `*.in` in the
input directory and writes one `<id>.out` per case:
`{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}`. Runs are deterministic and
byte-identical.

## License

MIT — see [LICENSE](LICENSE).
