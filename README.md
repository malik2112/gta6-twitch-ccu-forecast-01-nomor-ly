# GTA VI launch-day Twitch CCV forecast — one pooled day-curve

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/malik2112/gta6-twitch-ccu-forecast-01-nomor-ly/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/malik2112/gta6-twitch-ccu-forecast-01-nomor-ly/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`: forecast how
many people watch Grand Theft Auto VI live on Twitch in each hour of its
release day (2026-11-19 UTC), as five quantiles (5/25/50/75/95%) per hour,
given only everything recorded up to the end of hour 1.

## Program

`main.py` (Python 3.13, standard library only). Run it the way the grader
does:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory and writes
`<id>.out` (`{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}`,
hours 2–24) to the output directory. Deterministic, no network, well
under the 8 s per-case limit (~0.07 s a case locally).

## Method

The case file ships `comparables`: past big launches' hourly Twitch
category counts. Instead of fitting a separate model per hour, this
entry fits ONE pooled regression over every (launch, hour) point at
once, for the log ratio of an hour's viewers to hour 1's:

```
log(viewers_h / viewers_1)
    = b0 + b1·u + b2·u² + b3·v + b4·v·u + b5·v·u²
```

where `u = log(h − 1)` is (log) time since hour 1 and
`v = log(viewers_1) − 12` is how big hour 1 already was. One curve,
shared across the whole day:

- The `u, u²` terms give the day's shape — a rise into the evening
  (US prime time) and a fall overnight — as a smooth parabola in log
  time, the same shape every launch follows up to scale.
- The `v` terms let the shape tilt with launch size: a huge
  simultaneous-midnight launch (Elden Ring, Cyberpunk) is already at
  its peak in hour 1 and stays flat, while a rolling-midnight launch
  whose hour 1 is still small (Ghost of Yōtei, Stellar Blade) climbs
  steeply.

The fit is ordinary least squares solved by Gaussian elimination on
the 6×6 normal equations (standard library only), over comparables
whose hour 1 was already a real launch hour (hour-1 viewers ≥ 20,000,
relaxed to ≥ 5,000 then ≥ 100 if too few qualify). The target game's
own pre-release category history (the "Grand Theft Auto VI"
comparable, whose day is the 2026-08-27 premiere, not a launch) is
excluded. Using one pooled fit (6 numbers for the whole day) instead
of 23 separate hour fits keeps the variance low where any single hour
has few training launches.

Spread: the residuals' standard deviation per hour, smoothed across
hours with a centred 3-hour moving average, floored at 0.20 and
capped at 0.80 log units; all five quantiles are plain normal
quantiles, median + z·σ.

## Validation

Leave-one-out over the kit's comparables (each launch held out in
turn, forecast from the rest), scored with the contest's own pinball
loss on log scale:

| set | this model | house persistence baseline |
| --- | --- | --- |
| all 37 usable comparables (× 4 practice cases) | **0.634** | 0.930 |

Local checks on the kit's four practice cases: outputs valid (23 rows,
five non-decreasing quantiles each), byte-identical across two runs,
~0.3 s total for all four cases.

## Fallbacks

If the fit cannot be solved, the median falls back to the plain median
log-ratio curve of the training set with the same spread rule; with no
usable comparables at all the program falls back to the house
persistence baseline (flat at hour 1, spread 0.15·√(h−1)), so the
output is always valid.

## License

MIT — see `LICENSE`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=pwfn5nwa&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
