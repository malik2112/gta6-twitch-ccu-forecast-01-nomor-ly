"""GTA VI launch-day Twitch CCV forecaster (gta6-twitch-ccu-forecast-01)
-- nomor-ly entry: one pooled parametric day-curve.

Method
------
The case file ships `comparables`: past big launches' hourly Twitch
category counts. Instead of fitting a separate model per hour, this
program fits ONE pooled regression over every (launch, hour) point at
once, for the log ratio of an hour's viewers to hour 1's:

    log(viewers_h / viewers_1)
        = b0 + b1*u + b2*u^2 + b3*v + b4*v*u + b5*v*u^2

where u = log(h - 1) is the (log) time since hour 1 and
v = log(viewers_1) - 12 is how big hour 1 already was. One curve,
shared across the whole day:

  * the u, u^2 terms give the day's shape -- a rise into the evening
    (US prime time) and a fall overnight -- as a smooth parabola in
    log time, the same shape every launch follows up to scale;
  * the v terms let the shape tilt with launch size: a huge
    simultaneous-midnight launch (Elden Ring, Cyberpunk) is already
    at its peak in hour 1 and stays flat (negative tilt), while a
    rolling-midnight launch whose hour 1 is still small (Ghost of
    Yotei, Stellar Blade) climbs steeply (positive tilt).

The fit is ordinary least squares solved by Gaussian elimination on
the 6x6 normal equations (standard library only), over comparables
whose hour 1 was already a real launch hour (hour-1 viewers >=
10,000, relaxed to >= 5,000 then >= 100 if too few qualify). The
target game's own pre-release category history ("Grand Theft Auto
VI", day_start 2026-08-27, a premiere not a launch) is excluded.

Spread: the residuals' standard deviation per hour, smoothed across
hours with a centred 3-hour moving average, floored at 0.20 and
capped at 0.80 log units; all five quantiles are the plain normal
quantiles median + z_tau * sigma_h. Using one pooled fit (6 numbers
for the whole day) instead of 23 separate hour fits keeps the
variance low where any single hour has few training launches.

Fallbacks: if the fit cannot be solved (singular system, too few
training rows) the median falls back to the plain median log-ratio
curve of the training set with the same spread rule; with no usable
comparables at all, the program falls back to the house persistence
baseline (flat at hour 1, spread 0.15 * sqrt(h - 1)). Deterministic,
no network, no clock reads, standard library only.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

LOG_P1_CENTER = 12.0
SIGMA_FLOOR = 0.20
SIGMA_CAP = 0.80
BASELINE_SIGMA = 0.15
MIN_TRAIN = 8


def round2(x):
    """Round to 2 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def _release_day_rows(comp):
    """(hour-1 viewers, {hour: viewers}) for a comparable's release day."""
    hourly = comp.get("hourly") or []
    day = (comp.get("day_start") or "")[:10]
    if len(day) != 10:
        return None
    by_hour = {}
    for row in hourly:
        start = row.get("from") or ""
        if not start.startswith(day):
            continue
        try:
            hour = int(start[11:13]) + 1  # 00:00 UTC -> hour 1
        except (ValueError, IndexError):
            continue
        viewers = row.get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            by_hour[hour] = float(viewers)
    p1 = by_hour.get(1)
    if p1 is None or p1 < 100:
        return None
    answers = {h: v for h, v in by_hour.items() if 2 <= h <= 24}
    if len(answers) < 20:
        return None
    return p1, answers


def _training_set(case):
    rows = []
    for comp in case.get("comparables") or []:
        if "Grand Theft Auto VI" in (comp.get("name") or ""):
            continue
        parsed = _release_day_rows(comp)
        if parsed is not None:
            rows.append(parsed)
    for threshold in (10000.0, 5000.0, 100.0):
        kept = [r for r in rows if r[0] >= threshold]
        if len(kept) >= MIN_TRAIN:
            return kept
    return rows


def _features(hour, p1):
    """[1, u, u^2, v, v*u, v*u^2] with u = log(h-1), v = log p1 - 12."""
    u = math.log(hour - 1)
    v = math.log(p1) - LOG_P1_CENTER
    return [1.0, u, u * u, v, v * u, v * u * u]


def _solve(matrix, rhs):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, n):
            f = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= f * aug[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = aug[r][n] - sum(aug[r][c] * out[c] for c in range(r + 1, n))
        out[r] = s / aug[r][r]
    return out


def _fit_pooled(train):
    """Pooled OLS coefficients + per-hour residual std, or None."""
    points = []
    for comp_p1, answers in train:
        for hour, viewers in answers.items():
            points.append((_features(hour, comp_p1),
                           math.log(viewers / comp_p1), hour))
    if len(points) < 24:
        return None
    k = 6
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for x, y, _ in points:
        for i in range(k):
            atb[i] += x[i] * y
            for j in range(k):
                ata[i][j] += x[i] * x[j]
    # Tiny ridge for numerical stability.
    for i in range(k):
        ata[i][i] += 1e-6
    beta = _solve(ata, atb)
    if beta is None:
        return None
    resid_by_hour = {h: [] for h in range(2, 25)}
    for x, y, hour in points:
        pred = sum(b * xi for b, xi in zip(beta, x))
        resid_by_hour[hour].append(y - pred)
    sigma = {}
    for hour in range(2, 25):
        r = resid_by_hour[hour]
        if len(r) >= 4:
            mean = sum(r) / len(r)
            var = sum((v - mean) ** 2 for v in r) / len(r)
            sigma[hour] = math.sqrt(var)
        else:
            sigma[hour] = None
    # Fill gaps from neighbours, then smooth with a centred 3-hour mean.
    filled = {}
    for hour in range(2, 25):
        if sigma[hour] is not None:
            filled[hour] = sigma[hour]
        else:
            near = [sigma[h] for h in range(2, 25)
                    if sigma[h] is not None]
            filled[hour] = sum(near) / len(near) if near else 0.35
    smooth = {}
    for hour in range(2, 25):
        vals = [filled[h] for h in (hour - 1, hour, hour + 1)
                if 2 <= h <= 24]
        smooth[hour] = max(SIGMA_FLOOR,
                           min(SIGMA_CAP, sum(vals) / len(vals)))
    return beta, smooth


def _median_curve(train):
    """Fallback: plain median log-ratio per hour, smoothed."""
    by_hour = {h: [] for h in range(2, 25)}
    for comp_p1, answers in train:
        for hour, viewers in answers.items():
            by_hour[hour].append(math.log(viewers / comp_p1))
    med = {}
    for hour in range(2, 25):
        vals = sorted(by_hour[hour])
        if not vals:
            med[hour] = None
        elif len(vals) % 2:
            med[hour] = vals[len(vals) // 2]
        else:
            med[hour] = (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2.0
    smooth = {}
    for hour in range(2, 25):
        vals = [med[h] for h in (hour - 1, hour, hour + 1)
                if 2 <= h <= 24 and med[h] is not None]
        smooth[hour] = sum(vals) / len(vals) if vals else None
    return smooth


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError("%s asks for levels %s, not %s"
                         % (name, case.get("levels"), LEVELS))
    first = next((k for k in case.get("known", []) if k.get("hour") == 1),
                 None)
    if first is None:
        raise ValueError("%s has no hour-1 print" % name)
    p1 = max(1.0, float(first["viewers"]))
    train = _training_set(case)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]

    fitted = _fit_pooled(train) if train else None
    medians = _median_curve(train) if train else {}
    hours = []
    for row in rows:
        hour = row["hour"]
        if fitted is not None:
            beta, sigma = fitted
            x = _features(hour, p1)
            median = sum(b * xi for b, xi in zip(beta, x))
            spread = sigma[hour]
        elif medians.get(hour) is not None:
            median = medians[hour]
            spread = 0.35
        else:
            median = None
            spread = None
        if median is None:
            offsets = [z * BASELINE_SIGMA * math.sqrt(hour - 1) for z in Z]
        else:
            offsets = [median + z * spread for z in Z]
        quantiles = []
        for off in offsets:
            value = p1 * math.exp(max(-4.0, min(5.5, off)))
            value = max(1.0, min(1_000_000_000.0, value))
            quantiles.append(round2(value))
        for i in range(1, 5):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        hours.append({"hour": hour, "q": quantiles})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
