#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads every case file (*.in) in the input directory and writes one forecast
file (<id>.out) per case to the output directory. Each forecast gives, for
every trophy of the case, five quantiles (5/25/50/75/95%) of the trophy's
Exophase earn rate at the case's target time.

Model in one paragraph: an earn rate is a ratio, earners / players tracked,
and the two sides move for different reasons. The tracked pool grows as more
players sync a profile; earners of a trophy grow as the pool's players reach
that trophy. The history Exophase leaves behind is a staircase: long flat
stretches (the page is only refreshed every few hours) separated by jumps.
So the history is first compressed into distinct states, then the pool size
and each trophy's earner count are extrapolated separately on a log-time
axis, and the forecast rate is the ratio of the two extrapolations. The
spread around the median combines a random-walk floor with a refresh-risk
term that stays large while the observed history is still flat.

Standard library only, no network, fully deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Quantile levels the case asks for, and the matching standard normal z.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_OF_LEVEL = (-1.6448536269514722, -0.6744897501960817, 0.0,
              0.6744897501960817, 1.6448536269514722)

# Rates are printed to 2 decimals; 0 and 100 have no logit, so the scoring
# clamps to this range and the model works inside it too.
RATE_FLOOR = 0.01
RATE_CEIL = 99.99

# Spread (logit scale) ingredients.
WALK_SIGMA_PER_SQRT_DAY = 0.5     # the house random-walk scale, kept as a floor
FLAT_HISTORY_SIGMA = 1.15         # refresh risk while few states were seen
SIGMA_MIN = 0.35
SIGMA_MAX = 2.2
SHRINK_K = 3.0                    # earner-slope shrinkage strength, in states
DILUTION_LOGIT_SHIFT = -0.4       # early-pool enrichment prior, logit units
DEFAULT_RESID_SD = 0.6            # spread stand-in when a fit has no residuals


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def hours_between(start, end):
    return (end - start).total_seconds() / 3600.0


def logit(rate_percent):
    p = min(max(rate_percent, RATE_FLOOR), RATE_CEIL) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    exp_v = math.exp(value)
    return 100.0 * exp_v / (1.0 + exp_v)


def clamp_rate(rate_percent):
    return min(max(rate_percent, 0.0), 100.0)


def round4(value):
    return round(value, 4)


def compress_states(history):
    """Collapse consecutive identical reads into distinct Exophase states.

    Exophase refreshes its counts on its own schedule, so a run of reads with
    the same players_tracked and the same rates is one state of the page, not
    many observations. Returns a list of (moment, players, rates) keeping the
    first moment each state was seen.
    """
    states = []
    for read in history:
        rates = read.get("rates") or []
        players = read.get("players_tracked")
        if states and states[-1][1] == players and states[-1][2] == rates:
            continue
        states.append((parse_time(read["at"]), players, rates))
    return states


def fit_line(xs, ys):
    """Ordinary least squares y ~ a + b*x; returns (a, b, residual_sd)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0 or n == 1:
        return mean_y, 0.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    return intercept, slope, sd


def forecast_pool(states, u_target, u_of):
    """Extrapolate players_tracked to the target on the log-time axis."""
    last_players = None
    for _, players, _ in reversed(states):
        if players:
            last_players = players
            break
    if last_players is None:
        return 1.0
    points = [(u_of(moment), math.log(players))
              for moment, players, _ in states if players]
    if len(points) < 2:
        return float(last_players)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    intercept, slope, _ = fit_line(xs, ys)
    fitted = math.exp(intercept + slope * u_target)
    # Recent-half growth rate, as a cross-check on the whole-history fit.
    half = points[len(points) // 2:]
    if len(half) >= 2 and half[-1][0] > half[0][0]:
        recent_slope = (half[-1][1] - half[0][1]) / (half[-1][0] - half[0][0])
        recent = math.exp(ys[-1] + recent_slope * (u_target - xs[-1]))
        fitted = math.sqrt(fitted * recent)
    # A tracked pool does not shrink, and even a launch pool does not grow
    # without bound inside one forecast window.
    return min(max(fitted, float(last_players)), float(last_players) * 60.0)


def tier_group(trophy):
    """Pooling group for slope shrinkage: platinum rides with gold, since
    both are finish-the-game-scale achievements with very few earners."""
    tier = trophy.get("tier", "bronze")
    if tier in ("platinum", "gold"):
        return "platinum+gold"
    return tier


def earner_points(idx, states, u_of):
    """(u, implied earners, rate) for each state that lists this trophy."""
    points = []
    for moment, players, rates in states:
        if not players or idx >= len(rates) or rates[idx] is None:
            continue
        points.append((u_of(moment), rates[idx] * players / 100.0, rates[idx]))
    return points


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(case.get("horizon_minutes", 0), 0) / 1440.0

    states = compress_states(history)
    if states:
        origin = states[0][0]
    else:
        origin = checkpoint

    def u_of(moment):
        return math.log1p(max(hours_between(origin, moment), 0.0))

    u_target = u_of(target)
    pool_target = forecast_pool(states, u_target, u_of)

    # The last rate each trophy was seen at (walking back over the reads).
    last_rate = [None] * len(trophies)
    for read in reversed(history):
        rates = read.get("rates") or []
        for idx, value in enumerate(rates):
            if idx < len(last_rate) and last_rate[idx] is None and value is not None:
                last_rate[idx] = value
        if all(v is not None for v in last_rate):
            break

    n_states = len(states)
    walk = WALK_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)
    # Refresh risk: with at most two distinct states the page may not have
    # refreshed at all inside the observed window, so a jump of the kind the
    # launch-night refresh produced sits inside the forecast window.
    if n_states <= 2:
        refresh = FLAT_HISTORY_SIGMA
    else:
        refresh = FLAT_HISTORY_SIGMA * math.exp(-(n_states - 2) / 6.0)

    # Earner trajectories per trophy, and their log-time slopes shrunk
    # toward the pooled slope of each trophy's tier group: a slope read off
    # two or three states is mostly noise, a tier's pooled slope is not.
    trajectories = [earner_points(idx, states, u_of)
                    for idx in range(len(trophies))]
    raw_slopes = []
    for points in trajectories:
        if len(points) >= 2:
            _, slope, _ = fit_line(
                [p[0] for p in points],
                [math.log(max(p[1], 0.5)) for p in points])
            raw_slopes.append(slope)
        else:
            raw_slopes.append(None)
    group_slope = {}
    for group in ("platinum+gold", "silver", "bronze"):
        members = [raw_slopes[i] for i, t in enumerate(trophies)
                   if tier_group(t) == group and raw_slopes[i] is not None]
        group_slope[group] = sum(members) / len(members) if members else 0.0
    all_slopes = [s for s in raw_slopes if s is not None]
    global_slope = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0

    # Early-pool dilution prior: while no refresh has been observed and the
    # game is days old, the tracked pool is still the self-selected first
    # wave of players, who earn trophies faster than the players who will
    # join the pool by the target. Rates are biased up; shift the median
    # down a fixed step on the logit scale.
    dilution = 0.0
    if n_states <= 2 and states:
        release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
        if hours_between(release, checkpoint) <= 72.0:
            dilution = DILUTION_LOGIT_SHIFT

    rows = []
    for idx, trophy in enumerate(trophies):
        points = trajectories[idx]
        slope = raw_slopes[idx]
        if slope is None:
            slope = group_slope.get(tier_group(trophy), global_slope)
        else:
            pooled = group_slope.get(tier_group(trophy), global_slope)
            weight = len(points) / (len(points) + SHRINK_K)
            slope = weight * slope + (1.0 - weight) * pooled
        median, resid_sd = forecast_trophy_rate(
            points, slope, u_target, pool_target, last_rate[idx])
        last_u = points[-1][0] if points else u_target
        fit_sigma = resid_sd * math.sqrt(1.0 + max(u_target - last_u, 0.0) / 2.0)
        sigma = min(max(math.sqrt(walk * walk + refresh * refresh +
                                  fit_sigma * fit_sigma), SIGMA_MIN), SIGMA_MAX)
        centre = logit(median) + dilution
        quantiles = []
        for z in Z_OF_LEVEL:
            quantiles.append(round4(clamp_rate(inv_logit(centre + z * sigma))))
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def forecast_trophy_rate(points, slope, u_target, pool_target, last_rate):
    """Median earn-rate forecast for one trophy, plus the fit's residual SD.

    points holds (u, implied earners, rate) per observed state; slope is the
    trophy's shrunk log-earner growth slope on the u axis.
    """
    if not points:
        # Never observed: fall back to the last read's rate if there is one,
        # else a small prior; the wide shared spread carries the doubt.
        if last_rate is not None:
            return min(max(last_rate, RATE_FLOOR), RATE_CEIL), DEFAULT_RESID_SD
        return 0.5, DEFAULT_RESID_SD
    if all(rate >= 100.0 for _, _, rate in points):
        # Earned by every tracked player at every read (the "start the game"
        # trophy): it stays a hair under 100.
        return RATE_CEIL, 0.05
    last_earners = points[-1][1]
    if len(points) >= 2:
        xs = [p[0] for p in points]
        ys = [math.log(max(p[1], 0.5)) for p in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        intercept = mean_y - slope * mean_x
        earners = math.exp(intercept + slope * u_target)
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        resid_sd = math.sqrt(sum(r * r for r in resid) / max(len(xs) - 2, 1))
    else:
        # A single state: earners grow with the pool, rate stays put.
        resid_sd = DEFAULT_RESID_SD
        earners = last_earners
        if last_rate is not None:
            return min(max(last_rate, RATE_FLOOR), RATE_CEIL), resid_sd
    earners = max(earners, last_earners * 0.9)
    rate = 100.0 * earners / max(pool_target, 1.0)
    return min(max(rate, RATE_FLOOR), RATE_CEIL), resid_sd


def main(argv):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
