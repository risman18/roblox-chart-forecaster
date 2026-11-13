#!/usr/bin/env python3
"""Roblox Up-and-Coming forecaster — make.fun roblox-up-and-coming-forecast-01.

Reads hourly snapshots of Roblox's Up-and-Coming chart from /input
(history.ndjson + task.json) and writes one forecast to /output/forecast.json:
for each of the top 10 places, a probability for each game named being at
that place over the scored window, plus the guessed top 10 in order.

Model: every charted game gets a latent rank path — its last rank plus a
damped drift (from its rank velocity and player-count growth) plus noise
that widens with the horizon. A seeded Monte Carlo rolls those paths over
the window's hours; a place's probabilities are the games' time-averaged
occupancy of that place, shrunk toward a persistence prior and a diffuse
top-30 prior so no plausible game is ever given a hopeless floor.

Deterministic: fixed seed, no network, standard library only.
"""
import json
import math
import os
import random
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261113
SIMS = 500
MAX_RANK = 140.0
# Tunable constants (calibrated by backtest on the practice archive).
DRIFT_DAMP = 0.55          # how much of the measured rank velocity persists
GROWTH_TO_DRIFT = 20.0     # places per unit of log-playing growth per hour
BASE_VOL = 2.0             # rank noise, places per sqrt(hour), near the top
VOL_RANK_SLOPE = 0.055     # extra noise per place of current rank
SHRINK_PERSIST = 0.30      # weight of the persistence prior
SHRINK_WIDE = 0.10         # weight of the diffuse top-30 prior
LEAVE_UNNAMED = 0.06       # probability mass left unnamed at each place


def load_history(input_dir):
    snaps = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("ok", True):
                continue
            rows = d.get("up_and_coming") or []
            if not rows:
                continue
            t = _hours(d["at"])
            chart = {}
            for r in rows:
                chart[int(r["universe_id"])] = (int(r["rank"]), float(r.get("playing") or 0))
            created = {}
            det = d.get("details")
            if isinstance(det, dict):
                for gid, info in det.items():
                    try:
                        created[int(gid)] = info.get("created")
                    except (TypeError, ValueError):
                        pass
            snaps.append((t, chart, created))
    snaps.sort(key=lambda s: s[0])
    return snaps


def _hours(iso):
    # "2026-11-07T20:53:19.000Z" -> hours since epoch (float)
    import datetime
    dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return dt.timestamp() / 3600.0


def build_features(snaps):
    """Per game: observations [(t, rank, playing)], plus created time if known."""
    games = {}
    created = {}
    for t, chart, crt in snaps:
        for gid, (rank, playing) in chart.items():
            games.setdefault(gid, []).append((t, rank, playing))
        for gid, c in crt.items():
            if c:
                created[gid] = _hours(c)
    return games, created


def game_params(obs, t_end):
    """(r0, drift places/hour, last_playing) for one game at the history's end."""
    t_last, r_last, p_last = obs[-1]
    # Absence since the last snapshot pushes the effective rank down.
    gap = max(0.0, t_end - t_last)
    r0 = r_last + 1.5 * gap
    # Weighted regression of rank on time over the recent observations.
    recent = obs[-8:]
    if len(recent) >= 3:
        t0 = recent[0][0]
        ws, sx, sy, sxx, sxy = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, (t, r, _p) in enumerate(recent):
            w = 0.75 ** (len(recent) - 1 - i)
            x = t - t0
            ws += w; sx += w * x; sy += w * r; sxx += w * x * x; sxy += w * x * r
        den = ws * sxx - sx * sx
        slope = (ws * sxy - sx * sy) / den if den > 1e-9 else 0.0
    else:
        slope = 0.0
    # Player-count growth (log per hour) over the same span.
    growth = 0.0
    if len(recent) >= 2:
        (ta, _ra, pa), (tb, _rb, pb) = recent[0], recent[-1]
        if tb > ta and pa > 0 and pb > 0:
            growth = (math.log(pb) - math.log(pa)) / (tb - ta)
    drift = DRIFT_DAMP * slope - GROWTH_TO_DRIFT * growth
    drift = max(-1.2, min(1.2, drift))
    return r0, drift, p_last


def forecast(snaps, task):
    places = int(task.get("places", 10))
    window = task.get("window") or {}
    slots = int(window.get("slots", 72))
    games, created = build_features(snaps)
    t_end = snaps[-1][0]
    window_end = t_end + slots
    cands = []
    for gid, obs in games.items():
        r0, drift, _p = game_params(obs, t_end)
        if r0 > 70 and drift > -0.25:
            continue
        # 28-day rule: a game published more than 28 days before the window
        # ends cannot stay on the chart; ramp its rank away near the end.
        age_out_h = None
        if gid in created:
            age_out_h = created[gid] + 28 * 24 - t_end
        cands.append((gid, r0, drift, age_out_h))
    if not cands:
        return {"probabilities": [], "top10": []}
    rng = random.Random(SEED)
    occ = [[0.0] * len(cands) for _ in range(places)]
    top_votes = [0.0] * len(cands)
    steps = list(range(1, slots + 1))
    for _sim in range(SIMS):
        # One noise draw per game per step would be slow; draw a per-game
        # Brownian path incrementally instead.
        pos = [c[1] for c in cands]
        noises = [0.0] * len(cands)
        vols = [BASE_VOL + VOL_RANK_SLOPE * c[1] for c in cands]
        for h in steps:
            order = []
            for i, (gid, r0, drift, age_out_h) in enumerate(cands):
                noises[i] += rng.gauss(0.0, vols[i])
                r = r0 + drift * h + noises[i]
                if age_out_h is not None and h > age_out_h:
                    r += (h - age_out_h) * 2.0
                order.append((r, i))
            order.sort()
            for place in range(min(places, len(order))):
                occ[place][order[place][1]] += 1.0
            for place, (_r, i) in enumerate(order[:places]):
                top_votes[i] += (places - place)
    total = float(SIMS * slots)
    # Persistence prior from the final chart.
    last_chart = snaps[-1][1]
    persist = {}
    for gid, (rank, _p) in last_chart.items():
        persist[gid] = rank
    wide = sorted(cands, key=lambda c: c[1])[:30]
    wide_ids = {c[0] for c in wide}
    probabilities = []
    for place in range(1, places + 1):
        row = {}
        for i, (gid, _r0, _d, _a) in enumerate(cands):
            p = occ[place - 1][i] / total
            pr = 0.0
            rk = persist.get(gid)
            if rk is not None:
                if rk == place:
                    pr = 0.5
                elif abs(rk - place) == 1:
                    pr = 0.2 if place == 1 else 0.15
            wd = 0.5 / max(1, len(wide_ids)) if gid in wide_ids else 0.0
            v = (1.0 - SHRINK_PERSIST - SHRINK_WIDE) * p + SHRINK_PERSIST * pr + SHRINK_WIDE * wd
            if v > 0.0012:
                row[gid] = v
        s = sum(row.values())
        cap = 1.0 - LEAVE_UNNAMED
        if s > cap:
            row = {g: v * cap / s for g, v in row.items()}
        for gid, v in sorted(row.items(), key=lambda kv: -kv[1])[:100]:
            probabilities.append({"universe_id": gid, "place": place, "p": round(v, 6)})
    ranked = sorted(range(len(cands)), key=lambda i: -top_votes[i])
    top10 = [cands[i][0] for i in ranked[:places]]
    return {"probabilities": probabilities, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    snaps = load_history(INPUT_DIR)
    result = forecast(snaps, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    sys.exit(main())
