#!/usr/bin/env python3
"""backtest — calibrate the sunsetglow index against observed glow.

Input: a labels CSV with columns
    date,lat,lon,event,observed[,baseline,source,note]
where date is the local date (YYYY-MM-DD), event is sunset|sunrise and
observed is a 0-4 grade: 0 none, 1 faint (微烧), 2 small (小烧), 3 medium (中烧), 4 large (大烧);
half grades (2.5 = 小到中烧) are allowed. baseline, if present, is another
forecaster's grade on the same scale and is scored the same way for comparison.

For each row the archived GFS forecast (Open-Meteo Historical Forecast API) and
CAMS AOD are fetched (cached in .cache/), the index is recomputed for every
combination of the tunable parameters, and the script reports
  - Spearman rank correlation between index and observed grade,
  - ROC AUC for "observed >= 2" (worth going out for),
then, for the best combination, index thresholds that best separate the grades.
"""
import argparse
import csv
import datetime as dt
import itertools
import sys

import glowmap
import sunsetglow as sg


def read_labels(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"date": dt.date.fromisoformat(r["date"]), "lat": float(r["lat"]), "lon": float(r["lon"]),
                         "event": r["event"].strip(), "observed": float(r["observed"]),
                         "baseline": float(r["baseline"]) if r.get("baseline") not in (None, "") else None,
                         "source": r.get("source", ""), "note": r.get("note", "")})
    return rows


def make_columns(model, memo):
    """Archived-forecast column source; raw data memoised so parameter sweeps don't refetch."""
    def columns(points, t):
        key = (tuple(points), t)
        if key not in memo:
            memo[key] = sg.fetch_raw(points, t - dt.timedelta(hours=2), t + dt.timedelta(hours=2), model,
                                     request=glowmap.cached_request, url=sg.HISTORICAL_URL, cloud_only=True)
        return [sg.raw_to_column(r, t) for r in memo[key]]
    return columns


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def auc(scores, positive):
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fit_thresholds(scores, observed):
    """For each grade k = 1..4, the index cut that best separates observed >= k (balanced accuracy)."""
    cuts = []
    cands = sorted(set(scores))
    for k in range(1, 5):
        pos = [o >= k for o in observed]
        if all(pos) or not any(pos):
            cuts.append(None)
            continue
        best = None
        for i in range(len(cands)):
            c = (cands[i] + cands[i + 1]) / 2 if i + 1 < len(cands) else cands[i] + 1e-6
            tp = sum(s >= c and p for s, p in zip(scores, pos))
            tn = sum(s < c and not p for s, p in zip(scores, pos))
            ba = 0.5 * (tp / sum(pos) + tn / (len(pos) - sum(pos)))
            if best is None or ba > best[0]:
                best = (ba, c)
        cuts.append(best[1])
    # cuts must be non-decreasing
    for i in range(1, len(cuts)):
        if cuts[i] is not None and cuts[i - 1] is not None and cuts[i] < cuts[i - 1]:
            cuts[i] = cuts[i - 1]
    return cuts


def floats(text):
    return [float(v) for v in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description="Calibrate the sunsetglow index against observed glow labels.")
    ap.add_argument("labels", help="labels CSV (date,lat,lon,event,observed[,source,note])")
    ap.add_argument("--model", default="gfs025")
    ap.add_argument("--haze-heights", type=floats, default=[1.0, 1.5, 2.5], help="aerosol scale heights to try, km")
    ap.add_argument("--beta-scales", type=floats, default=[1.0], help="cloud extinction multipliers to try")
    ap.add_argument("--element-kms", type=floats, default=[0.0, 10.0, 30.0, 100.0],
                    help="cloud element sizes to try, km (0 = uniform c*beta, the old model)")
    ap.add_argument("--good", type=float, default=2.0, help="observed grade counted as positive for AUC (default 2)")
    ap.add_argument("--fast", action="store_true", help="coarser time/angle steps (map settings)")
    ap.add_argument("--out", default="backtest_results.csv")
    args = ap.parse_args()

    rows = read_labels(args.labels)
    memo = {}
    columns = make_columns(args.model, memo)
    steps = dict(offsets=glowmap.MAP_OFFSETS, angles=glowmap.MAP_ANGLES) if args.fast else {}
    observed = [r["observed"] for r in rows]
    good = [o >= args.good for o in observed]
    print("%d labelled events, %d with observed >= %g" % (len(rows), sum(good), args.good))
    base = [r["baseline"] for r in rows]
    if all(b is not None for b in base):
        print("baseline forecaster: spearman %+.2f   AUC %.2f" % (spearman(base, observed), auc(base, good)))

    results = []
    for haze, beta, elem in itertools.product(args.haze_heights, args.beta_scales, args.element_kms):
        sg.CLOUD_BETA_SCALE = beta
        sg.CLOUD_ELEMENT_KM = elem
        scores = []
        for r in rows:
            f = sg.forecast(r["lat"], r["lon"], r["event"], r["date"], columns, haze, **steps)
            scores.append(f["index"])
        rho, a = spearman(scores, observed), auc(scores, good)
        results.append((rho, a, haze, beta, elem, scores))
        print("  haze %.1f km  beta x%.1f  element %4.0f km   spearman %+.2f   AUC %.2f"
              % (haze, beta, elem, rho, a), file=sys.stderr)

    # NaN-safe ranking: Spearman first, AUC as tie-break
    def key(res):
        return tuple(-1e9 if v != v else v for v in res[:2])

    rho, a, haze, beta, elem, scores = max(results, key=key)
    print()
    print("best: haze-height %.1f km, cloud beta x%.1f, element %.0f km  (spearman %+.2f, AUC %.2f)"
          % (haze, beta, elem, rho, a))
    cuts = fit_thresholds(scores, observed)
    names = ["微烧", "小烧", "中烧", "大烧"]
    print("index thresholds:", "  ".join("%s>=%s" % (n, "%.2f" % c if c is not None else "?")
                                         for n, c in zip(names, cuts)))
    if len(rows) < 30:
        print("warning: only %d events; treat these numbers as a smoke test, not a calibration" % len(rows))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "lat", "lon", "event", "observed", "baseline", "index", "source", "note"])
        for r, s in zip(rows, scores):
            w.writerow([r["date"], r["lat"], r["lon"], r["event"], r["observed"], r["baseline"], "%.3f" % s,
                        r["source"], r["note"]])
    print("per-event results: %s" % args.out)


if __name__ == "__main__":
    main()
