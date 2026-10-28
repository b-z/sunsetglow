#!/usr/bin/env python3
"""glowmap — regional sunrise/sunset glow map built on sunsetglow.

1. Fetch one shared model field (cloud levels + AOD) on a regular grid covering
   the map area extended 700 km toward the sun and 300 km away from it.
   Requests are chunked, cached on disk and retried on HTTP 429.
2. For every map point, build its sun-azimuth cross-section by bilinear
   interpolation from that field and run sunsetglow.forecast (in parallel).
3. Write a single self-contained HTML page (Leaflet) with the results inline.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
import urllib.parse

import sunsetglow as sg

MAP_OFFSETS = list(range(-9, 46, 3))
MAP_ANGLES = [1 + i for i in range(179)]
CHUNK = 100  # locations per request
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


# ---------------------------------------------------------------- cached, chunked requests

def cached_request(url, params):
    lats, lons = params["latitude"].split(","), params["longitude"].split(",")
    out = []
    for i in range(0, len(lats), CHUNK):
        p = dict(params, latitude=",".join(lats[i:i + CHUNK]), longitude=",".join(lons[i:i + CHUNK]))
        full = url + "?" + urllib.parse.urlencode(p, safe=",")
        path = os.path.join(CACHE_DIR, hashlib.sha1(full.encode()).hexdigest() + ".json")
        if os.path.exists(path):
            with open(path) as f:
                out += json.load(f)
            continue
        for attempt in range(6):
            try:
                data = sg.http_json(url, p)
                break
            except SystemExit as e:
                if "429" not in str(e) or attempt == 5:
                    raise
                print("  rate limited, waiting 65 s ...", file=sys.stderr)
                time.sleep(65)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        out += data
        print("  fetched %s %d/%d" % (url.split("/")[2], min(i + CHUNK, len(lats)), len(lats)), file=sys.stderr)
    return out


# ---------------------------------------------------------------- shared field

class GridField:
    def __init__(self, lat0, lon0, nlat, nlon, step, raws):
        self.lat0, self.lon0, self.nlat, self.nlon, self.step = lat0, lon0, nlat, nlon, step
        self.raws = raws  # row-major, lat index outer
        self._memo = {}

    def _node(self, i, j, t):
        key = (i, j, t)
        col = self._memo.get(key)
        if col is None:
            raw = self.raws[i * self.nlon + j]
            col = sg.raw_to_column(raw, t)
            # index layers by level so neighbours can be matched level by level
            col = {"aod": col["aod"], "by_z": col["layers"]}
            if len(self._memo) > 20000:
                self._memo.clear()
            self._memo[key] = col
        return col

    def column(self, lat, lon, t):
        fi = min(max((lat - self.lat0) / self.step, 0.0), self.nlat - 1.000001)
        fj = min(max((lon - self.lon0) / self.step, 0.0), self.nlon - 1.000001)
        i, j = int(fi), int(fj)
        di, dj = fi - i, fj - j
        nodes = [(self._node(i, j, t), (1 - di) * (1 - dj)), (self._node(i + 1, j, t), di * (1 - dj)),
                 (self._node(i, j + 1, t), (1 - di) * dj), (self._node(i + 1, j + 1, t), di * dj)]
        n = min(len(c["by_z"]) for c, _ in nodes)
        layers = []
        for k in range(n):
            layers.append(tuple(sum(c["by_z"][k][q] * w for c, w in nodes) for q in range(3)))
        return {"layers": layers, "aod": sum(c["aod"] * w for c, w in nodes)}

    def columns(self, points, t):
        # 10-minute time buckets make neighbouring forecasts share node columns
        t = t.replace(minute=t.minute - t.minute % 10, second=0, microsecond=0)
        return [self.column(la, lo, t) for la, lo in points]


def build_field(bbox, event, date, model, step):
    s, n, w, e = bbox
    clat, clon = (s + n) / 2, (w + e) / 2
    t_c = sg.find_event(date, clat, clon, event)
    _, az = sg.sun_position(t_c, clat, clon)
    lats, lons, times = [s, n], [w, e], []
    for la in (s, clat, n):
        for lo in (w, clon, e):
            times.append(sg.find_event(date, la, lo, event))
            for d in (sg.X_MIN, sg.X_MAX):
                p = sg.destination(la, lo, az, d)
                lats.append(p[0])
                lons.append(p[1])
    lat0 = math.floor(min(lats) / step) * step
    lon0 = math.floor(min(lons) / step) * step
    nlat = int(math.ceil((max(lats) - lat0) / step)) + 1
    nlon = int(math.ceil((max(lons) - lon0) / step)) + 1
    pts = [(lat0 + i * step, lon0 + j * step) for i in range(nlat) for j in range(nlon)]
    t0, t1 = min(times) - dt.timedelta(hours=2), max(times) + dt.timedelta(hours=2)
    print("field %d x %d nodes (%.2f°), %s .. %s UTC" % (nlat, nlon, step, t0.strftime("%H:%M"),
                                                         t1.strftime("%H:%M")), file=sys.stderr)
    # all levels matter (dropping 450/550 hPa once removed a light-blocking deck), so keep
    # them and save request weight with cloud_only instead
    raws = sg.fetch_raw(pts, t0, t1, model, request=cached_request, cloud_only=True)
    return GridField(lat0, lon0, nlat, nlon, step, raws)


# ---------------------------------------------------------------- per-point work

_FIELD = None
_OPTS = None


def _init(field, opts):
    global _FIELD, _OPTS
    _FIELD, _OPTS = field, opts


def _work(pt):
    lat, lon = pt
    try:
        r = sg.forecast(lat, lon, _OPTS["event"], _OPTS["date"], _FIELD.columns, _OPTS["haze"],
                        offsets=MAP_OFFSETS, angles=MAP_ANGLES)
    except SystemExit:  # polar day/night
        return None
    return {
        "lat": round(lat, 4), "lon": round(lon, 4), "index": round(r["index"], 3),
        "duration": r["duration"], "lit": r["lit_range"], "peak": r["peak_offset"],
        "hue": r["hue"], "color": "#%02x%02x%02x" % r["color"] if r["color"] else None,
        "aod": round(r["aod"], 3), "event": r["event_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def compute_map(bbox, step, field, event, date, haze, workers):
    s, n, w, e = bbox
    pts = [(s + i * step, w + j * step)
           for i in range(int(round((n - s) / step)) + 1) for j in range(int(round((e - w) / step)) + 1)]
    print("forecasting %d map points on %d workers ..." % (len(pts), workers), file=sys.stderr)
    opts = {"event": event, "date": date, "haze": haze}
    t = time.time()
    with multiprocessing.Pool(workers, initializer=_init, initargs=(field, opts)) as pool:
        res = []
        for k, r in enumerate(pool.imap(_work, pts, chunksize=4)):
            res.append(r)
            if (k + 1) % 50 == 0:
                print("  %d/%d  %.0f s" % (k + 1, len(pts), time.time() - t), file=sys.stderr)
    return [r for r in res if r]


# ---------------------------------------------------------------- HTML

def write_html(path, meta, cells):
    tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_template.html")
    with open(tpl_path, encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps({"meta": meta, "cells": cells}, ensure_ascii=False, separators=(",", ":"))
    html = html.replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description="Regional sunrise/sunset glow map.")
    ap.add_argument("--bbox", required=True, help="S,N,W,E in degrees, e.g. 33,38,136,142")
    ap.add_argument("--step", type=float, default=0.25, help="map resolution, degrees (default 0.25)")
    ap.add_argument("--field-step", type=float, default=0.5, help="model field resolution, degrees (default 0.5)")
    ap.add_argument("--event", choices=["sunset", "sunrise"], default="sunset")
    ap.add_argument("--date", help="local date YYYY-MM-DD (default: today)")
    ap.add_argument("--model", default="gfs_seamless")
    ap.add_argument("--haze-height", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default="map.html")
    args = ap.parse_args()

    bbox = tuple(float(v) for v in args.bbox.split(","))
    date = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    field = build_field(bbox, args.event, date, args.model, args.field_step)
    cells = compute_map(bbox, args.step, field, args.event, date, args.haze_height, args.workers)
    meta = {"event": args.event, "date": date.isoformat(), "model": args.model, "bbox": bbox,
            "step": args.step, "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    write_html(args.out, meta, cells)
    top = sorted(cells, key=lambda c: -c["index"])[:5]
    print("wrote %s  (%d cells, max index %.2f)" % (args.out, len(cells), top[0]["index"] if top else 0))
    for c in top:
        if c["index"] > 0:
            print("  %.2f  %.2f,%.2f  %s  %d min" % (c["index"], c["lat"], c["lon"], c["hue"], c["duration"]))


if __name__ == "__main__":
    main()
