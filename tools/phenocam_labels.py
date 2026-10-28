#!/usr/bin/env python3
"""Automatic sunset-glow labels from PhenoCam webcam archives.

PhenoCam (https://phenocam.nau.edu) keeps ~30-minute images from research
webcams, many of them west-facing with open sky. For each sampled date this
script picks the frames between sunset+3 and sunset+35 min, measures the share
of sky pixels with glow colour (red/orange/pink, saturated, not too dark), and
writes a labels CSV for backtest.py with observed = min(4, 10 * share), i.e.
observed >= 2 means at least 20% of the visible sky glows.

The sky mask of each site comes from three midday images (bright pixels in the
upper part of the frame that are sky in at least two of them), minus the text
overlay box in the top-left corner.

Labels are noisy: 30-minute cadence can miss short glow, auto exposure shifts
colours, and the view is one direction only. Use them for ranking checks, and
look at the montage (--montage) before trusting them.

Requires Pillow (pip install pillow). PhenoCam data: free to use with
acknowledgement of the PhenoCam Network.
"""
import argparse
import colorsys
import csv
import datetime as dt
import io
import json
import os
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sunsetglow as sg  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

BASE = "https://phenocam.nau.edu"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "phenocam")
# west-facing open-sky sites whose sky is correctly exposed at dusk. Tried and dropped: meadpasturesw,
# saskatoon1 (sky blown out to white), laudernz01 (blown out; dusk frames black)
SITES = ["hillslough", "eastend"]
SMALL = (324, 240)  # analysis resolution


def get(url, binary=False):
    path = os.path.join(CACHE, re.sub(r"[^A-Za-z0-9_.-]", "_", url.replace(BASE, "")))
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
    else:
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        time.sleep(0.3)  # be gentle with the archive
    return data if binary else data.decode("utf-8", "replace")


def site_info(name):
    # the API ignores Sitename filters, so look the site up in the full list
    d = json.loads(get(BASE + "/api/cameras/?format=json&limit=5000"))
    r = next(c for c in d["results"] if c["Sitename"] == name)
    return {"name": name, "lat": r["Lat"], "lon": r["Lon"], "utc_offset": r["utc_offset"],
            "first": r["date_first"], "last": r["date_last"]}


def day_images(site, day):
    """[(local standard datetime, archive URL)] for one local date."""
    html = get("%s/webcam/browse/%s/%04d/%02d/%02d/" % (BASE, site, day.year, day.month, day.day))
    out = []
    for m in sorted(set(re.findall(r"%s_(\d{4})_(\d{2})_(\d{2})_(\d{6})\.jpg" % site, html))):
        y, mo, d, hms = m
        t = dt.datetime(int(y), int(mo), int(d), int(hms[:2]), int(hms[2:4]), int(hms[4:]))
        out.append((t, "%s/data/archive/%s/%s/%s/%s_%s_%s_%s_%s.jpg" % (BASE, site, y, mo, site, y, mo, d, hms)))
    return out


def load_small(url):
    return Image.open(io.BytesIO(get(url, binary=True))).convert("RGB").resize(SMALL)


def sky_mask(site, days):
    votes = [0] * (SMALL[0] * SMALL[1])
    used = 0
    for day in days:
        imgs = [u for t, u in day_images(site, day) if 11 <= t.hour <= 13]
        if not imgs:
            continue
        px = list(load_small(imgs[len(imgs) // 2]).get_flattened_data())
        for i, (r, g, b) in enumerate(px):
            y = i // SMALL[0]
            if y < 0.6 * SMALL[1] and (r + g + b) / 3 > 120 and b >= r - 15:
                votes[i] += 1
        used += 1
    need = min(2, used)
    mask = [v >= need and need > 0 for v in votes]
    # drop the band just above the horizon: twilight glow and the sun itself live there
    w, h = SMALL
    band = int(0.12 * h)  # clear-sky twilight is a thin band; lit clouds reach higher
    for x in range(w):
        rows = [y for y in range(h) if mask[y * w + x]]
        if rows:
            for y in range(max(rows) - band, max(rows) + 1):
                mask[y * w + x] = False
    for i in range(len(mask)):  # overlay text box, top-left
        x, y = i % SMALL[0], i // SMALL[0]
        if x < 0.65 * SMALL[0] and y < 0.10 * SMALL[1]:
            mask[i] = False
    return mask


def sun_visible(img):
    """Camera clocks at some sites drift by an hour in DST; a saturated blob means the sun is still up."""
    hot = sum(1 for r, g, b in img.get_flattened_data() if min(r, g, b) >= 250)
    return hot >= 40


def glow_share(img, mask):
    n = glow = 0
    for (r, g, b), m in zip(img.get_flattened_data(), mask):
        if not m:
            continue
        n += 1
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        hue = h * 360
        if (hue <= 50 or hue >= 300) and s >= 0.35 and v >= 0.25:
            glow += 1
    return glow / n if n else 0.0


def write_sheets(prefix, rows, per_sheet=24):
    """Contact sheets: one row per event, its post-sunset frames left to right, labelled by id."""
    tw, th, cols = 200, 148, 3
    for n in range(0, len(rows), per_sheet):
        chunk = rows[n:n + per_sheet]
        pairs = 2  # events per sheet row
        sheet = Image.new("RGB", (tw * cols * pairs, (th + 14) * ((len(chunk) + pairs - 1) // pairs)), "white")
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            x0, y = (k % pairs) * tw * cols, (k // pairs) * (th + 14)
            d.text((x0 + 2, y + 1), "#%d %s %s" % (r["id"], r["source"].split(":")[1], r["date"]), fill="black")
            for j, u in enumerate(r["frames"][:cols]):
                sheet.paste(load_small(u).resize((tw, th)), (x0 + j * tw, y + 14))
        path = "%s_%d.jpg" % (prefix, n // per_sheet + 1)
        sheet.save(path, quality=82)
        print("sheet -> %s" % path)


def main():
    ap = argparse.ArgumentParser(description="Sunset glow labels from PhenoCam archives.")
    ap.add_argument("--sites", default=",".join(SITES))
    ap.add_argument("--start", default="2022-09-01", help="first date (CAMS AOD archive starts ~2022-08)")
    ap.add_argument("--end", default="2025-08-31")
    ap.add_argument("--per-site", type=int, default=24, help="random dates per site")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--window", type=lambda v: [float(x) for x in v.split(",")], default=[3, 35],
                    help="minutes after sunset to look at (default 3,35: the sun is down)")
    ap.add_argument("--out", default="labels_phenocam.csv")
    ap.add_argument("--montage", help="write a JPEG of the evening frames sorted by score")
    ap.add_argument("--sheets", help="write contact sheets PREFIX_N.jpg (all frames per event, labelled by id) "
                                     "for grading by eye")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    rows, thumbs = [], []
    for name in args.sites.split(","):
        info = site_info(name)
        lo = max(start, dt.date.fromisoformat(info["first"]))
        hi = min(end, dt.date.fromisoformat(info["last"]))
        span = (hi - lo).days
        mask = sky_mask(name, [lo + dt.timedelta(days=rng.randrange(span)) for _ in range(4)])
        print("%s: sky %.0f%% of frame" % (name, 100 * sum(mask) / len(mask)), file=sys.stderr)
        if sum(mask) < 0.05 * len(mask):
            print("  too little sky, skipped", file=sys.stderr)
            continue
        days = sorted(lo + dt.timedelta(days=rng.randrange(span)) for _ in range(args.per_site))
        for day in days:
            t_set = sg.find_event(day, info["lat"], info["lon"], "sunset")
            local = (t_set + dt.timedelta(hours=info["utc_offset"])).replace(tzinfo=None)
            frames = [(t, u) for t, u in day_images(name, day)
                      if args.window[0] <= (t - local).total_seconds() / 60 <= args.window[1]]
            if not frames:
                continue
            scored = []
            for t, u in frames:
                img = load_small(u)
                if not sun_visible(img):
                    scored.append((glow_share(img, mask), t, u))
            if not scored:
                continue
            share, t_best, u_best = max(scored)
            rows.append({"date": day.isoformat(), "lat": info["lat"], "lon": info["lon"], "event": "sunset",
                         "observed": round(min(4.0, 10 * share), 2), "source": "phenocam:" + name,
                         "note": "glow %.1f%% at %+d min %s" % (100 * share, (t_best - local).total_seconds() / 60,
                                                              u_best.rsplit("/", 1)[1])})
            thumbs.append((share, u_best, name + " " + day.isoformat()))
            rows[-1]["frames"] = [u for _, _, u in sorted(scored, key=lambda x: x[1])]
            print("  %s %s  glow %5.1f%%  (%d frames)" % (name, day, 100 * share, len(frames)), file=sys.stderr)

    for i, r in enumerate(rows):
        r["id"] = i
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["id", "date", "lat", "lon", "event", "observed", "source", "note"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("%d events -> %s (%d with observed >= 2)" % (len(rows), args.out, sum(r["observed"] >= 2 for r in rows)))

    if args.sheets:
        write_sheets(args.sheets, rows)

    if args.montage and thumbs:
        thumbs.sort(reverse=True)
        tw, th, cols = 216, 160, 6
        sheet = Image.new("RGB", (tw * cols, (th + 14) * ((len(thumbs) + cols - 1) // cols)), "white")
        d = ImageDraw.Draw(sheet)
        for i, (share, url, label) in enumerate(thumbs):
            x, y = (i % cols) * tw, (i // cols) * (th + 14)
            sheet.paste(load_small(url).resize((tw, th)), (x, y + 14))
            d.text((x + 2, y + 1), "%s %.0f%%" % (label, 100 * share), fill="black")
        sheet.save(args.montage, quality=80)
        print("montage -> %s" % args.montage)


if __name__ == "__main__":
    main()
