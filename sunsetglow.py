#!/usr/bin/env python3
"""sunsetglow — open-source sunrise/sunset glow (火烧云) forecaster.

Method (following the ideas in sunsetbot's public tutorial, chapters 1-4):
  1. Find sunset/sunrise time and the sun's azimuth.
  2. Sample model columns along the sun's azimuth (-300 .. +700 km) and build a
     2-D atmospheric cross-section of cloud fraction (GFS pressure levels via
     Open-Meteo) and aerosol optical depth (CAMS via Open-Meteo).
  3. For every cloudy cell and every minute around sunset, trace the sunlight
     ray back toward the sun on a curved Earth (parabola in flattened coords),
     integrating Rayleigh + aerosol + cloud extinction at 450/550/650 nm.
     A ray that hits the terrain (Copernicus DEM profile along the azimuth) is blocked.
  4. Render what the observer sees: march sightlines across the meridian arc
     (sun-side horizon -> zenith -> anti-sun horizon), accumulate lit cloud
     along each line, attenuated by cloud/haze in front.
  5. Summarise as glow coverage over time -> an (uncalibrated) vividness index,
     duration, peak time and colour; optionally write a PNG panorama strip.

Pure standard library. Data: Open-Meteo (GFS / ECMWF) and CAMS, no API key.
"""
import argparse
import datetime as dt
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

R_EARTH = 6371.0  # km
LAMBDAS = (650.0, 550.0, 450.0)  # R, G, B in nm
K_RAYLEIGH_550 = 0.0116  # sea-level Rayleigh extinction at 550 nm, km^-1 (visibility ~ 336 km)
H_RAYLEIGH = 8.0  # km
ANGSTROM = 1.3  # aerosol extinction ~ lambda^-ANGSTROM
CLOUD_BETA_SCALE = 1.0  # multiplier on cloud extinction; tuned by backtest.py
# horizontal size of cloud elements in a partly cloudy layer. Backtest on 20 sunsetbot-graded Shanghai
# events (2022-06/07) improved monotonically up to 100 km (Spearman 0.39 at 0 -> 0.63 at 30 -> 0.76 at 100);
# on 87 eye-graded PhenoCam California events 30 km was about the best (AUC 0.67 at 0 -> 0.85 at 30, 0.83 at 100).
CLOUD_ELEMENT_KM = 30.0
LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200)

X_MIN, X_MAX, DX = -300.0, 700.0, 10.0  # cross-section along sun azimuth, km (+ = toward sun)
Z_MAX, DZ = 13.0, 0.25  # km
SAMPLE_STEP = 50.0  # km between model columns
# terrain profile along the azimuth: elevation every TERRAIN_DX km out to TERRAIN_X_MAX (sun rays from
# the far end of the section still skim the ground a few hundred km further on)
TERRAIN_DX = 2.0
TERRAIN_X_MAX = X_MAX + 300.0
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"


# ---------------------------------------------------------------- sun (NOAA)

def sun_position(t, lat, lon):
    """Return (elevation_deg, azimuth_deg) of the sun at UTC datetime t."""
    jd = t.timestamp() / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = math.radians(M)
    C = (math.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T))
         + math.sin(2 * Mr) * (0.019993 - 0.000101 * T) + math.sin(3 * Mr) * 0.000289)
    omega = math.radians(125.04 - 1934.136 * T)
    lam = math.radians(L0 + C - 0.00569 - 0.00478 * math.sin(omega))
    eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
    eps = math.radians(eps0 + 0.00256 * math.cos(omega))
    decl = math.asin(math.sin(eps) * math.sin(lam))
    y = math.tan(eps / 2) ** 2
    L0r = math.radians(L0)
    eot = 4 * math.degrees(y * math.sin(2 * L0r) - 2 * e * math.sin(Mr)
                           + 4 * e * y * math.sin(Mr) * math.cos(2 * L0r)
                           - 0.5 * y * y * math.sin(4 * L0r) - 1.25 * e * e * math.sin(2 * Mr))
    minutes = t.hour * 60 + t.minute + t.second / 60
    ha = math.radians(((minutes + eot + 4 * lon) % 1440) / 4 - 180)
    latr = math.radians(lat)
    cosz = math.sin(latr) * math.sin(decl) + math.cos(latr) * math.cos(decl) * math.cos(ha)
    elev = 90 - math.degrees(math.acos(max(-1, min(1, cosz))))
    az = (math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(latr)
                                  - math.tan(decl) * math.cos(latr))) + 180) % 360
    return elev, az


def find_event(date, lat, lon, event):
    """UTC time of sunset/sunrise (apparent elevation -0.833 deg) on a local solar date."""
    noon = dt.datetime(date.year, date.month, date.day, 12, tzinfo=dt.timezone.utc) - dt.timedelta(hours=lon / 15)
    sign = 1 if event == "sunset" else -1
    target = -0.833
    prev_t, prev_e = noon, sun_position(noon, lat, lon)[0]
    for k in range(1, 13 * 6 + 1):
        t = noon + sign * dt.timedelta(minutes=10 * k)
        e = sun_position(t, lat, lon)[0]
        if (prev_e - target) * (e - target) <= 0:
            a, b = prev_t, t
            for _ in range(30):
                m = a + (b - a) / 2
                if (sun_position(a, lat, lon)[0] - target) * (sun_position(m, lat, lon)[0] - target) <= 0:
                    b = m
                else:
                    a = m
            return a
        prev_t, prev_e = t, e
    raise SystemExit("No %s on this date at this latitude (polar day/night)." % event)


def destination(lat, lon, azimuth, dist_km):
    d = dist_km / R_EARTH
    th = math.radians(azimuth)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(th))
    l2 = l1 + math.atan2(math.sin(th) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


# ---------------------------------------------------------------- data

def http_json(url, params):
    full = url + "?" + urllib.parse.urlencode(params, safe=",")
    try:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(full, timeout=60) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, TimeoutError):  # transient network trouble
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            reason = e.read()[:300].decode("utf-8", "replace")
            if "Daily" in reason:  # callers retry 429s, but this one only clears at 00:00 UTC
                raise SystemExit("Open-Meteo daily limit exceeded (HTTP 429, 10000 weighted calls/day); retry after 00:00 UTC.")
            raise SystemExit("Open-Meteo rate limit hit (HTTP 429); each run costs ~60 weighted calls. Wait a minute and retry.")
        raise SystemExit("HTTP %d from %s: %s" % (e.code, url, e.read()[:300].decode("utf-8", "replace")))
    if isinstance(data, dict) and data.get("error"):
        raise SystemExit("API error: %s" % data.get("reason"))
    return data if isinstance(data, list) else [data]


def interp_hourly(series, times, t):
    """Linear interpolation of an hourly series (list) at UTC datetime t."""
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            a, b = series[i], series[i + 1]
            if a is None or b is None:
                return a if b is None else b
            f = (t - times[i]) / (times[i + 1] - times[i])
            return a + (b - a) * f
    return None


def parse_times(strings):
    return [dt.datetime.fromisoformat(v).replace(tzinfo=dt.timezone.utc) for v in strings]


ANCHOR_LEVELS = (850, 700, 500, 300, 200)


def isa_level(p_hpa):
    """Standard-atmosphere (height km, temperature C) of a pressure level."""
    z = (1 - (p_hpa / 1013.25) ** (1 / 5.25588)) / 2.25577e-5 / 1000.0
    return z, 15.0 - 6.5 * min(z, 11.0)


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# archived past forecasts (same variables), used for backtesting
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_raw(points, t0, t1, model, levels=LEVELS, request=None, cloud_only=False, url=FORECAST_URL):
    """Hourly level data + AOD for points between UTC t0 and t1.

    Returns one dict per point: {"times", "hourly", "atimes", "aod"}.
    `request(url, params)` defaults to http_json; the map passes a cached, chunked one.
    cloud_only fetches temperature only at ANCHOR_LEVELS and height only at 500 hPa
    (other levels are interpolated), cutting the Open-Meteo request weight by ~55%.
    """
    request = request or http_json
    lats = ",".join("%.4f" % p[0] for p in points)
    lons = ",".join("%.4f" % p[1] for p in points)
    base = {"latitude": lats, "longitude": lons, "timezone": "GMT",
            "start_date": t0.date().isoformat(), "end_date": t1.date().isoformat()}
    hourly = []
    for lv in levels:
        hourly += ["cloud_cover_%dhPa" % lv]
        if not cloud_only:
            hourly += ["geopotential_height_%dhPa" % lv, "temperature_%dhPa" % lv]
    if cloud_only:
        hourly += ["temperature_%dhPa" % lv for lv in ANCHOR_LEVELS] + ["geopotential_height_500hPa"]
    wx = request(url, dict(base, models=model, hourly=",".join(hourly)))
    if all(v is None for v in wx[0]["hourly"]["cloud_cover_500hPa"]):
        # model without level cloud cover: fetch RH and derive cloud fraction
        rh = request(url, dict(
            base, models=model, hourly=",".join("relative_humidity_%dhPa" % lv for lv in levels)))
        for w, r in zip(wx, rh):
            w["hourly"].update(r["hourly"])
    aq = request("https://air-quality-api.open-meteo.com/v1/air-quality", dict(base, hourly="aerosol_optical_depth"))
    return [{"times": parse_times(w["hourly"]["time"]), "hourly": w["hourly"],
             "atimes": parse_times(a["hourly"]["time"]), "aod": a["hourly"]["aerosol_optical_depth"]}
            for w, a in zip(wx, aq)]


def raw_to_column(raw, t, levels=LEVELS):
    """Column at UTC t: {"layers": [(z_km, temp_C, cloud_frac)], "aod"}; layers keep level order."""
    h, times = raw["hourly"], raw["times"]
    anchors = [(math.log(lv), interp_hourly(h["temperature_%dhPa" % lv], times, t))
               for lv in ANCHOR_LEVELS if "temperature_%dhPa" % lv in h]
    anchors = sorted(a for a in anchors if a[1] is not None)
    z500 = interp_hourly(h.get("geopotential_height_500hPa", [None] * len(times)), times, t)
    layers = []
    for lv in levels:
        z = temp = None
        if "geopotential_height_%dhPa" % lv in h:
            z = interp_hourly(h["geopotential_height_%dhPa" % lv], times, t)
        if "temperature_%dhPa" % lv in h:
            temp = interp_hourly(h["temperature_%dhPa" % lv], times, t)
        if z is None or temp is None:
            zi, ti = isa_level(lv)
            if z is None:  # scale the standard atmosphere to the actual 500 hPa height
                z = zi * 1000.0 * (z500 / (isa_level(500)[0] * 1000.0) if z500 else 1.0)
            if temp is None:
                temp = interp_log_p(anchors, math.log(lv)) if len(anchors) >= 2 else ti
        cc = interp_hourly(h["cloud_cover_%dhPa" % lv], times, t)
        if cc is None and "relative_humidity_%dhPa" % lv in h:  # Sundqvist scheme from RH
            rh = interp_hourly(h["relative_humidity_%dhPa" % lv], times, t)
            cc = rh_to_cloud(rh, temp) * 100 if rh is not None else None
        if z is None or temp is None or cc is None:
            continue
        layers.append((z / 1000.0, temp, cc / 100.0))
    aod = interp_hourly(raw["aod"], raw["atimes"], t)
    return {"layers": sorted(layers), "aod": aod if aod is not None else 0.2}


def interp_log_p(anchors, x):
    """Linear interpolation/extrapolation of (ln p, value) pairs sorted by ln p."""
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x <= x1 or (x1, y1) == anchors[-1]:
            if x < anchors[0][0]:
                x0, y0, x1, y1 = anchors[0] + anchors[1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def fetch_elevation(points, request=None):
    """Terrain height (m) of each point from Open-Meteo's elevation API (Copernicus DEM, 90 m)."""
    request = request or http_json
    out = []
    for i in range(0, len(points), 100):  # API limit per request
        chunk = points[i:i + 100]
        for d in request(ELEVATION_URL, {"latitude": ",".join("%.4f" % p[0] for p in chunk),
                                         "longitude": ",".join("%.4f" % p[1] for p in chunk)}):
            out += d["elevation"]
    return [max(e or 0.0, 0.0) for e in out]  # sea / missing -> 0


def terrain_xs():
    return [X_MIN + n * TERRAIN_DX for n in range(int((TERRAIN_X_MAX - X_MIN) / TERRAIN_DX) + 1)]


def fetch_columns(points, t, model):
    raws = fetch_raw(points, t - dt.timedelta(hours=2), t + dt.timedelta(hours=2), model)
    return [raw_to_column(r, t) for r in raws]


def rh_to_cloud(rh, temp):
    if temp is not None and temp < -20:  # rough ice-saturation correction
        rh = rh * (1 + 0.01 * (-20 - temp))
    rhc = 0.8
    r = min(rh / 100.0, 1.0)
    return 0.0 if r <= rhc else 1 - math.sqrt((1 - r) / (1 - rhc))


def cloud_beta(temp):
    """Cloud extinction per km at full cover: dense liquid ~10, thin ice ~1."""
    if temp >= -15:
        beta = 10.0
    elif temp <= -38:
        beta = 1.0
    else:
        beta = 10.0 + (temp + 15) / (-23.0) * (1.0 - 10.0)
    return beta * CLOUD_BETA_SCALE


# ---------------------------------------------------------------- cross-section

def effective_extinction(c, beta):
    """Mean extinction of a layer with cloud fraction c made of elements CLOUD_ELEMENT_KM wide.

    Spreading c uniformly (c * beta) makes even 30% cover opaque to grazing sunlight
    within a few km, although real broken cloud lets light through its gaps. Instead,
    each element-sized step is clear with probability 1 - c and crosses one cloud
    element (transmission exp(-beta * L)) otherwise.
    """
    if c <= 0 or beta <= 0:
        return 0.0
    L = CLOUD_ELEMENT_KM
    if L <= 0:  # uniform spreading (the original model)
        return c * beta
    t = 1 - min(c, 1.0) * (1 - math.exp(-beta * L))
    return beta if t <= 1e-12 else -math.log(t) / L



class Section:
    def __init__(self, cols, xs, terrain=None):
        """terrain: elevations (m) at terrain_xs(), or None for a flat sea-level Earth."""
        self._set_terrain(terrain)
        self.nx = int(round((X_MAX - X_MIN) / DX)) + 1
        self.nz = int(round(Z_MAX / DZ)) + 1
        self.ext = [[0.0] * self.nz for _ in range(self.nx)]  # cloud extinction km^-1
        self.cf = [[0.0] * self.nz for _ in range(self.nx)]  # cloud fraction
        self.aod = [0.0] * self.nx
        # cell k spans [k*DZ, (k+1)*DZ) and is sampled at its centre
        col_grid = [[self._column_at(c, (k + 0.5) * DZ) for k in range(self.nz)] for c in cols]
        for i in range(self.nx):
            x = X_MIN + i * DX
            j = min(max(int((x - xs[0]) // SAMPLE_STEP), 0), len(xs) - 2)
            f = min(max((x - xs[j]) / (xs[j + 1] - xs[j]), 0.0), 1.0)
            self.aod[i] = cols[j]["aod"] * (1 - f) + cols[j + 1]["aod"] * f
            for k in range(self.nz):
                c0, b0 = col_grid[j][k]
                c1, b1 = col_grid[j + 1][k]
                c = c0 * (1 - f) + c1 * f
                self.cf[i][k] = c
                self.ext[i][k] = effective_extinction(c, b0 * (1 - f) + b1 * f)

    def _set_terrain(self, terrain):
        # ground (km) on a 1 km grid from X_MIN, linearly interpolated, for sightlines; ridge = ground
        # dilated by +-3 km so sun rays marching in 5 km steps cannot skip over a narrow peak
        n = int(TERRAIN_X_MAX - X_MIN) + 1
        if not terrain:
            self.ground = self.ridge = [0.0] * n
            return
        g = []
        for m in range(n):
            f = m / TERRAIN_DX
            j = min(int(f), len(terrain) - 2)
            g.append((terrain[j] + (terrain[j + 1] - terrain[j]) * (f - j)) / 1000.0)
        self.ground = g
        self.ridge = [max(g[max(m - 3, 0):m + 4]) for m in range(n)]

    def ground_at(self, x, ridge=False):
        m = int(round(x - X_MIN))
        if m < 0 or m >= len(self.ground):
            return 0.0  # beyond the profile: assume sea level
        return (self.ridge if ridge else self.ground)[m]

    @staticmethod
    def _column_at(col, z):
        # Each pressure level owns the slab between the midpoints to its neighbours.
        # Linear interpolation would smear one cloudy level into a ~4 km thick slab
        # whose base can never be lit by grazing sunlight.
        L = col["layers"]
        for n, (lz, temp, c) in enumerate(L):
            lo = (L[n - 1][0] + lz) / 2 if n > 0 else lz - 0.2
            hi = (L[n + 1][0] + lz) / 2 if n + 1 < len(L) else lz + 0.5
            if lo <= z < hi:
                # spread a finite layer optical depth over the slab thickness
                return c, cloud_beta(temp) / max(hi - lo, 0.3)
        return 0.0, 0.0

    def idx(self, x, z):
        i = int(round((x - X_MIN) / DX))
        k = int(math.floor(z / DZ))
        if i < 0:
            i = 0
        elif i >= self.nx:
            i = self.nx - 1  # beyond the section: assume the last column persists
        if k < 0 or k >= self.nz:
            return i, -1
        return i, k


def sun_ray_transmission(sec, x, z, depression, ha):
    """Spectral transmission (R,G,B) of sunlight reaching (x, z).

    depression: sun depression angle at the observer (rad, >0 below horizon).
    In flattened coordinates the ray toward the sun from (x, z) is
    h(s) = z - d_x*s + s^2/(2R), with d_x = depression - x/R.
    """
    d_x = depression - x / R_EARTH
    ds = 5.0
    tau_r = tau_a = tau_c = 0.0
    s = 0.0
    while True:
        s += ds
        h = z - d_x * s + s * s / (2 * R_EARTH)
        xs = x + s
        if h <= sec.ground_at(xs, ridge=True):
            return (0.0, 0.0, 0.0)
        if h > 30.0 or s > 2500.0:
            break
        i, k = sec.idx(xs, h)
        tau_r += math.exp(-h / H_RAYLEIGH) * ds
        a = sec.aod[i]
        tau_a += a / ha * math.exp(-h / ha) * ds
        if k >= 0:
            tau_c += sec.ext[i][k] * ds
            if tau_c > 12:
                return (0.0, 0.0, 0.0)
    out = []
    for lam in LAMBDAS:
        kr = K_RAYLEIGH_550 * (550.0 / lam) ** 4
        ka = (550.0 / lam) ** ANGSTROM
        out.append(math.exp(-(kr * tau_r + ka * tau_a + tau_c)))
    return tuple(out)


def illumination_field(sec, depression, ha):
    field = {}
    for i in range(sec.nx):
        x = X_MIN + i * DX
        for k in range(sec.nz):
            if sec.cf[i][k] > 0.03 and (k + 1) * DZ > sec.ground_at(x):
                # light enters a cloud through its bottom face, so start just below the cell
                field[(i, k)] = sun_ray_transmission(sec, x, k * DZ - 0.01, depression, ha)
    return field


def render_column(sec, field, ha, angles):
    """For each meridian angle (deg from sun-side horizon) return lit-cloud RGB radiance."""
    pixels = []
    for ang in angles:
        a = math.radians(ang)
        dirx = 1.0 if ang < 90 else -1.0
        elev = a if ang < 90 else math.pi - a
        ce, se = math.cos(elev), math.sin(elev)
        T = 1.0
        tau_r = tau_a = 0.0
        acc = [0.0, 0.0, 0.0]
        s = 0.0
        z0 = sec.ground_at(0.0) + 0.002  # eye 2 m above the observer's ground
        while T > 0.01:
            ds = 1.0 if s < 100 else 2.0
            s += ds
            xh = s * ce
            h = z0 + s * se + xh * xh / (2 * R_EARTH)
            x = dirx * xh
            if h > Z_MAX or x > X_MAX or x < X_MIN:
                break
            if h < sec.ground_at(x):
                break  # mountains in front hide the rest of this sightline
            i, k = sec.idx(x, h)
            tau_r += math.exp(-h / H_RAYLEIGH) * ds
            tau_a += sec.aod[i] / ha * math.exp(-h / ha) * ds
            if k < 0:
                continue
            e = sec.ext[i][k]
            if e <= 0:
                continue
            hit = T * (1 - math.exp(-e * ds))
            L = field.get((i, k))
            if L:
                for c, lam in enumerate(LAMBDAS):
                    kr = K_RAYLEIGH_550 * (550.0 / lam) ** 4
                    ka = (550.0 / lam) ** ANGSTROM
                    acc[c] += hit * L[c] * math.exp(-(kr * tau_r + ka * tau_a))
            T -= hit
        pixels.append(tuple(acc))
    return pixels


# ---------------------------------------------------------------- scoring / output

def pixel_glow(rgb):
    """(glow weight 0..1, saturation 0..1) of one pixel.

    Brightness thresholds follow the tutorial: >1/30 of incident sunlight is bright,
    <1/1000 is of little value, <1/5000 is invisible.
    """
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum <= 2e-4:
        return 0.0, 0.0
    bright = min(max((math.log10(lum) - math.log10(2e-4)) / (math.log10(1 / 30) - math.log10(2e-4)), 0.0), 1.0)
    # golden/yellow light (B/R > 0.5) is not glow; deep red (B/R < 0.05) is fully saturated
    sat = min(max((0.5 - b / r) / 0.45, 0.0), 1.0) if r > 0 else 0.0
    return bright, sat


def to_display(rgb):
    r, g, b = rgb
    m = max(rgb)
    if m <= 0:
        return (12, 16, 34)
    w, _ = pixel_glow(rgb)
    out = []
    for c, base in zip((r, g, b), (12, 16, 34)):
        v = (c / m) ** (1 / 2.2) * 255
        out.append(int(base + (v - base) * w))
    return tuple(out)


def write_png(path, rows):
    h, w = len(rows), len(rows[0])
    raw = b"".join(b"\x00" + bytes(v for px in row for v in px) for row in rows)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def hue_name(rgb):
    r, g, b = rgb
    if r <= 0:
        return "-"
    gr, br = g / r, b / r
    if gr > 0.8:
        return "金黄/白"
    if gr > 0.55:
        return "橙黄"
    if gr > 0.35:
        return "橙红"
    return "深红" if br < 0.15 else "粉紫"


DEFAULT_ANGLES = [1 + 0.5 * i for i in range(357)]  # meridian arc 1..179 deg
DEFAULT_OFFSETS = list(range(-10, 46, 2))  # minutes after sunset (before sunrise)


def forecast(lat, lon, event, date, columns, haze_height=1.0, offsets=DEFAULT_OFFSETS,
             angles=DEFAULT_ANGLES, verbose=False, terrain=None):
    """Glow forecast for one location.

    columns(points, t) -> list of columns (see raw_to_column) for (lat, lon) points at UTC t.
    terrain(points) -> elevations (m), e.g. fetch_elevation; None = flat sea-level Earth.
    """
    t_event = find_event(date, lat, lon, event)
    _, az = sun_position(t_event, lat, lon)
    sign = 1 if event == "sunset" else -1
    t_mid = t_event + sign * dt.timedelta(minutes=15)

    xs = [X_MIN + i * SAMPLE_STEP for i in range(int((X_MAX - X_MIN) / SAMPLE_STEP) + 1)]
    relief = terrain([destination(lat, lon, az, x) for x in terrain_xs()]) if terrain else None
    sec = Section(columns([destination(lat, lon, az, x) for x in xs], t_mid), xs, relief)

    strip, coverage = [], []
    for m in offsets:
        t = t_event + sign * dt.timedelta(minutes=m)
        elev, _ = sun_position(t, lat, lon)
        field = illumination_field(sec, math.radians(-elev), haze_height)
        col = render_column(sec, field, haze_height, angles)
        strip.append(col)
        g = [pixel_glow(p) for p in col]
        coverage.append(sum(w * sat for w, sat in g) / len(g))
        if verbose:
            print("  %+3d min  sun %5.1f°  glow coverage %5.1f%%" % (m * sign, elev, coverage[-1] * 100), file=sys.stderr)

    step = offsets[1] - offsets[0]
    # full meridian arc lit for 6 min ~ 1.0. Scaled so that the backtest cut for
    # "small glow or better" (小烧) falls near 1.0; grades in map_template.html
    index = sum(c * step for c in coverage) / 6
    lit = [m * sign for m, c in zip(offsets, coverage) if c > 0.02]
    peak_i = max(range(len(coverage)), key=lambda i: coverage[i])
    best = max(strip[peak_i], key=lambda p: pixel_glow(p)[0] * (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]))
    return {
        "event_utc": t_event, "azimuth": az, "index": index,
        "duration": len(lit) * step, "lit_range": (min(lit), max(lit)) if lit else None,
        "peak_offset": offsets[peak_i] * sign, "peak_coverage": coverage[peak_i],
        "color": to_display(best) if lit else None, "hue": hue_name(best) if lit else None,
        "aod": sec.aod[int(round(-X_MIN / DX))],
        "elevation": sec.ground_at(0.0) * 1000.0,
        "strip": strip, "offsets": offsets, "angles": angles,
    }


def main():
    ap = argparse.ArgumentParser(description="Sunrise/sunset glow forecast (open-source sunsetbot-style).")
    ap.add_argument("lat", type=float)
    ap.add_argument("lon", type=float)
    ap.add_argument("--event", choices=["sunset", "sunrise"], default="sunset")
    ap.add_argument("--date", help="local date YYYY-MM-DD (default: today)")
    ap.add_argument("--model", default="gfs025", help="Open-Meteo model (default gfs025: pure GFS; gfs_seamless blends in HRRR over the US, which misses high cloud), e.g. ecmwf_ifs025")
    ap.add_argument("--haze-height", type=float, default=1.0, help="aerosol scale height km (default 1.0, from backtest)")
    ap.add_argument("--flat", action="store_true", help="ignore terrain (flat sea-level Earth)")
    ap.add_argument("--png", help="write a panorama PNG (time x meridian angle)")
    args = ap.parse_args()

    date = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    print("fetching model columns along the sun azimuth ...", file=sys.stderr)
    r = forecast(args.lat, args.lon, args.event, date,
                 lambda pts, t: fetch_columns(pts, t, args.model), args.haze_height, verbose=True,
                 terrain=None if args.flat else fetch_elevation)

    print("%s %s UTC  sun azimuth %.1f°  model %s"
          % (args.event, r["event_utc"].strftime("%Y-%m-%d %H:%M"), r["azimuth"], args.model))
    print()
    print("鲜艳度指数(未校准)  %.2f" % r["index"])
    if r["lit_range"]:
        print("持续时间            约 %d 分钟 (%+d ~ %+d min)" % ((r["duration"],) + r["lit_range"]))
        print("峰值                %+d min, 覆盖 %.0f%% 天空弧, 颜色 %s"
              % (r["peak_offset"], r["peak_coverage"] * 100, r["hue"]))
    else:
        print("预计无明显火烧云")
    print("本地 AOD            %.2f" % r["aod"])
    print("观测点海拔          %.0f m%s" % (r["elevation"], "（--flat：忽略地形）" if args.flat else ""))

    if args.png:
        # rows = meridian angle (top = sun-side horizon), columns = time; 6x upscale in time
        strip = r["strip"]
        rows = [[px for ti in range(len(strip)) for px in [to_display(strip[ti][ai])] * 6]
                for ai in range(len(r["angles"]))]
        write_png(args.png, rows)
        print("panorama            %s  (纵轴: 太阳侧地平线→天顶→背日侧, 横轴: 时间)" % args.png)


if __name__ == "__main__":
    main()
