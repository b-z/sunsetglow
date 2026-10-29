"""Sanity checks with synthetic cross-sections (no network)."""
import math
import sunsetglow as s

xs = [s.X_MIN + i * s.SAMPLE_STEP for i in range(int((s.X_MAX - s.X_MIN) / s.SAMPLE_STEP) + 1)]
LEVELS_Z = [(0.1, 15), (1.0, 8), (1.5, 5), (3.0, -5), (5.5, -20), (7.2, -32), (9.2, -45), (11.7, -55)]


def column(cloud_z, cover, aod):
    return {"aod": aod, "layers": [(z, t, cover if abs(z - cloud_z) < 0.1 else 0.0) for z, t in LEVELS_Z]}


def ridge(top_km, at_km, half_width_km=20):
    """Elevation profile (m) with one triangular ridge toward the sun."""
    return [max(0.0, top_km * 1000 * (1 - abs(x - at_km) / half_width_km)) for x in s.terrain_xs()]


def run(cloud_z, edge_km, aod, label, terrain=None):
    cols = [column(cloud_z, 0.9 if x <= edge_km else 0.0, aod) for x in xs]
    sec = s.Section(cols, xs, terrain)
    angles = [1 + 0.5 * i for i in range(357)]
    cov = []
    for m in range(-10, 46, 2):
        dep = math.radians(0.833 + m * 0.2)  # ~0.2 deg/min at mid latitudes
        col = s.render_column(sec, s.illumination_field(sec, dep, 1.5), 1.5, angles)
        g = [s.pixel_glow(p) for p in col]
        cov.append(sum(w * sat for w, sat in g) / len(g))
    print("%-34s index %.2f  lit-steps %d" % (label, sum(c * 2 for c in cov) / 6, sum(c > 0.02 for c in cov)))


run(9.2, 100, 0.1, "cirrus 9 km, edge +100 km, clean")
run(9.2, 100, 0.8, "cirrus 9 km, edge +100 km, hazy")
run(5.5, 100, 0.1, "altocumulus 5.5 km, edge +100 km")
run(1.5, 100, 0.1, "low cloud 1.5 km, edge +100 km")
run(9.2, 700, 0.1, "cirrus 9 km, no edge (overcast)")

# terrain: what matters is how high a ridge stands above the cloud as seen from the cloud. A 2 km ridge
# 80 km away hardly shades 1.5 km cloud overhead (it rises only ~0.4 deg above it); a 3 km ridge 40 km
# away does, and high cloud stays lit above it
run(1.5, 100, 0.1, "low cloud 1.5 km + ridge 2 km @80", ridge(2.0, 80))
run(1.5, 100, 0.1, "low cloud 1.5 km + ridge 3 km @40", ridge(3.0, 40))
run(5.5, 100, 0.1, "altocumulus 5.5 km + ridge 3 km @40", ridge(3.0, 40))
run(9.2, 100, 0.1, "cirrus 9 km + ridge 3 km @40", ridge(3.0, 40))
