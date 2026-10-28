"""Sanity checks with synthetic cross-sections (no network)."""
import math
import sunsetglow as s

xs = [s.X_MIN + i * s.SAMPLE_STEP for i in range(int((s.X_MAX - s.X_MIN) / s.SAMPLE_STEP) + 1)]
LEVELS_Z = [(0.1, 15), (1.0, 8), (1.5, 5), (3.0, -5), (5.5, -20), (7.2, -32), (9.2, -45), (11.7, -55)]


def column(cloud_z, cover, aod):
    return {"aod": aod, "layers": [(z, t, cover if abs(z - cloud_z) < 0.1 else 0.0) for z, t in LEVELS_Z]}


def run(cloud_z, edge_km, aod, label):
    cols = [column(cloud_z, 0.9 if x <= edge_km else 0.0, aod) for x in xs]
    sec = s.Section(cols, xs)
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
