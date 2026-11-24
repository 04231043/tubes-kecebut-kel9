"""Single-input Mamdani FIS for a project-specific CO monitoring priority."""
import numpy as np

LEVELS = ["Rendah", "Terpantau", "Meningkat", "Tinggi", "Sangat tinggi"]
ACTIONS = ["Catat hasil pemantauan rutin.", "Amati tren pada jam berikutnya.",
           "Bandingkan hasil dengan pengukuran referensi.", "Prioritaskan verifikasi pengukuran.",
           "Segera periksa sensor dan ulangi pengukuran referensi."]
IN_CENTERS = [0., 2., 4., 7., 10.]
OUT_CENTERS = [0., 25., 50., 75., 100.]
GRID = np.linspace(0., 100., 201)


def memberships(values, centers):
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all():
        raise ValueError("Masukan fuzzy harus berupa angka hingga.")
    c = np.array(centers)
    # Linear partitions with shoulders at the two ends.
    return np.stack([np.interp(v, c, row) for row in np.eye(len(c))], axis=-1)


def fuzzy_detail(co):
    if not np.isfinite(co) or co < 0:
        raise ValueError("Konsentrasi CO harus berupa angka nonnegatif.")
    degrees = memberships(float(co), IN_CENTERS)
    consequent = memberships(GRID, OUT_CENTERS)
    aggregate = np.minimum(degrees, consequent).max(axis=1)
    score = float(np.trapezoid(GRID * aggregate, GRID) / np.trapezoid(aggregate, GRID))
    idx = int(np.argmin(abs(np.array(OUT_CENTERS) - score)))
    return score, idx, degrees, aggregate


def compute_fuzzy_risk(co_value):
    score, idx, _, _ = fuzzy_detail(float(co_value))
    return score, LEVELS[idx], ["🟢", "🟡", "🟠", "🔴", "🟣"][idx], ACTIONS[idx]
