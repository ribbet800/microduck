"""Imaginary legs for testing brains without MuJoCo."""


def lazy_legs(cmd, walking):
    """What velstand actually does with a command (measured in MuJoCo with BAM,
    2026-10-23): it only starts walking at vx >= ~0.25 m/s or |wz| >= ~0.7 rad/s,
    keeps going down to vx 0.15 once walking, never sidesteps or walks
    backwards, and reaches ~30% of the commanded forward speed.
    Returns (vx, vy, wz achieved, walking)."""
    vx, _, wz = (float(c) for c in cmd)
    if vx >= (0.15 if walking else 0.25):
        return 0.3 * vx, 0.0, 0.8 * wz, True
    if abs(wz) >= 0.7:
        return 0.0, 0.0, 0.8 * wz, True
    return 0.0, 0.0, 0.0, False
