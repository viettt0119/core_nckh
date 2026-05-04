from __future__ import annotations

DIRECTION_LABELS = ("left", "right", "center", "up", "down")
DIRECTION_TO_INDEX = {name: idx for idx, name in enumerate(DIRECTION_LABELS)}


def direction_from_xy(x: float, y: float, center: float = 0.20) -> str:
    dx = float(x) - 0.5
    dy = float(y) - 0.5
    if abs(dx) <= center and abs(dy) <= center:
        return "center"
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "up" if dy < 0 else "down"


def xy_for_direction(direction: str) -> tuple[float, float]:
    points = {
        "left": (0.18, 0.50),
        "right": (0.82, 0.50),
        "center": (0.50, 0.50),
        "up": (0.50, 0.18),
        "down": (0.50, 0.82),
    }
    return points[direction]
