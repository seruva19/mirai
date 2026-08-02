"""Video frame selection helpers."""

from __future__ import annotations


def select_evenly_spaced_indices_from_pts(
    pts_values: list[float],
    frame_count: int,
) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0.")
    if not pts_values:
        raise ValueError("pts_values cannot be empty.")
    if frame_count == 1:
        return [0]
    if frame_count >= len(pts_values):
        return list(range(len(pts_values)))

    start = float(pts_values[0])
    end = float(pts_values[-1])
    targets = [
        start + (end - start) * (i / float(frame_count - 1))
        for i in range(frame_count)
    ]

    out: list[int] = []
    cursor = 0
    for target in targets:
        best_idx = cursor
        best_dist = abs(float(pts_values[best_idx]) - target)
        probe = cursor + 1
        while probe < len(pts_values):
            dist = abs(float(pts_values[probe]) - target)
            if dist <= best_dist:
                best_idx = probe
                best_dist = dist
                probe += 1
                continue
            break
        cursor = best_idx
        out.append(best_idx)
    return out
