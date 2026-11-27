
import cv2
import numpy as np


def draw_waveform(frame, signal, x=30, y=380, width=580, height=110):
    """Draw a normalized signal waveform onto an OpenCV frame."""

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (180, 180, 180),
        1
    )

    cv2.putText(
        frame,
        "Filtered rPPG waveform (relative units)",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    if signal is None or len(signal) < 2:
        cv2.putText(
            frame,
            "Collecting signal...",
            (x + 10, y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        return

    values = np.asarray(signal, dtype=float)

    if not np.all(np.isfinite(values)):
        return

    # Normalize for display only; the actual signal is unchanged.
    values = values - np.mean(values)
    scale = np.max(np.abs(values))

    if scale < 1e-9:
        return

    values = values / scale

    # Downsample for display if the signal has many samples.
    indices = np.linspace(
        0, len(values) - 1, min(width, len(values))
    ).astype(int)

    points = []

    for i, index in enumerate(indices):
        px = x + int(i * (width - 1) / max(1, len(indices) - 1))
        py = y + height // 2 - int(values[index] * (height * 0.40))
        points.append((px, py))

    if len(points) >= 2:
        cv2.polylines(
            frame,
            [np.array(points, dtype=np.int32)],
            False,
            (0, 255, 255),
            2
        )

    # Center line
    cv2.line(
        frame,
        (x, y + height // 2),
        (x + width, y + height // 2),
        (90, 90, 90),
        1
    )

