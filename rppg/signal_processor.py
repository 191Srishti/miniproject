
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt


def process_green_signal(timestamps, green_values):
    """
    Convert collected green-channel measurements into
    a filtered, pulse-related rPPG signal.

    Returns None until enough usable samples are available.
    """

    t = np.asarray(timestamps, dtype=float)
    green = np.asarray(green_values, dtype=float)

    # Require enough data before filtering.
    if len(t) < 90 or len(green) < 90:
        return None

    if len(t) != len(green):
        return None

    # Estimate webcam sampling rate from timestamps.
    intervals = np.diff(t)

    if np.any(intervals <= 0):
        return None

    fs = 1.0 / np.median(intervals)

    # Check that the sampling rate supports our filter.
    if fs < 8:
        return None

    # Remove slow trends, such as gradual brightness changes.
    signal = detrend(green, type="linear")

    # Band-pass range: approximately 0.7–3.0 Hz.
    low_cut = 0.7
    high_cut = min(3.0, fs * 0.40)

    if high_cut <= low_cut:
        return None

    sos = butter(
        3,
        [low_cut, high_cut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError:
        return None

    # Reject near-constant signals with no useful variation.
    if np.std(filtered) < 0.01:
        return None

    return {
        "signal": filtered,
        "sampling_rate": fs,
        "duration": t[-1] - t[0]
    }
