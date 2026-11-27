
import numpy as np
from scipy.signal import welch


def estimate_heart_rate(filtered_signal, sampling_rate, duration):
    """Return a preliminary BPM estimate, or None if unusable."""

    signal = np.asarray(filtered_signal, dtype=float)

    if duration < 10 or len(signal) < 100:
        return None

    if sampling_rate < 8 or not np.all(np.isfinite(signal)):
        return None

    if np.std(signal) < 0.01:
        return None

    # Use a longer segment when available for better frequency resolution.
    nperseg = min(len(signal), int(sampling_rate * 15))

    frequencies, power = welch(
        signal,
        fs=sampling_rate,
        nperseg=nperseg,
        detrend="constant"
    )

    # 42–180 BPM is the initial search range.
    valid = (frequencies >= 0.7) & (frequencies <= 3.0)
    frequencies = frequencies[valid]
    power = power[valid]

    if len(power) < 3 or not np.all(np.isfinite(power)):
        return None

    peak_index = int(np.argmax(power))
    peak_power = power[peak_index]
    median_power = np.median(power)

    if median_power <= 0:
        return None

    peak_ratio = peak_power / median_power

    # Compare the peak against nearby frequencies too.
    # Exclude the peak and its immediate neighbors.
    background = power.copy()

    start = max(0, peak_index - 1)
    end = min(len(background), peak_index + 2)
    background[start:end] = np.nan

    local_background = np.nanmedian(background)

    if not np.isfinite(local_background) or local_background <= 0:
        return None

    local_peak_ratio = peak_power / local_background

    # Initial experimental thresholds; tune using reference data.
    if peak_ratio < 3.0 or local_peak_ratio < 2.0:
        return None

    bpm = frequencies[peak_index] * 60.0

    return {
        "bpm": float(bpm),
        "peak_ratio": float(peak_ratio),
        "local_peak_ratio": float(local_peak_ratio)
    }


