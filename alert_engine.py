
import time
from collections import deque

try:
    import winsound
except ImportError:
    winsound = None


class AlertEngine:

    def __init__(
        self,
        heart_rate_limits=None,
        perclos_threshold=0.20,
        prolonged_closure_seconds=2.0,
        heart_rate_sustain_seconds=8.0,
        signal_timeout_seconds=5.0,
        face_timeout_seconds=2.0,
        perclos_window_seconds=60.0
    ):

        # Configure heart-rate limits after validation.
        self.heart_rate_limits = heart_rate_limits

        self.perclos_threshold = perclos_threshold
        self.prolonged_closure_seconds = prolonged_closure_seconds
        self.heart_rate_sustain_seconds = heart_rate_sustain_seconds

        self.signal_timeout_seconds = signal_timeout_seconds
        self.face_timeout_seconds = face_timeout_seconds

        self.perclos_window_seconds = perclos_window_seconds

        # Rolling eye-closure history
        self.eye_history = deque()

        self.hr_outside_since = None
        self.face_missing_since = None
        self.signal_missing_since = None

        self.previous_alerts = set()

    def calculate_perclos(self, now):

        # Remove samples outside the rolling time window.
        while self.eye_history:
            if now - self.eye_history[0][0] > self.perclos_window_seconds:
                self.eye_history.popleft()
            else:
                break

        if not self.eye_history:
            return 0.0

        # Avoid calculating fatigue from a very short observation.
        observation_time = (
            self.eye_history[-1][0] - self.eye_history[0][0]
        )

        if observation_time < 10:
            return None

        closed_samples = sum(
            1 for _, closed in self.eye_history if closed
        )

        return closed_samples / len(self.eye_history)

    def play_sound(self):

        if winsound is not None:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except RuntimeError:
                pass
        else:
            print("\a", end="", flush=True)

    def update(
        self,
        face_detected,
        signal_valid,
        bpm,
        eye_closed,
        closure_duration,
        now=None
    ):

        if now is None:
            now = time.monotonic()

        alerts = set()
        messages = []

        # ---------------------------------
        # 1. SIGNAL / FACE ALERT
        # ---------------------------------

        if not face_detected:

            if self.face_missing_since is None:
                self.face_missing_since = now

            self.signal_missing_since = None

            if now - self.face_missing_since >= self.face_timeout_seconds:
                alerts.add("SIGNAL")
                messages.append("Face not detected - check camera position")

        else:

            self.face_missing_since = None

            if not signal_valid:

                if self.signal_missing_since is None:
                    self.signal_missing_since = now

                if now - self.signal_missing_since >= self.signal_timeout_seconds:
                    alerts.add("SIGNAL")
                    messages.append("Poor rPPG signal - check lighting and movement")

            else:
                self.signal_missing_since = None

        # ---------------------------------
        # 2. FATIGUE ALERT
        # ---------------------------------

        perclos = None

        if face_detected and eye_closed is not None:

            self.eye_history.append((now, bool(eye_closed)))

            perclos = self.calculate_perclos(now)

            prolonged_closure = (
                closure_duration >= self.prolonged_closure_seconds
            )

            high_perclos = (
                perclos is not None
                and perclos >= self.perclos_threshold
            )

            if prolonged_closure or high_perclos:

                alerts.add("FATIGUE")

                if prolonged_closure:
                    messages.append("Prolonged eye closure detected")

                if high_perclos:
                    messages.append(
                        f"High eye-closure percentage: {perclos * 100:.1f}%"
                    )

        # ---------------------------------
        # 3. HEART-RATE ALERT
        # ---------------------------------

        if (
            self.heart_rate_limits is not None
            and signal_valid
            and bpm is not None
        ):

            low, high = self.heart_rate_limits

            outside_limits = bpm < low or bpm > high

            if outside_limits:

                if self.hr_outside_since is None:
                    self.hr_outside_since = now

                if now - self.hr_outside_since >= self.heart_rate_sustain_seconds:

                    alerts.add("HEART")

                    messages.append(
                        f"Sustained unusual heart-rate reading: {bpm:.1f} BPM"
                    )

            else:
                self.hr_outside_since = None

        else:
            self.hr_outside_since = None

        # ---------------------------------
        # 4. COMBINED ALERT
        # ---------------------------------

        if "FATIGUE" in alerts and "HEART" in alerts:

            alerts.add("COMBINED")

            messages.append(
                "Combined monitoring warning: fatigue and heart-rate indicators"
            )

        # ---------------------------------
        # 5. SOUND NOTIFICATION
        # ---------------------------------

        # Beep only when a new alert becomes active.
        new_alerts = alerts - self.previous_alerts

        if new_alerts:
            self.play_sound()

        self.previous_alerts = alerts.copy()

        return {
            "active_alerts": alerts,
            "messages": messages,
            "perclos": perclos,
            "status": "WARNING" if alerts else "NORMAL"
        }
