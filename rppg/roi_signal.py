
import cv2
import numpy as np
from collections import deque
import time


class RPPGSignalCollector:
    def __init__(self, max_samples=600):
        self.timestamps = deque(maxlen=max_samples)
        self.red_values = deque(maxlen=max_samples)
        self.green_values = deque(maxlen=max_samples)
        self.blue_values = deque(maxlen=max_samples)

    def process_frame(self, frame, landmarks):
        """
        frame: OpenCV BGR video frame
        landmarks: MediaPipe Face Landmarker landmarks
        """

        if not landmarks:
            return None

        height, width = frame.shape[:2]

        # Find the face bounding box from normalized landmarks.
        xs = [int(point.x * width) for point in landmarks]
        ys = [int(point.y * height) for point in landmarks]

        x_min = max(0, min(xs))
        x_max = min(width, max(xs))
        y_min = max(0, min(ys))
        y_max = min(height, max(ys))

        face_width = x_max - x_min
        face_height = y_max - y_min

        if face_width < 20 or face_height < 20:
            return None

        # Approximate forehead and upper-cheek regions.
        # These are initial ROI estimates, not skin segmentation.
        regions = regions = [
        # Forehead
        (
            x_min + int(0.30 * face_width),
            y_min + int(0.10 * face_height),
            x_min + int(0.70 * face_width),
            y_min + int(0.23 * face_height)
        ),

        # Left cheek - moved lower
        (
            x_min + int(0.18 * face_width),
            y_min + int(0.60 * face_height),
            x_min + int(0.40 * face_width),
            y_min + int(0.76 * face_height)
        ),

        # Right cheek - moved lower
        (
            x_min + int(0.60 * face_width),
            y_min + int(0.60 * face_height),
            x_min + int(0.82 * face_width),
            y_min + int(0.76 * face_height)
        )
    ]

        means = []

        for x1, y1, x2, y2 in regions:
            x1 = max(0, min(width, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height, y1))
            y2 = max(0, min(height, y2))

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Draw the ROI rectangles for visual verification.
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )

            # OpenCV stores channels as BGR.
            mean_bgr = np.mean(roi, axis=(0, 1))
            means.append(mean_bgr)

        if not means:
            return None

        # Average the selected regions.
        blue, green, red = np.mean(means, axis=0)

        self.timestamps.append(time.time())
        self.red_values.append(float(red))
        self.green_values.append(float(green))
        self.blue_values.append(float(blue))

        return {
            "red": float(red),
            "green": float(green),
            "blue": float(blue),
            "samples": len(self.timestamps)
        }
