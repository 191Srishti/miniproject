import time


class BlinkDetector:

    def __init__(self, ear_threshold=0.20):
        self.ear_threshold = ear_threshold

        self.eye_closed = False
        self.close_start_time = None

        self.total_blinks = 0
        self.long_closures = 0

    def update(self, ear):

        current_time = time.time()

        # Eye is closing
        if ear < self.ear_threshold:

            if not self.eye_closed:
                self.eye_closed = True
                self.close_start_time = current_time

            closure_duration = current_time - self.close_start_time

        # Eye is open
        else:

            closure_duration = 0

            if self.eye_closed:

                self.total_blinks += 1

                # Consider > 1 second a prolonged closure
                previous_duration = (
                    current_time - self.close_start_time
                )

                if previous_duration > 1.0:
                    self.long_closures += 1

            self.eye_closed = False
            self.close_start_time = None

        return {
            "eye_closed": self.eye_closed,
            "closure_duration": closure_duration,
            "total_blinks": self.total_blinks,
            "long_closures": self.long_closures
        }