import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fatigue.ear import calculate_ear
from fatigue.blink import BlinkDetector


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "face_landmarker.task"

# MediaPipe Face Mesh landmark indices
# These correspond to the left and right eye.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Initial eye-closure threshold.
# We will calibrate this later.
EAR_THRESHOLD = 0.20

blink_detector = BlinkDetector(EAR_THRESHOLD)

# --------------------------------------------------
# Create Face Landmarker
# --------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------
# Start webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not access webcam.")
    exit()

print("EAR monitoring started!")
print("Press Q to quit.")

timestamp_ms = 0


# --------------------------------------------------
# Main loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face landmarks
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    timestamp_ms += 33

    # --------------------------------------------------
    # Calculate EAR
    # --------------------------------------------------

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        # Average both eyes
        ear = (left_ear + right_ear) / 2.0

        blink_data = blink_detector.update(ear)
        eye_closed = blink_data["eye_closed"]
        closure_duration = blink_data["closure_duration"]
        total_blinks = blink_data["total_blinks"]
        long_closures = blink_data["long_closures"]

        # Determine eye status
        if ear < EAR_THRESHOLD:
            eye_status = "CLOSED"
        else:
            eye_status = "OPEN"

        # --------------------------------------------------
        # Display information
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Closure: {closure_duration:.2f}s",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Blinks: {total_blinks}",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Long Closures: {long_closures}",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # --------------------------------------------------
        # Draw eye landmarks
        # --------------------------------------------------

        eye_points = LEFT_EYE + RIGHT_EYE

        for index in eye_points:

            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 255),
                -1
            )

    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Show webcam
    cv2.imshow(
        "Contactless Health Monitor - EAR",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
detector.close()
cv2.destroyAllWindows()