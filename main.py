import cv2
import time
import numpy as np

from rppg.data_logger import MonitoringLogger

from collections import deque

from rppg.visualization import draw_waveform
import mediapipe as mp

from rppg.roi_signal import RPPGSignalCollector

rppg_collector = RPPGSignalCollector()

from rppg.signal_processor import process_green_signal

from rppg.heart_rate import estimate_heart_rate

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fatigue.ear import calculate_ear
from fatigue.blink import BlinkDetector

from alert_engine import AlertEngine


import ctypes

screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)





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
# DASHBOARD DISPLAY FUNCTION
# --------------------------------------------------


def build_dashboard(
    frame, bpm, ear, eye_status, total_blinks,
    long_closures, perclos, signal_data,
    alert_messages, monitoring_status
):

    W, H = 1366, 768

    dashboard = np.zeros((H, W, 3), dtype=np.uint8)
    dashboard[:] = (245, 247, 250)

    margin = 18
    gap = 14

    # ======================================
    # HEADER
    # ======================================

    cv2.putText(
        dashboard,
        "CONTACTLESS REAL-TIME HEALTH & FATIGUE MONITORING SYSTEM USING AI & WEBCAM",
        (margin, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (35, 45, 60),
        2
    )

    # cv2.putText(
    #     dashboard,
    #     "Using AI & Webcam",
    #     (margin, 57),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.38,
    #     (110, 120, 135),
    #     1
    # )

    # ======================================
    # MAIN LAYOUT
    # ======================================

    left_x = margin
    top_y = 90

    left_w = 810
    right_x = left_x + left_w + gap
    right_w = W - right_x - margin

    content_h = 440

    # ======================================
    # 1. WEBCAM PREVIEW
    # ======================================

    cv2.putText(
        dashboard,
        "LIVE WEBCAM",
        (left_x, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (35, 45, 60),
        1
    )

    frame_h, frame_w = frame.shape[:2]

    scale = min(left_w / frame_w, content_h / frame_h)

    new_w = max(1, int(frame_w * scale))
    new_h = max(1, int(frame_h * scale))

    preview = cv2.resize(frame, (new_w, new_h))

    px = left_x + (left_w - new_w) // 2
    py = top_y + (content_h - new_h) // 2

    dashboard[py:py + new_h, px:px + new_w] = preview

    cv2.rectangle(
        dashboard,
        (left_x, top_y),
        (left_x + left_w, top_y + content_h),
        (210, 215, 225),
        1
    )

    # ======================================
    # 2. HEALTH METRICS
    # ======================================

    cv2.putText(
        dashboard,
        "HEALTH METRICS",
        (right_x, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (35, 45, 60),
        1
    )

    metrics = [
        ("HEART RATE", f"{bpm:.1f} BPM" if bpm is not None else "--"),
        ("EAR", f"{ear:.3f}" if ear is not None else "--"),
        ("EYE STATUS", str(eye_status)),
        ("TOTAL BLINKS", str(total_blinks)),
        ("LONG CLOSURES", str(long_closures)),
        ("PERCLOS", f"{perclos * 100:.1f}%" if perclos is not None else "--")
    ]

    card_gap = 9
    card_w = (right_w - card_gap) // 2
    card_h = 137

    for i, (label, value) in enumerate(metrics):

        col = i % 2
        row = i // 2

        x = right_x + col * (card_w + card_gap)
        y = top_y + row * (card_h + card_gap)

        cv2.rectangle(
            dashboard,
            (x, y),
            (x + card_w, y + card_h),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            dashboard,
            label,
            (x + 10, y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (105, 115, 130),
            1
        )

        cv2.putText(
            dashboard,
            value[:18],
            (x + 10, y + 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (45, 155, 95),
            1
        )

    # ======================================
    # 3. BOTTOM WAVEFORM PANEL
    # ======================================

    bottom_y = 555

    waveform_signal = (
        signal_data.get("signal")
        if isinstance(signal_data, dict)
        else signal_data
    )

    draw_waveform(
        dashboard,
        waveform_signal,
        x=margin,
        y=620,
        width=810,
        height=90
    )

    # ======================================
    # 4. ALERT PANEL
    # ======================================


    # ======================================
    # 4. ALERT PANEL
    # ======================================

    alert_x = right_x

    cv2.putText(
        dashboard,
        "SYSTEM ALERTS",
        (alert_x, 575),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (35, 45, 60),
        1
    )

    # White alert panel
    cv2.rectangle(
        dashboard,
        (alert_x, 590),
        (W - margin, 710),
        (255, 255, 255),
        -1
    )

    cv2.rectangle(
        dashboard,
        (alert_x, 590),
        (W - margin, 710),
        (220, 225, 232),
        1
    )

    # Show break warning after 3 long closures
    if long_closures >= 3:

        cv2.putText(
            dashboard,
            "FATIGUE DETECTED!",
            (alert_x + 15, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 200),
            2
        )

        cv2.putText(
            dashboard,
            "3 LONG EYE CLOSURES DETECTED",
            (alert_x + 15, 655),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 180),
            1
        )

        cv2.putText(
            dashboard,
            "PLEASE TAKE A SHORT BREAK!",
            (alert_x + 15, 685),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 200),
            2
        )

    else:

        messages = (
            alert_messages[:3]
            if alert_messages
            else ["NO ACTIVE ALERTS"]
        )

        for i, message in enumerate(messages):

            alert_color = (
                (0, 0, 180) if alert_messages
                else (45, 155, 95)
            )

            cv2.putText(
                dashboard,
                str(message)[:38],
                (alert_x + 12, 620 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                alert_color,
                1
            )


        
    # ======================================
    # 5. FOOTER / MONITORING STATUS
    # ======================================

    cv2.line(
        dashboard,
        (margin, 735),
        (W - margin, 735),
        (80, 95, 110),
        1
    )

    cv2.putText(
        dashboard,
        f"MONITORING STATUS: {monitoring_status}",
        (margin, 758),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (100, 230, 140),
        1
    )

    return dashboard





import ctypes

screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)

cv2.namedWindow(
    "Health Monitoring Dashboard",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Health Monitoring Dashboard",
    1366,
    768
)


# --------------------------------------------------
# Main loop
# --------------------------------------------------

monitor_logger = MonitoringLogger()

# Initialize monitoring state ONCE
bpm_history = deque(maxlen=5)

stable_bpm = None
signal_data = None
hr_data = None

last_hr_update = 0.0

# Initialize alert engine
# from alert_engine import AlertEngine

alert_engine = AlertEngine(
    heart_rate_limits=None
)

# Main loop
last_valid_bpm_time = None
total_blinks = 0
long_closures = 0
eye_status = "NO FACE"
# Initialize before the while loop
long_closures = 0
closure_start_time = None
long_closure_counted = False



while True:




    # Heart-rate stabilization state
    # bpm_history = deque(maxlen=5)
    # stable_bpm = None

    # signal_data = None
    # last_hr_update = 0.0

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
    # Collect RGB values over time
    # --------------------------------------------------

    rgb_data = None
    ear = None
    eye_closed = None
    closure_duration = 0.0
    eye_status = "NO FACE"

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        rgb_data = rppg_collector.process_frame(
            frame, landmarks
        )

    if rgb_data is not None:
        cv2.putText(
            frame,
            f"RGB: R={rgb_data['red']:.1f} "
            f"G={rgb_data['green']:.1f} "
            f"B={rgb_data['blue']:.1f}",
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"RGB Samples: {rgb_data['samples']}",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )




    
    

    
    # --------------------------------------------------
    # Periodic rPPG processing and BPM stabilization
    # --------------------------------------------------

    
    # --------------------------------------------------
    # Periodic rPPG processing and BPM stabilization
    # --------------------------------------------------

    current_time = time.time()

    if current_time - last_hr_update >= 1.0:

        last_hr_update = current_time

        # Reset before every processing update
        hr_data = None

        signal_data = process_green_signal(
            list(rppg_collector.timestamps),
            list(rppg_collector.green_values)
        )

        if signal_data is not None:

            hr_data = estimate_heart_rate(
                signal_data["signal"],
                signal_data["sampling_rate"],
                signal_data["duration"]
            )

        # Accept and stabilize valid BPM
        if hr_data is not None:

            new_bpm = hr_data["bpm"]

            if (
                not bpm_history
                or abs(new_bpm - float(np.median(bpm_history))) <= 20
            ):

                bpm_history.append(new_bpm)

                stable_bpm = round(
                    float(np.median(bpm_history))
                )

                last_valid_bpm_time = current_time

        # --------------------------------------------------
        # Reject stale BPM readings
        # --------------------------------------------------

        if (
            last_valid_bpm_time is None
            or current_time - last_valid_bpm_time > 5
        ):

            stable_bpm = None
            bpm_history.clear()

        # --------------------------------------------------
        # CSV logging: one record per update
        # --------------------------------------------------

        red = (
            rppg_collector.red_values[-1]
            if rppg_collector.red_values else None
        )

        green = (
            rppg_collector.green_values[-1]
            if rppg_collector.green_values else None
        )

        blue = (
            rppg_collector.blue_values[-1]
            if rppg_collector.blue_values else None
        )

        monitor_logger.log_reading(
            red=red,
            green=green,
            blue=blue,

            estimated_bpm=(
                round(hr_data["bpm"], 2)
                if hr_data is not None else None
            ),

            stable_bpm=stable_bpm,

            peak_ratio=(
                round(hr_data["peak_ratio"], 2)
                if hr_data is not None else None
            ),

            sampling_rate=(
                round(signal_data["sampling_rate"], 2)
                if signal_data is not None else None
            ),

            signal_duration=(
                round(signal_data["duration"], 2)
                if signal_data is not None else None
            ),

            signal_status=(
                "Processed"
                if signal_data is not None
                else "Unavailable"
            )
        )


    # --------------------------------------------------
    # Display stabilized BPM
    # --------------------------------------------------

    if stable_bpm is not None:
        bpm_text = f"Stable Estimated HR: {stable_bpm} BPM"
        bpm_color = (0, 255, 0)
    else:
        bpm_text = "Heart Rate: Waiting for usable signal"
        bpm_color = (0, 255, 255)

    cv2.putText(
        frame,
        bpm_text,
        (30, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        bpm_color,
        2
    )

    # --------------------------------------------------
    # Draw filtered rPPG waveform
    # --------------------------------------------------

    waveform = (
        signal_data["signal"]
        if signal_data is not None
        else None
    )

    






    # -----------------------------------------
    # Initialize EAR variables for each frame
    # -----------------------------------------

    ear = None
    eye_closed = None
    closure_duration = 0.0






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



        face_detected = bool(result.face_landmarks)

        signal_valid = (
            face_detected
            and signal_data is not None
            and stable_bpm is not None
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

        



    # -----------------------------------------
    # ALERT ENGINE INTEGRATION
    # -----------------------------------------

    face_detected = bool(result.face_landmarks)

    # A valid BPM must come from a usable signal.
    signal_valid = (
        face_detected
        and signal_data is not None
        and stable_bpm is not None
    )

    alert_result = alert_engine.update(

        face_detected=face_detected,

        signal_valid=signal_valid,

        bpm=stable_bpm if signal_valid else None,

        eye_closed=eye_closed,

        closure_duration=closure_duration
    )


    # # Show webcam
    
    
    # -----------------------------------------
    # BUILD DASHBOARD
    # -----------------------------------------

    dashboard_frame = build_dashboard(
        frame=frame,
        bpm=stable_bpm,
        ear=ear,
        eye_status=eye_status,
        total_blinks=total_blinks,
        long_closures=long_closures,
        perclos=alert_result["perclos"],
        signal_data=signal_data,
        alert_messages=alert_result["messages"],
        monitoring_status=alert_result["status"]
    )
    cv2.imshow("Health Monitoring Dashboard", dashboard_frame)
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('t'):
        long_closures = 3

    elif key == ord('r'):
        long_closures = 0

    elif key == ord('q'):
        break

    

    
    


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
detector.close()
monitor_logger.close()
cv2.destroyAllWindows()