# -*- coding: utf-8 -*-

import os
os.environ.setdefault("DISPLAY", ":0")

import cv2
import numpy as np
import math
import time
from collections import deque

DEVICE = "/dev/video31"

CAP_WIDTH = 1920
CAP_HEIGHT = 1080
FPS = 30

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# This must be the real black-white ArUco marker size, not the outer board size.
# If your ArUco marker is 40 cm, use 0.40.
# If your ArUco marker is 50 cm, use 0.50.
MARKER_SIZE = 0.40

CALIB_FILE = "/root/imx219_calib.npz"

calib = np.load(CALIB_FILE)
camera_matrix = calib["camera_matrix"]
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

print("[INFO] Loaded camera calibration:")
print("camera_matrix:")
print(camera_matrix)
print("dist_coeffs:")
print(dist_coeffs)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

try:
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    USE_NEW_ARUCO = True
except Exception:
    aruco_params = cv2.aruco.DetectorParameters_create()
    detector = None
    USE_NEW_ARUCO = False

half = MARKER_SIZE / 2.0

object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

gst_pipeline = (
    "v4l2src device={} ! "
    "video/x-raw,format=NV12,width={},height={},framerate={}/1 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink max-buffers=1 drop=true sync=false"
).format(DEVICE, CAP_WIDTH, CAP_HEIGHT, FPS)

print("[INFO] Pipeline:")
print(gst_pipeline)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("[ERR] Failed to open camera")
    exit(1)

print("[OK] Camera opened:", DEVICE)
print("[INFO] Press q or ESC to exit")

x_buf = deque(maxlen=10)
z_buf = deque(maxlen=10)
d_buf = deque(maxlen=10)
a_buf = deque(maxlen=10)

last_print_time = 0.0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERR] Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if USE_NEW_ARUCO:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params
            )

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                image_points = corners[i][0].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    x = float(tvec[0][0])
                    z = float(tvec[2][0])

                    distance_2d = math.sqrt(x * x + z * z)
                    angle_2d = math.degrees(math.atan2(x, z))

                    x_buf.append(x)
                    z_buf.append(z)
                    d_buf.append(distance_2d)
                    a_buf.append(angle_2d)

                    x_show = sum(x_buf) / len(x_buf)
                    z_show = sum(z_buf) / len(z_buf)
                    d_show = sum(d_buf) / len(d_buf)
                    a_show = sum(a_buf) / len(a_buf)

                    if a_show > 0:
                        direction = "Right"
                    elif a_show < 0:
                        direction = "Left"
                    else:
                        direction = "Center"

                    now = time.time()
                    if now - last_print_time > 0.3:
                        print(
                            "ID:{}  Z:{:.3f}m  X:{:.3f}m  D:{:.3f}m  Angle:{:.2f}deg  {}".format(
                                marker_id,
                                z_show,
                                x_show,
                                d_show,
                                a_show,
                                direction
                            )
                        )
                        last_print_time = now

                    p = image_points[0].astype(int)

                    text1 = "ID:{}  Z:{:.3f}m  X:{:.3f}m".format(
                        marker_id, z_show, x_show
                    )
                    text2 = "D:{:.3f}m  Angle:{:.2f}deg  {}".format(
                        d_show, a_show, direction
                    )

                    cv2.putText(
                        frame,
                        text1,
                        (p[0], max(40, p[1] - 50)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        text2,
                        (p[0], max(80, p[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )

                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        MARKER_SIZE * 0.5
                    )

        show_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("IMX219 ArUco Calibrated 2D", show_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exit")