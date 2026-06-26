# -*- coding: utf-8 -*-

import os
os.environ.setdefault("DISPLAY", ":0")

import cv2
import time

DEVICE = "/dev/video31"
WIDTH = 1920
HEIGHT = 1080
FPS = 30

SAVE_DIR = "/root/calib_images"
os.makedirs(SAVE_DIR, exist_ok=True)

gst_pipeline = (
    "v4l2src device={} ! "
    "video/x-raw,format=NV12,width={},height={},framerate={}/1 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink max-buffers=1 drop=true sync=false"
).format(DEVICE, WIDTH, HEIGHT, FPS)

print("[INFO] Pipeline:")
print(gst_pipeline)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("[ERR] Failed to open camera")
    exit(1)

print("[OK] Camera opened")
print("[INFO] Press s to save image")
print("[INFO] Press q or ESC to exit")

idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERR] Failed to read frame")
        break

    show = cv2.resize(frame, (960, 540))
    cv2.imshow("Calibration Capture", show)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = "{}/calib_{:03d}.jpg".format(SAVE_DIR, idx)
        cv2.imwrite(filename, frame)
        print("[SAVE]", filename)
        idx += 1

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()