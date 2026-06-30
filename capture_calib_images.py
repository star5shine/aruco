# -*- coding: utf-8 -*-

import argparse
import glob
import os

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;3000000"
)

import cv2


APP_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RTSP_URL = "rtsp://admin:sua07f18@192.168.1.11:554/channel=1&stream=1.sdp"
DEFAULT_SAVE_DIR = os.path.join(APP_DIR, "calib_images")

# Keep these the same as aruco_netcamera.py DEFAULT_WIDTH/DEFAULT_HEIGHT.
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

DEFAULT_DISPLAY_WIDTH = 1280
DEFAULT_DISPLAY_HEIGHT = 720


def parse_args():
    parser = argparse.ArgumentParser(description="Capture calibration images from RTSP camera.")
    parser.add_argument("--url", default=DEFAULT_RTSP_URL, help="RTSP camera URL.")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Directory to save calibration images.")
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH, help="Saved image width.")
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT, help="Saved image height.")
    parser.add_argument("--display-width", type=int, default=DEFAULT_DISPLAY_WIDTH)
    parser.add_argument("--display-height", type=int, default=DEFAULT_DISPLAY_HEIGHT)
    return parser.parse_args()


def next_image_index(save_dir):
    images = glob.glob(os.path.join(save_dir, "calib_*.jpg"))
    max_idx = -1

    for path in images:
        name = os.path.basename(path)
        stem, _ = os.path.splitext(name)
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        max_idx = max(max_idx, idx)

    return max_idx + 1


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("[INFO] RTSP URL:", args.url)
    print("[INFO] Save dir:", args.save_dir)
    print("[INFO] Save size: {}x{}".format(args.target_width, args.target_height))
    print("[INFO] DISPLAY:", os.environ.get("DISPLAY"))
    print("[INFO] OpenCV FFmpeg options:", os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"))

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[ERR] Failed to open RTSP camera")
        return 1

    print("[OK] Camera opened")
    print("[INFO] Press s to save image")
    print("[INFO] Press q or ESC to exit")

    idx = next_image_index(args.save_dir)
    printed_frame_size = False

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[ERR] Failed to read frame")
            break

        if not printed_frame_size:
            print("[INFO] Source frame size: {}x{}".format(frame.shape[1], frame.shape[0]))
            printed_frame_size = True

        save_frame = cv2.resize(frame, (args.target_width, args.target_height))
        show = cv2.resize(save_frame, (args.display_width, args.display_height))
        cv2.imshow("Calibration Capture", show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            filename = os.path.join(args.save_dir, "calib_{:03d}.jpg".format(idx))
            cv2.imwrite(filename, save_frame)
            print("[SAVE]", filename, "{}x{}".format(args.target_width, args.target_height))
            idx += 1

        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
