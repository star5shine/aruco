# -*- coding: utf-8 -*-

import argparse
import glob
import os

import cv2
import numpy as np


APP_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_IMAGE_DIR = os.path.join(APP_DIR, "calib_images")
DEFAULT_OUTPUT_FILE = os.path.join(APP_DIR, "netcamera_calib.npz")

# Keep these the same as aruco_netcamera.py DEFAULT_WIDTH/DEFAULT_HEIGHT.
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

CHESSBOARD_SIZE = (10, 6)
SQUARE_SIZE = 0.030


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate camera from chessboard images.")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--cols", type=int, default=CHESSBOARD_SIZE[0])
    parser.add_argument("--rows", type=int, default=CHESSBOARD_SIZE[1])
    parser.add_argument("--square-size", type=float, default=SQUARE_SIZE)
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT)
    return parser.parse_args()


def main():
    args = parse_args()

    chessboard_size = (args.cols, args.rows)
    target_size = (args.target_width, args.target_height)

    print("[INFO] Image dir:", args.image_dir)
    print("[INFO] Output:", args.output)
    print("[INFO] Calibration image size: {}x{}".format(args.target_width, args.target_height))
    print("[INFO] Chessboard inner corners: {}x{}".format(args.cols, args.rows))
    print("[INFO] Square size:", args.square_size)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:chessboard_size[0],
        0:chessboard_size[1]
    ].T.reshape(-1, 2)
    objp = objp * args.square_size

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))

    if len(images) == 0:
        print("[ERR] No images found in", args.image_dir)
        return 1

    gray_shape = target_size
    valid_count = 0

    for fname in images:
        img = cv2.imread(fname)

        if img is None:
            print("[FAIL] Cannot read", fname)
            continue

        src_h, src_w = img.shape[:2]
        if (src_w, src_h) != target_size:
            print(
                "[WARN] Resize {} from {}x{} to {}x{}".format(
                    fname,
                    src_w,
                    src_h,
                    args.target_width,
                    args.target_height,
                )
            )
            img = cv2.resize(img, target_size)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            objpoints.append(objp)
            imgpoints.append(corners2)
            valid_count += 1
            print("[OK]", fname)
        else:
            print("[FAIL]", fname)

    print("[INFO] Valid images:", valid_count)

    if valid_count < 10:
        print("[ERR] Too few valid images. Need at least 10, better 20+.")
        return 1

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray_shape,
        None,
        None
    )

    print("")
    print("========== Calibration Result ==========")
    print("RMS error:")
    print(ret)

    print("")
    print("camera_matrix:")
    print(camera_matrix)

    print("")
    print("dist_coeffs:")
    print(dist_coeffs)

    mean_error = 0.0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error = mean_error / len(objpoints)

    print("")
    print("mean reprojection error:")
    print(mean_error)

    np.savez(
        args.output,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=ret,
        mean_error=mean_error,
        image_width=args.target_width,
        image_height=args.target_height,
    )

    print("")
    print("[SAVE]", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
