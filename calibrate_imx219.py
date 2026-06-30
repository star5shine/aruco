# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import os

IMAGE_DIR = "/root/calib_images"

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.030

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp = objp * SQUARE_SIZE

objpoints = []
imgpoints = []

images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

if len(images) == 0:
    print("[ERR] No images found in", IMAGE_DIR)
    exit(1)

gray_shape = None
valid_count = 0

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print("[FAIL] Cannot read", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

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
    exit(1)

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
    "/root/imx219_calib.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rms=ret,
    mean_error=mean_error
)

print("")
print("[SAVE] /root/imx219_calib.npz")