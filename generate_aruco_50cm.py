import cv2
import numpy as np
from PIL import Image

# ======================
# 参数设置
# ======================
DPI = 300

BOARD_CM = 18        # 整块板子 50cm
MARKER_CM = 10       # 中间 ArUco 码 40cm
MARKER_ID = 4        # 标定板 ID

# OpenCV ArUco 字典，后续检测代码也要用同一个
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# cm 转像素
px_per_cm = DPI / 2.54

board_px = int(BOARD_CM * px_per_cm)
marker_px = int(MARKER_CM * px_per_cm)

# 生成 ArUco marker
try:
    marker_img = cv2.aruco.generateImageMarker(
        aruco_dict,
        MARKER_ID,
        marker_px
    )
except AttributeError:
    marker_img = np.zeros((marker_px, marker_px), dtype=np.uint8)
    cv2.aruco.drawMarker(
        aruco_dict,
        MARKER_ID,
        marker_px,
        marker_img,
        1
    )

# 创建白色背景板
board = np.ones((board_px, board_px), dtype=np.uint8) * 255

# 居中放置 marker
start = (board_px - marker_px) // 2
board[start:start + marker_px, start:start + marker_px] = marker_img

# 保存图片
img = Image.fromarray(board)
img.save("MARKER_ID.png", dpi=(DPI, DPI))

print("MARKER_ID:", MARKER_ID)
print("MARKER_CM:", MARKER_CM)
print("整板尺寸：50cm x 50cm")
print("中间 ArUco 尺寸：40cm x 40cm")
print("程序中 MARKER_SIZE 应设置为 0.40")