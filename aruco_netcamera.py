#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTSP ArUco location viewer with TCP JSON output.

Edit the DEFAULT_* values below, then run:
  python3 aruco.py

TCP packet format, one JSON object per line:
  {"type":"QR_PARKING","seq":1,"timestamp_ms":1710000000000,"valid":true,"confidence":1.0,"right_m":0.02,"forward_m":1.25,"yaw_error_deg":-1.8}

Quit: press q or ESC.
"""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np


MARKER_SIZE = 0.40
TCP_SEND_INTERVAL = 0.1
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# Change these values on the RK3588, then run: python3 aruco.py
DEFAULT_RTSP_URL = "rtsp://admin:sua07f18@192.168.1.11:554/channel=1&stream=1.sdp"
DEFAULT_RTSP_TRANSPORT = "tcp"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
#DEFAULT_CALIB_FILE = "/root/imx219_calib.npz"
DEFAULT_CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "netcamera_calib.npz")
DEFAULT_IGNORE_DIST = True

# Leave DEFAULT_TCP_HOST empty to disable TCP output.
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 10010
DEFAULT_TCP_RECONNECT = True


def mask_url(url):
    try:
        parts = urlsplit(url)
        if "@" not in parts.netloc:
            return url
        userinfo, host = parts.netloc.rsplit("@", 1)
        if ":" in userinfo:
            user = userinfo.split(":", 1)[0]
            netloc = "{}:******@{}".format(user, host)
        else:
            netloc = "{}@{}".format(userinfo, host)
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return url


def run_cmd(cmd, timeout=10):
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def ffprobe_stream(url, transport="tcp"):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-rtsp_transport", transport,
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
        "-of", "json",
        url,
    ]
    try:
        proc = run_cmd(cmd, timeout=12)
        if proc.returncode != 0:
            return None, proc.stderr.strip()
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        if not streams:
            return None, "No video stream found"
        return streams[0], ""
    except Exception as exc:
        return None, str(exc)


class FrameStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.count = 0
        self.last_update = 0.0
        self.stopped = False

    def put(self, frame):
        with self.lock:
            self.frame = frame
            self.count += 1
            self.last_update = time.time()

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, self.count, self.last_update
            return self.frame.copy(), self.count, self.last_update


class TcpJsonSender:
    def __init__(self, host, port, reconnect):
        self.host = host
        self.port = port
        self.reconnect = reconnect
        self.sock = None
        self.last_try = 0.0

    def enabled(self):
        return bool(self.host) and self.port > 0

    def connect(self):
        if not self.enabled():
            return False

        now = time.time()
        if self.sock is None and now - self.last_try < 2.0:
            return False

        self.last_try = now
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((self.host, self.port))
            sock.settimeout(0.5)
            self.sock = sock
            print("[INFO] TCP connected to {}:{}".format(self.host, self.port))
            return True
        except OSError as exc:
            self.sock = None
            print("[WARN] TCP connect failed: {}".format(exc))
            return False

    def send(self, packet):
        if not self.enabled():
            return

        if self.sock is None and not self.connect():
            return

        try:
            line = json.dumps(packet, separators=(",", ":")) + "\n"
            self.sock.sendall(line.encode("utf-8"))
        except OSError as exc:
            print("[WARN] TCP send failed: {}".format(exc))
            self.close()
            if self.reconnect:
                self.connect()

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None


def build_ffmpeg_cmd(args, width, height):
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", args.ffmpeg_loglevel,
        "-rtsp_transport", args.transport,
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-avioflags", "direct",
        "-max_delay", "0",
        "-probesize", str(args.probesize),
        "-analyzeduration", str(args.analyzeduration),
    ]

    if args.decoder != "sw":
        cmd += ["-c:v", args.decoder]

    cmd += [
        "-i", args.url,
        "-an",
        "-sn",
        "-dn",
    ]

    if width > 0 and height > 0:
        cmd += ["-vf", "scale={}:{}".format(width, height)]

    cmd += [
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-",
    ]
    return cmd


def reader_loop(proc, store, width, height):
    frame_size = width * height * 3
    fail_count = 0

    while not store.stopped:
        raw = proc.stdout.read(frame_size)
        if len(raw) == frame_size:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            store.put(frame)
            fail_count = 0
            continue

        fail_count += 1
        if store.stopped or fail_count >= 5:
            break
        time.sleep(0.02)


def stderr_loop(proc, store):
    while not store.stopped:
        line = proc.stderr.readline()
        if not line:
            break
        text = line.decode(errors="ignore").strip()
        if text:
            print("[FFMPEG]", text)


def load_calibration(path, ignore_dist):
    calib = np.load(path)
    camera_matrix = calib["camera_matrix"]

    if ignore_dist:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    elif "dist_coeffs" in calib:
        dist_coeffs = calib["dist_coeffs"]
    elif "dist" in calib:
        dist_coeffs = calib["dist"]
    else:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    return camera_matrix, dist_coeffs


def create_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    try:
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        return aruco_dict, detector, True
    except Exception:
        aruco_params = cv2.aruco.DetectorParameters_create()
        return aruco_dict, aruco_params, False


def build_qr_parking_packet(seq, timestamp, right_m, forward_m, yaw_error_deg, confidence=1.0):
    return {
        "type": "QR_PARKING",
        "seq": seq,
        "timestamp_ms": int(timestamp * 1000),
        "valid": True,
        "confidence": round(float(confidence), 3),
        "right_m": round(float(right_m), 3),
        "forward_m": round(float(forward_m), 3),
        "yaw_error_deg": round(float(yaw_error_deg), 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="RTSP ArUco location viewer with TCP JSON output.")
    parser.add_argument("--url", default=DEFAULT_RTSP_URL, help="Full RTSP URL. Use quotes if it contains &.")
    parser.add_argument("--calib-file", default=DEFAULT_CALIB_FILE)
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE)
    parser.add_argument("--transport", choices=["tcp", "udp"], default=DEFAULT_RTSP_TRANSPORT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--display-width", type=int, default=DISPLAY_WIDTH)
    parser.add_argument("--display-height", type=int, default=DISPLAY_HEIGHT)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--decoder", choices=["sw", "hevc_rkmpp", "h264_rkmpp"], default="sw")
    parser.add_argument("--probesize", type=int, default=32768)
    parser.add_argument("--analyzeduration", type=int, default=0)
    parser.add_argument("--ffmpeg-loglevel", choices=["quiet", "error", "warning", "info"], default="warning")
    parser.add_argument("--no-probe", action="store_true", help="Skip ffprobe.")
    parser.add_argument("--ignore-dist", action="store_true", default=DEFAULT_IGNORE_DIST,
                        help="Use zero distortion coefficients.")
    parser.add_argument("--use-dist", action="store_false", dest="ignore_dist",
                        help="Use distortion coefficients from calibration file.")
    parser.add_argument("--tcp-host", default=DEFAULT_TCP_HOST, help="TCP receiver IP. Empty means no TCP output.")
    parser.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT, help="TCP receiver port.")
    parser.add_argument("--tcp-reconnect", action="store_true", default=DEFAULT_TCP_RECONNECT,
                        help="Reconnect if TCP send fails.")
    parser.add_argument("--no-tcp-reconnect", action="store_false", dest="tcp_reconnect",
                        help="Disable TCP reconnect.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("DISPLAY", ":0")

    if "YOUR_PASSWORD" in args.url:
        print("[ERR] Please edit DEFAULT_RTSP_URL in aruco.py and replace YOUR_PASSWORD.")
        print("[ERR] Current URL:", args.url)
        return 1

    camera_matrix, dist_coeffs = load_calibration(args.calib_file, args.ignore_dist)
    print("[INFO] Loaded camera calibration:", args.calib_file)
    print("camera_matrix:")
    print(camera_matrix)
    print("dist_coeffs:")
    print(dist_coeffs)

    if not args.no_probe:
        info, err = ffprobe_stream(args.url, args.transport)
        if info:
            codec = info.get("codec_name", "unknown")
            src_w = int(info.get("width") or 0)
            src_h = int(info.get("height") or 0)
            rate = info.get("avg_frame_rate") or info.get("r_frame_rate") or "unknown"
            print("[INFO] stream: codec={}, source={}x{}, fps={}".format(codec, src_w, src_h, rate))
        else:
            print("[WARN] ffprobe failed:", err)

    print("[INFO] URL:", mask_url(args.url))
    print("[INFO] output size: {}x{}".format(args.width, args.height))
    print("[INFO] RTSP transport:", args.transport)
    print("[INFO] decoder:", args.decoder)
    print("[INFO] DISPLAY:", os.environ.get("DISPLAY"))
    print("[INFO] Press q or ESC to exit")

    tcp_sender = TcpJsonSender(args.tcp_host, args.tcp_port, args.tcp_reconnect)
    if tcp_sender.enabled():
        tcp_sender.connect()
    else:
        print("[INFO] TCP output disabled. Add --tcp-host and --tcp-port to enable it.")

    aruco_dict, detector_or_params, use_new_aruco = create_aruco_detector()

    half = args.marker_size / 2.0
    object_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)

    store = FrameStore()
    cmd = build_ffmpeg_cmd(args, args.width, args.height)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    reader = threading.Thread(target=reader_loop, args=(proc, store, args.width, args.height), daemon=True)
    reader.start()

    err_reader = threading.Thread(target=stderr_loop, args=(proc, store), daemon=True)
    err_reader.start()

    window_name = "RTSP ArUco TCP"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, args.display_width, args.display_height)

    stop_flag = {"stop": False}

    def handle_stop(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    x_buf = deque(maxlen=10)
    z_buf = deque(maxlen=10)
    d_buf = deque(maxlen=10)
    a_buf = deque(maxlen=10)
    last_print_time = 0.0
    packet_seq = 1

    try:
        while not stop_flag["stop"]:
            frame, count, updated = store.get()

            if frame is None:
                if proc.poll() is not None:
                    print("[ERR] FFmpeg exited before producing frames.")
                    break
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if use_new_aruco:
                corners, ids, rejected = detector_or_params.detectMarkers(gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    gray,
                    aruco_dict,
                    parameters=detector_or_params,
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
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )

                    if not success:
                        continue

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
                    if now - last_print_time >= TCP_SEND_INTERVAL:
                        packet = build_qr_parking_packet(
                            packet_seq,
                            now,
                            x_show,
                            z_show,
                            a_show,
                        )
                        print(
                            "ID:{}  Z:{:.3f}m  X:{:.3f}m  D:{:.3f}m  Angle:{:.2f}deg  {}".format(
                                marker_id,
                                z_show,
                                x_show,
                                d_show,
                                a_show,
                                direction,
                            )
                        )
                        tcp_sender.send(packet)
                        packet_seq += 1
                        last_print_time = now

                    p = image_points[0].astype(int)
                    text1 = "ID:{}  Z:{:.3f}m  X:{:.3f}m".format(marker_id, z_show, x_show)
                    text2 = "D:{:.3f}m  Angle:{:.2f}deg  {}".format(d_show, a_show, direction)

                    cv2.putText(
                        frame,
                        text1,
                        (p[0], max(40, p[1] - 50)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        text2,
                        (p[0], max(80, p[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        args.marker_size * 0.5,
                    )

            show_frame = cv2.resize(frame, (args.display_width, args.display_height))
            cv2.imshow(window_name, show_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if updated > 0 and time.time() - updated > 5:
                print("[WARN] no new frames for 5 seconds.")
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted")
    finally:
        store.stopped = True
        tcp_sender.close()
        try:
            proc.terminate()
            proc.wait(timeout=1.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("[INFO] Exit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


