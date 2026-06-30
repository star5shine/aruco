#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tkinter desktop UI with embedded RTSP ArUco video.

Run on RK3588:
  cd /root/aruco
  python3 aruco_desktop_ui.py
"""

import json
import math
import os
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;3000000"
)

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk


APP_DIR = Path(__file__).resolve().parent
CONFIG_FILE = APP_DIR / "aruco_desktop_ui_config.json"
DEFAULT_CALIB_FILE = APP_DIR / "netcamera_calib.npz"

MARKER_SIZE = 0.40
UI_PREVIEW_WIDTH = 800
UI_PREVIEW_HEIGHT = 450
UI_REFRESH_MS = 30
TCP_SEND_INTERVAL = 0.1
DEFAULTS = {
    "camera_ip": "192.168.1.11",
    "rtsp_user": "admin",
    "rtsp_password": "sua07f18",
    "rtsp_port": "554",
    "rtsp_path": "/channel=1&stream=1.sdp",
    "tcp_host": "127.0.0.1",
    "tcp_port": "10010",
    "width": "800",
    "height": "450",
    "display_width": str(UI_PREVIEW_WIDTH),
    "display_height": str(UI_PREVIEW_HEIGHT),
}


def parse_port(value, name="port"):
    try:
        port = int(str(value).strip())
    except ValueError as exc:
        raise ValueError("{} must be a number".format(name)) from exc
    if port < 1 or port > 65535:
        raise ValueError("{} must be between 1 and 65535".format(name))
    return port


def parse_positive_int(value, name):
    try:
        number = int(str(value).strip())
    except ValueError as exc:
        raise ValueError("{} must be a number".format(name)) from exc
    if number <= 0:
        raise ValueError("{} must be greater than 0".format(name))
    return number


def build_rtsp_url(camera_ip, user, password, port, path):
    camera_ip = str(camera_ip).strip()
    user = str(user).strip()
    password = str(password).strip()
    port = parse_port(port, "rtsp_port")
    path = str(path).strip() or DEFAULTS["rtsp_path"]
    if not path.startswith("/"):
        path = "/" + path
    return "rtsp://{}:{}@{}:{}{}".format(
        quote(user, safe=""),
        quote(password, safe=""),
        camera_ip,
        port,
        path,
    )


def build_qr_parking_packet(seq, timestamp, right_m, forward_m, yaw_error_deg, confidence=1.0):
    return {
        "type": "QR_PARKING",
        "seq": int(seq),
        "timestamp_ms": int(timestamp * 1000),
        "valid": True,
        "confidence": round(float(confidence), 3),
        "right_m": round(float(right_m), 3),
        "forward_m": round(float(forward_m), 3),
        "yaw_error_deg": round(float(yaw_error_deg), 2),
    }


def frame_to_ppm(rgb_frame):
    height, width = rgb_frame.shape[:2]
    header = "P6\n{} {}\n255\n".format(width, height).encode("ascii")
    return header + rgb_frame.tobytes()


def build_ffmpeg_cmd(url, width, height):
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-rtsp_transport", "tcp",
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-avioflags", "direct",
        "-max_delay", "0",
        "-probesize", "32768",
        "-analyzeduration", "0",
        "-i", url,
        "-an",
        "-sn",
        "-dn",
        "-vf", "scale={}:{}".format(width, height),
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-",
    ]


def read_raw_frame(proc, width, height):
    frame_size = width * height * 3
    raw = proc.stdout.read(frame_size)
    if len(raw) != frame_size:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))


def load_config():
    config = DEFAULTS.copy()
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            for key in config:
                if key in data and data[key] is not None:
                    config[key] = str(data[key])
        except (OSError, json.JSONDecodeError):
            pass
    config["display_width"] = str(UI_PREVIEW_WIDTH)
    config["display_height"] = str(UI_PREVIEW_HEIGHT)
    if (
        (config.get("width") == "1280" and config.get("height") == "720")
        or (config.get("width") == "800" and config.get("height") == "448")
    ):
        config["width"] = DEFAULTS["width"]
        config["height"] = DEFAULTS["height"]
    return config


def save_config(config):
    clean = DEFAULTS.copy()
    for key in clean:
        clean[key] = str(config.get(key, clean[key])).strip()
    CONFIG_FILE.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def load_calibration(path):
    calib = np.load(str(path))
    camera_matrix = calib["camera_matrix"]
    if "dist_coeffs" in calib:
        dist_coeffs = calib["dist_coeffs"]
    elif "dist" in calib:
        dist_coeffs = calib["dist"]
    else:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def create_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return aruco_dict, detector, True
    except Exception:
        params = cv2.aruco.DetectorParameters_create()
        return aruco_dict, params, False


class TcpJsonSender:
    def __init__(self, host, port):
        self.host = host.strip()
        self.port = int(port) if str(port).strip() else 0
        self.sock = None
        self.last_try = 0.0

    def enabled(self):
        return bool(self.host) and self.port > 0

    def connect(self):
        if not self.enabled():
            return False
        now = time.time()
        if self.sock is None and now - self.last_try < 1.0:
            return False
        self.last_try = now
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.3)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((self.host, self.port))
            sock.settimeout(0.2)
            self.sock = sock
            return True
        except OSError:
            self.sock = None
            return False

    def send(self, packet):
        if not self.enabled():
            return
        if self.sock is None and not self.connect():
            return
        try:
            line = json.dumps(packet, separators=(",", ":")) + "\n"
            self.sock.sendall(line.encode("utf-8"))
        except OSError:
            self.close()

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None


class RecognitionWorker:
    def __init__(self, config, on_status):
        self.config = config
        self.on_status = on_status
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.latest_ppm = None
        self.latest_text = "No result"
        self.latest_frame_id = 0
        self.tcp_sender = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.tcp_sender is not None:
            self.tcp_sender.close()

    def get_frame_data(self):
        with self.lock:
            return self.latest_frame_id, self.latest_ppm, self.latest_text

    def _set_frame(self, frame, text):
        data = frame_to_ppm(frame)
        with self.lock:
            self.latest_ppm = data
            self.latest_text = text
            self.latest_frame_id += 1

    def _run(self):
        try:
            self._capture_loop()
        except Exception as exc:
            self.on_status("Error: {}".format(exc))

    def _capture_loop(self):
        width = parse_positive_int(self.config["width"], "width")
        height = parse_positive_int(self.config["height"], "height")
        display_width = UI_PREVIEW_WIDTH
        display_height = UI_PREVIEW_HEIGHT
        tcp_host = self.config.get("tcp_host", "").strip()
        tcp_port = parse_port(self.config.get("tcp_port", "10010"), "tcp_port") if tcp_host else 0
        rtsp_url = build_rtsp_url(
            self.config["camera_ip"],
            self.config["rtsp_user"],
            self.config["rtsp_password"],
            self.config["rtsp_port"],
            self.config["rtsp_path"],
        )

        if not DEFAULT_CALIB_FILE.exists():
            raise RuntimeError("Cannot find calibration file: {}".format(DEFAULT_CALIB_FILE))

        camera_matrix, dist_coeffs = load_calibration(DEFAULT_CALIB_FILE)
        aruco_dict, detector_or_params, use_new_aruco = create_aruco_detector()
        self.tcp_sender = TcpJsonSender(tcp_host, tcp_port)

        proc = subprocess.Popen(
            build_ffmpeg_cmd(rtsp_url, width, height),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8,
        )

        self.on_status("Running")

        half = MARKER_SIZE / 2.0
        object_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        x_buf = deque(maxlen=10)
        z_buf = deque(maxlen=10)
        d_buf = deque(maxlen=10)
        a_buf = deque(maxlen=10)
        last_send_time = 0.0
        packet_seq = 1
        result_text = "No marker"

        try:
            while not self.stop_event.is_set():
                frame = read_raw_frame(proc, width, height)
                if frame is None:
                    if proc.poll() is not None:
                        raise RuntimeError("FFmpeg exited before producing frames")
                    self.on_status("Waiting for frame...")
                    time.sleep(0.05)
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if use_new_aruco:
                    corners, ids, _ = detector_or_params.detectMarkers(gray)
                else:
                    corners, ids, _ = cv2.aruco.detectMarkers(
                        gray,
                        aruco_dict,
                        parameters=detector_or_params,
                    )

                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    marker_id = int(ids[0][0])
                    image_points = corners[0][0].astype(np.float32)
                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
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

                        result_text = "ID:{}  forward:{:.3f}m  right:{:.3f}m  distance:{:.3f}m  yaw:{:.2f}deg".format(
                            marker_id,
                            z_show,
                            x_show,
                            d_show,
                            a_show,
                        )

                        now = time.time()
                        if now - last_send_time >= TCP_SEND_INTERVAL:
                            packet = build_qr_parking_packet(packet_seq, now, x_show, z_show, a_show)
                            self.tcp_sender.send(packet)
                            packet_seq += 1
                            last_send_time = now

                        p = image_points[0].astype(int)
                        cv2.putText(
                            frame,
                            "ID:{} F:{:.3f} R:{:.3f}".format(marker_id, z_show, x_show),
                            (p[0], max(35, p[1] - 45)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            frame,
                            "D:{:.3f} Yaw:{:.2f}".format(d_show, a_show),
                            (p[0], max(70, p[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                        cv2.drawFrameAxes(
                            frame,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            MARKER_SIZE * 0.5,
                        )
                else:
                    result_text = "No marker"

                show = cv2.resize(frame, (display_width, display_height))
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                self._set_frame(show, result_text)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            if self.tcp_sender is not None:
                self.tcp_sender.close()
            self.on_status("Stopped")


class DesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ArUco Desktop UI")
        self.root.resizable(False, False)
        self.worker = None
        self.photo = None
        self.last_frame_id = -1

        config = load_config()
        self.vars = {key: tk.StringVar(value=value) for key, value in config.items()}
        self.status_var = tk.StringVar(value="Stopped")
        self.result_var = tk.StringVar(value="No result")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(UI_REFRESH_MS, self._refresh_frame)

    def _build_ui(self):
        preview_w = UI_PREVIEW_WIDTH
        preview_h = UI_PREVIEW_HEIGHT

        window_w = preview_w + 400
        window_h = preview_h + 70
        self.root.geometry("{}x{}".format(window_w, window_h))

        main = tk.Frame(self.root, bg="#d9d9d9")
        main.place(x=0, y=0, width=window_w, height=window_h)

        self.video_label = tk.Label(
            main,
            text="Video stopped",
            anchor="center",
            bg="#111111",
            fg="#dddddd",
        )
        self.video_label.place(x=12, y=12, width=preview_w, height=preview_h)

        tk.Label(
            main,
            textvariable=self.result_var,
            anchor="w",
            bg="#d9d9d9",
            fg="#202020",
        ).place(x=12, y=preview_h + 22, width=preview_w, height=24)

        control_panel = tk.Frame(main, bg="#d9d9d9")
        control_x = preview_w + 36
        control_panel.place(x=control_x, y=12, width=350, height=window_h - 24)

        y = 0
        for label, key, secret in [
            ("Camera IP", "camera_ip", False),
            ("TCP IP", "tcp_host", False),
            ("TCP Port", "tcp_port", False),
        ]:
            tk.Label(
                control_panel,
                text=label,
                anchor="w",
                bg="#d9d9d9",
                fg="#202020",
            ).place(x=0, y=y, width=130, height=26)
            entry = tk.Entry(
                control_panel,
                textvariable=self.vars[key],
                show="*" if secret else "",
                relief="sunken",
                bd=1,
            )
            entry.place(x=110, y=y, width=220, height=26)
            y += 32

        y += 10
        tk.Label(
            control_panel,
            text="Status",
            anchor="w",
            bg="#d9d9d9",
            fg="#202020",
        ).place(x=0, y=y, width=130, height=26)
        tk.Label(
            control_panel,
            textvariable=self.status_var,
            anchor="w",
            bg="#d9d9d9",
            fg="#202020",
        ).place(x=110, y=y, width=220, height=26)
        y += 42

        self.start_button = tk.Button(control_panel, text="Start", command=self.start)
        self.start_button.place(x=0, y=y, width=160, height=32)

        self.stop_button = tk.Button(control_panel, text="Stop", command=self.stop, state="disabled")
        self.stop_button.place(x=170, y=y, width=160, height=32)
        y += 42

        tk.Button(control_panel, text="Save", command=self.save).place(x=0, y=y, width=330, height=32)

    def _current_config(self):
        return {key: var.get().strip() for key, var in self.vars.items()}

    def save(self):
        try:
            config = self._current_config()
            build_rtsp_url(config["camera_ip"], config["rtsp_user"], config["rtsp_password"], config["rtsp_port"], config["rtsp_path"])
            parse_port(config["tcp_port"], "tcp_port")
            parse_positive_int(config["width"], "width")
            parse_positive_int(config["height"], "height")
            config["display_width"] = str(UI_PREVIEW_WIDTH)
            config["display_height"] = str(UI_PREVIEW_HEIGHT)
            save_config(config)
            self.status_var.set("Saved")
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))

    def start(self):
        if self.worker is not None:
            messagebox.showinfo("Info", "Recognition is already running.")
            return
        try:
            config = self._current_config()
            save_config(config)
            worker = RecognitionWorker(config, self._set_status_threadsafe)
            self.worker = worker
            worker.start()
            self.status_var.set("Starting")
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
        except Exception as exc:
            self.worker = None
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            messagebox.showerror("Start failed", str(exc))

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.status_var.set("Stopped")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def _set_status_threadsafe(self, status):
        self.root.after(0, lambda: self._set_status(status))

    def _set_status(self, status):
        self.status_var.set(status)
        if status == "Running":
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
        if status == "Stopped" and self.worker is not None and self.worker.stop_event.is_set():
            self.worker = None
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def _refresh_frame(self):
        if self.worker is not None:
            frame_id, data, result = self.worker.get_frame_data()
            if data and frame_id != self.last_frame_id:
                self.photo = tk.PhotoImage(data=data, format="PPM")
                self.video_label.configure(image=self.photo, text="")
                self.last_frame_id = frame_id
            self.result_var.set(result)
        self.root.after(UI_REFRESH_MS, self._refresh_frame)

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    DesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
