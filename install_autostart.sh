#!/bin/sh
set -eu

APP_DIR="/root/aruco"
APP_FILE="$APP_DIR/aruco_desktop_ui.py"
AUTOSTART_DIR="/root/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/aruco_desktop_ui.desktop"

if [ ! -f "$APP_FILE" ]; then
    echo "[ERR] Cannot find $APP_FILE"
    exit 1
fi

mkdir -p "$AUTOSTART_DIR"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=ArUco Desktop UI
Comment=Start ArUco recognition UI on login
Exec=sh -lc 'sleep 5; cd /root/aruco && DISPLAY=\${DISPLAY:-:0} /usr/bin/python3 aruco_desktop_ui.py'
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

chmod +x "$APP_FILE"
chmod 644 "$DESKTOP_FILE"

echo "[OK] Installed autostart: $DESKTOP_FILE"
echo "[INFO] Reboot or log out/in to test."
