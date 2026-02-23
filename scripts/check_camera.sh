#!/bin/bash
# Run this on the Pi to see why OpenCV might not see the camera.
# Usage: bash scripts/check_camera.sh

echo "=== 1. Video devices (OpenCV uses these) ==="
ls -la /dev/video* 2>/dev/null || echo "No /dev/video* found."
echo ""

echo "=== 2. V4L2 device info (if v4l2-ctl installed) ==="
if command -v v4l2-ctl &>/dev/null; then
  v4l2-ctl --list-devices 2>/dev/null || true
else
  echo "Install with: sudo apt install v4l-utils"
fi
echo ""

echo "=== 3. rpicam sees camera? ==="
if command -v rpicam-hello &>/dev/null; then
  timeout 2 rpicam-hello -t 1 2>&1 | head -5 || true
else
  echo "rpicam-hello not found."
fi
echo ""

echo "=== 4. Try OpenCV in Python ==="
python3 -c "
import cv2
for i in range(3):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    opened = cap.isOpened()
    read_ok = False
    if opened:
        ret, frame = cap.read()
        read_ok = ret and frame is not None
        cap.release()
    print(f'  index {i}: opened={opened}, read_ok={read_ok}')
" 2>/dev/null || echo "  (run from project venv: source venv/bin/activate then run this script)"
echo ""

echo "If no /dev/video* or OpenCV fails: use USE_RPICAM=1 python app.py (rpicam-vid)."
echo "If /dev/video0 exists but OpenCV fails to read: may need different pixel format or driver."
