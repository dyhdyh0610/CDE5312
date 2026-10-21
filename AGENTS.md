# Project rules

This repository contains local camera applications on an Apple Silicon Mac.
`iphone_camera/` owns the iPhone Continuity Camera application. Sony scripts at
the root are a separate application; do not change them for iPhone features.

- Reuse the root `.venv/bin/python` and local `yolo26n.pt`.
- Sony tests: `.venv/bin/python -m unittest -v test_sony_wifi_yolo`.
- Sony object positions use bounding-box centers in native-frame pixels, with
  the frame center as `(0, 0)`, positive X right, and positive Y up.
- Run iPhone tests: `.venv/bin/python -m unittest discover -s iphone_camera -v`.
- Enumerate cameras without opening them: `cd iphone_camera && ../.venv/bin/python iphone_yolo.py --list`.
- Never substitute a built-in camera when the selected iPhone disappears.
- Keep camera frames and image descriptions local. Do not commit photos,
  recordings, models, credentials, virtual environments, or generated output.
- Report visual depth as an inference, never as measured metres.
- Preserve release tags. V1.0 is the original iPhone YOLO preview application.
- `docs/HANDOFF.md` carries current status and verification limits.

Version control is local unless the user explicitly requests a remote upload.
