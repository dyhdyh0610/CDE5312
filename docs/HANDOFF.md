# Handoff

## Sony Coordinate Checkpoint — 2026-10-21

Sony Wi-Fi preview now draws centered Cartesian axes and a 100 px grid.
Each detected object has a center cross and a label containing its class,
confidence, and `(X, Y)` in original-frame pixels. X is positive to the right;
Y is positive upward. Coordinates use the bounding-box center, are displayed
to 0.1 px, and represent image locations rather than relative positions or
world distances. Nearby coordinate labels are moved apart where space permits.

`--headless` retains existing detection fields and adds `position: {x, y}`, plus
frame-level `image_size` and `coordinate_system`. `--preview` shows axes without
running YOLO. Use the existing root `启动相机识别.command` launcher after joining
the Sony camera's DIRECT Wi-Fi. See `INSTALLATION.md` for coordinate semantics.

Verification: 13 Sony tests pass, including center/quadrant/odd-resolution
geometry, label placement, and loopback Sony API + framed JPEG + real local YOLO
inference with coordinate output and shutdown. All 44 existing iPhone tests
also pass. The annotated existing test image was visually inspected; its local
preview is ignored under `installation_check/sony_coordinates_preview.jpg`.
No physical camera was opened, and Sony Wi-Fi streaming, live window operation,
and sustained performance still require a connected alpha6000 to verify.
The pre-existing iPhone working-tree edits are outside this Sony checkpoint.
Release tags V1.0 and V1.1 are preserved; version control remains local.

## Current Status

2026-10-14: Original iPhone application preserved as tag V1.0, commit 8851d22
on main. V1.1 on codex/iphone-scene-description adds local scene descriptions
beside video. 30 tests pass (11 baseline + 19 new), including real YOLO inference,
stability timing, cancellation, local-only streaming API, Chinese rendering and
CLI integration. No physical camera was opened during this milestone.

Real-model checks on the supplied photo pass. In a simulated camera sequence,
initial motion lasts 1 second; the real local vision request starts at 3.11
seconds, first text at 14.95 seconds, and completion at 19.49 seconds. Description
generation takes 16.34 seconds. Generated previews/checks are ignored under
iphone_camera/output/. Source photos and model data are not committed.

## Confirmed Decisions

- Reuse root Python environment and YOLO model.
- Sony files are outside this milestone.
- Version history is local; no remote is configured.
- Default mode describes near/middle/background and left/right/front/behind
  relationships in a Chinese panel after 2 seconds of stability. This is the
  submission delay, not a guarantee of completion in 2 seconds.
- User selected local inference. Ollama is running; qwen3.5:4b and qwen3.5:9b
  are already installed and advertise vision support. Default: qwen3.5:9b.
- Camera thread observes raw frames at ~10 Hz independently of foreground YOLO
  inference. Motion invalidates descriptions immediately; stale requests cannot
  publish their text. Normal video operation does not store camera images.
- No new packages or model downloads were needed; reuse requests/Pillow/OpenCV.

## Open Questions

At the latest enumeration, only the built-in Mac camera was visible; iPhone
streaming, permission prompts and sustained GUI operation remain unverified.
The invoking environment reports camera authorization as not_determined.
Visual near/far ordering is approximate, not metric depth. The model can miss
small background people/objects or confuse cups with bottles; do not present its
descriptions as exact ground truth.

## Recommended Next Steps

Connect/trust/lock the iPhone, verify --list, and launch
iphone_camera/启动iPhone识别.command with Ollama running. Check that actual camera
motion clears text and stable scenes trigger after 2 seconds. Press R to retry,
Q/Esc to exit. --no-scene uses the original YOLO behavior without Ollama.

Run tests from root: .venv/bin/python -m unittest discover -s iphone_camera -v.
From iphone_camera/: ../.venv/bin/python iphone_yolo.py --check-scene checks the
local model without opening hardware; --image PATH --headless --output
output/preview.jpg performs explicit still-image verification without a camera.
