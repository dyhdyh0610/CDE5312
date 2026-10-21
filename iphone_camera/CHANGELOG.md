# Changelog

## V1.2 — 2026-10-21

- Replace relative scene prose with per-frame absolute image coordinates for
  existing YOLO detection boxes, following the user's selected realtime scope.
- Center the image-plane axes at (0, 0), with +X right and +Y up in native pixels.
- Show object centers, signed coordinates and grid/axes beside a Chinese table;
  keep coordinates independent of display scaling and support crowded tables.
- Remove Ollama and the 2-second stability gate from the default camera path.
  Keep earlier scene/stability modules for historical work, outside this path.
- Add frame-level coordinate metadata and per-object id/position to JSON output.

## V1.1 — 2026-10-14

- Add Chinese scene descriptions beside the live YOLO video.
- Start local image analysis after 2 continuous seconds of scene stability.
- Use the installed Ollama `qwen3.5:9b` vision model for relative positions and
  visual near/far ordering; no cloud API, frame upload or metric distance claim.
- Observe raw camera frames independently of YOLO latency; tolerate minor noise
  and reset for motion, slow accumulated drift or interrupted frame delivery.
- Stream descriptions, cancel changed scenes and reject stale results.
- Add R to retry, model checks, V1.0 mode and explicit still-image verification.
- Verify Chinese rendering, timing and cancellation with automated tests and the
  supplied photo using real YOLO and local vision inference.

## V1.0 — 2026-10-14

Original independent iPhone Continuity Camera application: device selection,
live preview, local YOLO detection, permissions and disconnect handling.
Reuses the parent Python environment and model. Baseline: 11 tests pass.
This version does not produce natural-language scene descriptions.
