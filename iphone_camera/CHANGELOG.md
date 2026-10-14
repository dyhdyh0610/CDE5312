# Changelog

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
