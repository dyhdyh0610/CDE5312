# Handoff

## Current Status

2026-10-14: Save the existing independent iPhone application as V1.0 before
adding scene descriptions. Baseline: 11 tests pass, including real local YOLO
inference on a simulated camera. No live camera was opened during this check.

## Confirmed Decisions

- Reuse root Python environment and YOLO model.
- Sony files are outside this milestone.
- Version history is local; no remote is configured.
- The requested next release describes foreground/background and relative
  positions in a Chinese panel beside the video after 2 seconds of stability.
- User selected local inference. Ollama is running; qwen3.5:4b and qwen3.5:9b
  are already installed and advertise vision support.

## Open Questions

Live iPhone frames, camera authorization and sustained GUI operation require
hardware verification. Visual near/far ordering is approximate, not metric depth.

## Recommended Next Steps

Implement asynchronous local image description, stability gating and the Chinese
side panel. Verify on the supplied reference photo and simulated moving/still
frames before checking the connected iPhone.
