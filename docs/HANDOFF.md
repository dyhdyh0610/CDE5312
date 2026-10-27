# Handoff

## GitHub Publication Preparation — 2026-10-27

The user explicitly requested uploading the complete camera project to GitHub,
including data sanitization, and selected account `dyhdyh0610`. The publication
includes both camera applications, the pending iPhone stability changes, Sony
handheld tuning, ESP32 haptics, firmware and tests. README.md supplies the project
entry point and fresh-install instructions. Personal absolute paths in the
current documentation have been replaced by `/path/to/CDE5312`; `.gitignore`
also excludes media, secrets, model exports and generated packages.

Verification: all 59 Sony/haptics tests and all 52 iPhone tests pass using the
existing local environment, models and validation images. This includes virtual
serial transport and the actual firmware sketch with C++ test stubs; it does
not newly validate physical camera connections, board flashing or motors.
Validation images/models remain local, so a fresh clone needs those resources
for the full integration suite, as explained in README.md.

Publication uses a separate sanitized Git repository under
`.git/github-publish/`, leaving original local release tags and history intact.
Its historical branches and V1.0/V1.1/V1.2 tags retain their code while personal
paths in historical blobs are replaced; commit hashes therefore differ.
GitHub `main` is the complete current checkpoint. Future uploads must use the
sanitized publication repository and repeat the content/history audit; do not
push the original unsanitized history directly. Older sections below are
historical records and may describe superseded local-only or hardware status.

## Sony Cup/Tree Recall Fix — 2026-10-27

The user reported missing cups and trees BEFORE choosing a subject. Reproduced
misses using mixed public-image JPEG previews: at 640x284/quality 90 the old
0.35 threshold dropped a tree at 0.340; at 480x212/quality 40 it dropped a cup
at 0.301. The old inventory also ran only once after stability, preserving any
initial misses until a scene change or explicit reset.

`sony_targets.py` now keeps person confidence at 0.35 and sets independent
cup/tree defaults to 0.20. Both models' BoT-SORT high/new-track thresholds are
bounded by the corresponding detection floor; otherwise its default 0.25
new-track gate would still discard accepted weak objects. Per-category output
filtering preserves the stronger person threshold. The existing prompt-free
YOLOE also detects cup/coffee cup/paper cup/insulated cup/mug/teacup alongside
natural trees. `target_class()` maps these to cup/tree; unrelated cup homonyms,
bottles, and potted plants are excluded. No new weights or dependencies.

`sony_wifi_yolo.py` adds --cup-confidence and --tree-confidence; --confidence
still controls people in guidance and all classes in --coordinates-only.
YOLOE IDs now have the `scene:` prefix and retain `source_class`. Do not relabel
all scene-model results as trees. `sony_guidance.py` refreshes a stable inventory
0.75 seconds after inference completion until selection, including empty
scenes. Refreshes keep trackers and existing numbers; new stable windows/R
reset them. Cross-model cup boxes with IoU >= 0.5 are merged, preferring the
selected/already-numbered track. Missing selected cups do not switch models or
identities. The 1.5-second timing, handheld tolerance, native coordinates, and
immediate tracking after selection are retained. AGENTS.md/INSTALLATION.md
document these rules.

Verification: 48 Sony tests and all 52 iPhone tests pass. Added real compressed
mixed-scene inference, actual 0.22-confidence tracker admission, cup-label
normalization, duplicate/locked-cup handling, stable numbers, and local camera
API/JPEG video -> incomplete inventory -> automatic complete inventory without
restabilizing. CPU and MPS both detect people, trees, and cups in the repro
previews; MPS results are 3 people / 1 tree / 1 cup for both sizes. The Chinese
selection panel was rendered and visually inspected in ignored
installation_check/sony_all_targets_fixed_preview.jpg. Test logs and local
video-emulation JSON also remain ignored. No debug instrumentation was added.
Physical Sony video remains unverified: SSDP discovery returned no cameras on
the current network. Reopen the existing launcher to use the changes. Pending
iPhone edits remain untouched and outside this local-only checkpoint.

## Sony Handheld Stability Checkpoint — 2026-10-27

The Sony 1.5-second wait now tolerates small handheld camera wobble. Previously,
the public bus fixture reset the countdown for just 6–7 native pixels of
translation or 0.6 degrees of rotation. `sony_guidance.py` now estimates global
translation with windowed phase correlation on the existing 160x96 thumbnail.
Reliable estimates (response >= 0.5) may be compensated up to 2.5% of width and
height from the fixed anchor (16x12 px at 640x480); larger motion resets the
timer. Alignment excludes the exposed border and does not alter camera frames,
model input, displayed images, or object coordinates. Unreliable/low-texture
estimates fall back to unaligned image differences. Mean difference tolerance
increases from 4 to 12, changed-pixel cutoff from 20 to 30, and allowed changed
fraction from 4% to 12%; uniform exposure compensation remains capped at 6.
The anchor stays fixed, so slow pans cannot accumulate tolerated frame-to-frame
shifts. Scene changes, frame gaps, and explicit resets still restart the wait;
selected targets continue live tracking without any stability wait.

Verification: 38 Sony tests pass (13 protocol/coordinates + 25 guidance),
including four new handheld/large-motion/slow-pan regressions. Translation
tests cover 640x480 and 1280x720, with sensor noise and exposure changes;
rotation tests preserve candidate selection through continued small wobble.
All 52 iPhone tests pass. A separate 100-frame deterministic public-image
simulation with +/-12 px horizontal, +/-8 px vertical, +/-0.8 degree rotation,
noise and exposure changes produced zero countdown resets; mean stability
check cost was 0.52 ms on this Mac. This is simulated jitter, not physical
Sony handheld validation. Camera hardware/actual Wi-Fi video remain unverified.
INSTALLATION.md documents the new defaults; restart the existing launcher to
use them. Unrelated pending iPhone files remain unchanged and uncommitted.
This checkpoint is local only; release tags remain preserved.

## Sony Shooting Guidance Checkpoint — 2026-10-27

The Sony default path now waits for 1.5 continuous seconds of stable video,
then inventories only people, trees, and cups. The user selects a particular
numbered object by clicking its box or its Chinese side-panel row. N/P pages
crowded lists; R clears selection and starts a new stable-scene scan. After
selection, the app tracks immediately on each inference without another
stability wait and gives Chinese pan/tilt instructions, native pixel offsets,
distance to the frame center, and small/moderate/large adjustment hints.
Centering uses 3% of frame width/height (minimum 8 px), with 1.3x exit hysteresis.
The instructions say when the subject is centered; no shutter is triggered.

`sony_guidance.py` owns the anchored stability gate, state machine, guidance,
and Chinese mouse-selectable panel. `sony_targets.py` reuses root `yolo26n.pt`
for person/cup and adds the local prompt-free `yoloe-26n-seg-pf.pt` for trees
and a whitelist of natural tree types. Do not substitute potted plants, cat
trees, shoe trees, or tree frogs. Tree subtype labels are normalized to `tree`
with the original label retained in `source_class`. No text encoder or cloud
inference is involved. The tree weights (~15.5 MB) and newly installed
scipy 1.17.1 / lap 0.5.13 are local; model files stay ignored.

BoT-SORT tracks camera motion. Both model trackers are initialized before any
visible target IDs are assigned to avoid resetting the shared ID counter while
another model has active tracks. Application selection stays bound to the
same source-prefixed tracking ID. A missing target clears directions and shows
a lost-target message; other detections do not substitute for it. Occlusion,
fast movement, or crossing subjects can still confuse the underlying tracker.
Before selection, visible scene changes invalidate candidates; initial model
computation time alone does not discard an already stable choice frame.

The existing launcher is unchanged. `--coordinates-only` retains the previous
full-class coordinate mode; `--preview` retains a no-inference coordinate view.
Default `--headless` JSON includes the `shooting` state and Chinese guidance;
`--target-number N` is an explicit headless choice after the stable inventory.
See INSTALLATION.md for dependency preparation, flags, and text semantics.

Verification: 34 Sony tests pass (13 existing + 21 workflow tests), including
exact timing, slow pan/noise/gap handling, same-class selection, lost-target
behavior, directions, centering hysteresis, mouse hit regions, pagination,
real person/tree/cup tracking, and local Sony API/JPEG stream -> selection ->
real YOLO -> Chinese guidance -> clean shutdown. All 52 iPhone tests pass.
A real OpenCV window using public test imagery ran for 26 frames, selected
candidate #2 through the panel's mouse handler, and entered live guidance.
Native external mouse delivery and physical Sony video remain unverified.
The latest SSDP discovery on the current network returned no Sony cameras.

The cached models also pass MPS inference: five-sample detector/tracker means
were 39.7 ms for people, 145.9 ms for trees, and 33.4 ms for cups on the local
test images; these figures exclude camera acquisition and UI rendering.
Selection/guidance previews were visually inspected under ignored
installation_check/sony_target_selection.jpg and sony_shooting_guidance.jpg.
Additional tree/cup fixtures and source notes remain local in installation_check/.
The tree fixture is a public forest photo; the cup fixture is COCO128 image
000000000562.jpg. No camera frames, model weights, or generated output are committed.
Pending iPhone working-tree changes remain outside this milestone. Release
tags are preserved and this checkpoint is local only.

## iPhone Coordinate Checkpoint — 2026-10-21

V1.2 changes the iPhone default path to live YOLO object coordinates. The user
explicitly selected existing detection boxes with realtime updates, rather than
open-vocabulary visual-language localization. Coordinates use native-frame box
centers: X=(x1+x2)/2-W/2, Y=H/2-(y1+y2)/2, +X right, +Y up, one decimal pixel.
The overlay and Chinese coordinate table use the same current detections; window
scaling does not change values. N/P pages crowded tables. No Ollama calls,
stability delay, photo storage or per-frame disk logging are used by the app.
The earlier scene/stability module and pending handheld-tolerance edits remain
available outside the default path; the README preserves the earlier notes.

Verification in the working tree: 52 tests pass, including the existing pending
stability tests and new center/quadrant/odd-resolution/scaling/label-placement/
pagination/live-changing-coordinate checks. An isolated export of exactly the
staged release passes 38 tests; the additional 14 pending stability checks remain
local and unstaged. Real local YOLO identifies 5 objects and outputs verified
native coordinates on the existing bus sample. A 30-sample warmed MPS benchmark
of YOLO plus the full coordinate panel averages 18.1 ms/frame (55.3 fps excluding
camera acquisition and display), with panel rendering averaging 5.0 ms. The
reviewed preview is ignored at iphone_camera/output/iphone_coordinates_preview.jpg.

Actual connected iPhone and authorized camera were verified. An 80-frame 1920x1080
headless run on MPS averaged 30.0 output fps (33.3 ms), with first output after
2.72 seconds including startup. The real GUI opened successfully; its 120-frame
measurement averaged 30.0 display fps with 3.8 ms panel rendering. These camera
measurements contained zero detected objects, so they verify live acquisition
and the empty-scene display path; live object localization in a populated scene
has not been measured. Object recognition and moving coordinate values are
covered by real-image inference and controlled frame-sequence tests respectively.
The runtime window was left open for the user; Q/Esc closes it.

Run: iphone_camera/启动iPhone识别.command. Tests: .venv/bin/python -m unittest
discover -s iphone_camera -v. Headless JSON includes image_size, coordinate_system
and detections with id/class/confidence/xyxy/position. IDs identify boxes within
each frame, not persistent object identities. The retired scene-model/stability
CLI options belong to V1.1 and are no longer used in V1.2; --no-scene remains
accepted for compatibility. Sony application files are unchanged by this task.

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
