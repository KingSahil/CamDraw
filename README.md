# CamDraw — Gesture Drawing App

A small hand-gesture drawing app using OpenCV and MediaPipe. Draw on a virtual canvas with your index finger, erase with your open palm, change colors with a 3-finger gesture, and toggle UI or vertical aspect ratio for social video.

## Features
- Index finger drawing with smooth strokes and simulated pressure (thickness based on speed)
- Palm-based eraser (follows palm center)
- 3-finger gesture to cycle drawing colors (debounced — changes once per gesture)
- Toggle overlays (UI, skeleton, text) with `Z`
- Toggle vertical 9:16 view (center-cropped for Instagram/TikTok) with `V`
- Clear canvas with `C` key
- Quit with `Q`

## Requirements
- Python 3.8+ (tested with 3.12)
- Packages listed in `requirements.txt`:
  - opencv-python
  - mediapipe
  - numpy

If you don't have them installed:

```powershell
py -3 -m pip install -r requirements.txt
```

## Run
From the project folder (Windows / PowerShell):

```powershell
py -3.12 app.py
```

If you want to hide MediaPipe/TensorFlow startup warnings that appear in stderr, run:

```powershell
py -3.12 app.py 2>$null
```

## Controls and Gestures
- Drawing
  - Raise only your index finger (other fingers down). Move your index finger to draw. Strokes are smoothed and thickness simulates pressure (slower movement -> thicker line).

- Erasing
  - Open your hand (all fingers) — the eraser follows the palm center. Move your whole hand to erase.

- Change color
  - Raise index + middle + ring fingers (3-finger gesture). The color cycles once per gesture. Close fingers and reopen to change again.

- UI / View
  - `Z` — Toggle overlays (hides skeleton, text, palette, cursor).
  - `V` — Toggle vertical 9:16 aspect view (center-cropped, no rotation).
  - `C` — Clear canvas.
  - `Q` — Quit app.

## Tweakable Parameters
Open `app.py` to adjust these values near the top of the file:
- `erase_radius` — Eraser radius in pixels (default set in the file).
- `min_thickness`, `max_thickness`, `base_thickness` — Pressure / stroke thickness settings.
- `max_lost_frames` — Number of frames the app will keep drawing after temporary tracking loss.

Example: to increase palm eraser size, set `erase_radius = 70`.

## Troubleshooting
- "ModuleNotFoundError: No module named 'mediapipe'": install requirements (`pip install mediapipe`).
- Camera not opening: check camera index in `cv2.VideoCapture(0)` or other apps using camera.
- If hand tracking drops when moving fast, the app reduces detection thresholds to be lenient, but good lighting and holding the hand in view helps performance.
- MediaPipe prints some startup warnings (TensorFlow backend). They are informational only. See "Run" above to redirect stderr if you want a clean console.

## Notes
- This app intentionally uses an approximate / heuristic approach for gesture detection — it may need calibration for different cameras, lighting, and hand sizes.
- If you want a permanent portrait output (e.g., save video) consider capturing frames and writing to a video file at 9:16 resolution.

## License
This is your project. Feel free to copy, modify, or integrate into your own work.

---

If you want, I can add a small screenshot, a sample saved video export command, or a simple README badge. Tell me what you'd like next.