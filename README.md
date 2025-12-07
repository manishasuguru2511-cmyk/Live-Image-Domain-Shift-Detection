# Domain Shift Detector (YouTube + Local) with Timezone and Web UI

A lightweight, real‑time domain shift/event detector for live video. It supports webcams, files/RTSP, and direct YouTube links (via `yt-dlp`).

Detects meaningful scene changes by combining fast image statistics with a semantic signal from a small CNN (MobileNetV2). Timestamps are written in a selected timezone and logged to JSONL; snapshots are saved for auditing. A simple HTML UI streams the video, shows a live history, and lets you download the run as CSV with snapshot links.

---

## 1) Installation (Windows)

- Prereqs: Python 3.10+ recommended, Chrome/Edge browser

```powershell
# from the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```powershell
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Notes:
- We deliberately avoid `timezonefinder`/`geopy` to simplify installation.
- Timezone handling uses stdlib `zoneinfo` + `tzdata` (Windows needs tzdata).

---

## 2) Run the HTML Web UI (recommended)

```powershell
python webapp.py
```
Then open: http://127.0.0.1:5000

- Enter your Video/YouTube link (YouTube supported via `yt-dlp`).
- Pick a US timezone from the dropdown.
- Click Start to begin.
- Live video is shown with an overlay.
- History table updates in real time.
- Click Download CSV to export all events with snapshot links.
- Click Stop to finish the run.
 - Switch to the **Analytics** tab to explore charts and a visual timeline of events.

Outputs:
- Log: `out/events.jsonl` (JSON Lines)
- Snapshots: `out/snapshots/` (start, periodic, and per event)

---

## 3) Alternative: Run from CLI

```powershell
# Example: YouTube source, USA Eastern time, display window enabled
python main.py --source "https://www.youtube.com/watch?v=ByED80IKdIU" --display --tz "America/New_York" \
  --threshold 0.08 --debounce 3 --save-first --snapshot-interval 10
```

CLI features:
- Timezone selection priority: `--tz` (IANA) > `--place` (location string) > `--auto-tz-ip` > local.
- YouTube URLs are resolved to a playable URL using `yt-dlp`.
- Snapshots can be forced with `--save-first` and `--snapshot-interval N`.

Key CLI flags (see `main.py`):
- `--source <path|rtsp|url|youtube>`
- `--display` to show a window with overlays
- `--tz <IANA>` (e.g., `America/New_York`)
- `--place "<city, state>"` to geocode and resolve timezone via HTTP (no extra deps)
- `--auto-tz-ip` infer timezone from public IP
- `--threshold 0.08` event sensitivity
- `--debounce 3` frames above threshold required to fire
- `--window 3.0` EMA time window (sec)
- `--cooldown 1.0` minimum seconds between events
- `--use-cnn / --no-cnn` enable/disable CNN
- `--cnn-weight 1.0` weight of semantic component

Outputs (CLI):
- `out/events.jsonl` (or `--out <dir>`)
- `out/snapshots/`

---

## 4) What model we use and why

- **Model**: MobileNetV2 (ImageNet pretrained), used as a fixed feature extractor (no training).
  - File: `model.py` → `_init_cnn()` builds a backbone: `features → AdaptiveAvgPool2d(1,1) → Flatten`.
  - Preprocessing: resize to 224×224, normalize with ImageNet mean/std (`_cnn_embed()`).
  - Output: a 1280‑D L2‑normalized embedding per frame.

- **Why MobileNetV2?**
  - Lightweight, fast on CPU. Good trade‑off between speed and semantic quality.
  - Robust general features from ImageNet pretraining to detect meaningful content changes.
  - Simple to deploy: works with CPU wheels of PyTorch; if PyTorch isn’t installed, the system still runs without CNN.

- **Overall approach (hybrid signal)**
  - Low‑level changes:
    - HS histogram L1 distance
    - Mean brightness change
    - Edge density change
    - SSIM distance (1 − SSIM)
  - Semantic change:
    - Cosine distance between the current frame embedding and an EMA embedding baseline: `cnn_d = 1 − cos(embed, ema_embed)`.
  - Weighted sum: `score = w_hist*hist + w_b*brightness + w_e*edge + w_s*ssim + cnn_weight*cnn_d`.
  - EMA (exponential moving average) smooths the baseline for gray, hist, brightness, edges, and embedding.
  - Event logic uses threshold + debounce + cooldown to reduce noise and duplicates.

- **Why this approach?**
  - Low‑level stats react quickly to luminance and structure changes (cheap, fast).
  - CNN features provide semantic awareness, reducing false positives from minor lighting or compression artifacts.
  - Combined score is robust and still efficient for live streaming.

Files to read:
- Detector and model: `model.py` (`DetectorConfig`, `DomainShiftDetector`)
- CLI app: `main.py`
- Web server: `webapp.py` (+ `templates/index.html`, `static/main.js`)

---

## 5) Logs and snapshots

- **JSONL log**: `out/events.jsonl`
  - First line is `run_start` with config and timezone.
  - Each event line has fields:
    - `time` (epoch seconds), `time_iso` (in selected tz), `tz`
    - `label` (lighting_change | camera_motion | scene_or_object_change)
    - `score` and `components` (hist, brightness, edge, ssim, cnn)
    - `threshold`

  Example event:
  ```json
  {
    "time": 1759898797.6,
    "time_iso": "2025-10-08T00:46:37-04:00",
    "tz": "America/New_York",
    "label": "scene_or_object_change",
    "score": 0.091,
    "components": {"hist": 0.05, "brightness": 0.001, "edge": 0.0009, "ssim": 0.0276, "cnn": 0.0114},
    "threshold": 0.08
  }
  ```

- **Snapshots**: `out/snapshots/`
  - Web UI: saves `start`, `periodic` (every 10s), and per‑event snapshots.
  - CLI: use `--save-first` and `--snapshot-interval N` for start/periodic; per‑event snapshots are saved when events fire.
  - Event snapshot naming: `<epoch>_<label>_<score>.jpg` (matches CSV mapping).

- **CSV export (Web UI)**:
  - Button: “Download CSV” → calls `GET /download.csv`.
  - Columns include `time`, `time_iso`, `tz`, `label`, `score`, components, and `snapshot`/`snapshot_url`.

### 5.1 Analytics JSON endpoint

- **JSON view**: `GET /events.json`
  - Returns a JSON payload containing:
    - `run`: the `run_start` header (or `null` if no run yet).
    - `events`: array of events with all original fields plus:
      - `snapshot`: best-matching snapshot filename (if any).
      - `snapshot_url`: relative URL (e.g., `/snapshots/<file>.jpg`).
  - Used by the Analytics visualizer to power charts and timelines.

---

## 6) Timezone handling

- All timestamps are stored in epoch seconds and formatted to `time_iso` using the selected timezone.
- Web UI: select a US timezone from the dropdown.
- CLI priority: `--tz` > `--place` (geocode+HTTP tz lookup) > `--auto-tz-ip` > local.
- Console lines and overlays also display the selected timezone.

---

## 7) Tuning tips

- **Sensitivity**: lower `--threshold` (e.g., 0.05) and/or `--debounce` (e.g., 2) to detect more events.
- **Semantic weight**: increase `--cnn-weight` to emphasize CNN distance.
- **Performance**: disable CNN (`--no-cnn`) if CPU is slow, or install Torch CPU wheels.
- **EMA window**: adjust `--window` (seconds) for smoother/faster adaptation.

---

## 8) Troubleshooting

- Video won’t open:
  - Check the link works in a browser.
  - For YouTube, ensure `yt-dlp` installed (already in `requirements.txt`).
  - Some streams require `cv2.CAP_FFMPEG` fallback (handled automatically in code).

- No CNN in components:
  - Install Torch CPU wheels: `pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision`.
  - Or run with `--no-cnn` to ignore the CNN component explicitly.

- Snapshots are missing:
  - Web UI: let the run continue >10s for periodic shots; events will save snapshots as they occur.
  - CLI: ensure `--save-first` and `--snapshot-interval 10` are present.

- Port in use / firewall:
  - The server runs on `127.0.0.1:5000`. Ensure nothing else uses that port or change it in `webapp.py`.

---

## 9) Project structure

```
├─ main.py                # CLI app (YouTube support, timezone options, snapshots, JSONL)
├─ model.py                 # Detector & MobileNetV2 feature extractor
├─ webapp.py               # Flask web server (HTML UI, MJPEG stream, SSE, CSV download)
├─ templates/
│  └─ index.html           # HTML page
├─ static/
│  └─ main.js              # Frontend logic
├─ out/
│  ├─ events.jsonl         # Log (overwritten per run)
│  └─ snapshots/           # Saved snapshots
├─ requirements.txt        # Python deps
└─ README.md               # This file

---

## 11) Analytics visualizer (Web UI)

- **Tabs**:
  - **Monitor**: existing live stream and real-time history table.
  - **Analytics**: dashboard-style view over past and live events.
- **Analytics overview**:
  - Summary cards: total events, per-label counts, time span, average score, last event.
  - Time-series chart: score over time, with a threshold line and points colored by label:
    - `lighting_change`, `camera_motion`, `scene_or_object_change`.
  - Timeline strip:
    - Scrollable horizontal list of event tiles with thumbnails (where snapshots exist), labels, time, and score.
    - Clicking a tile highlights it and syncs with the chart selection.
- **Filters & live updates**:
  - Label filters to toggle individual labels on/off in both chart and timeline.
  - Analytics loads existing runs via `GET /events.json` and then updates live via the same SSE stream that powers the Monitor view.
```

---

## 10) License

This code is provided as‑is for demonstration and internal evaluation purposes.
