## Domain Shift Detector – End‑to‑End Design and Rationale

This document explains **what the project is**, **how it works end to end**, and **why specific design choices and parameter values were made**. It is written as an engineering design doc so someone new to the repo can understand and extend it confidently.

---

## 1. Problem statement and goals

- **Problem**: Given a live or streamed video (webcam, RTSP, file, or YouTube), detect **meaningful “domain shifts”**:
  - Sudden **lighting changes** (lights on/off, exposure jumps).
  - **Camera motion** (pans, shakes) that significantly alter the view.
  - **Scene or major object changes** (cut to another shot, new objects, people entering/leaving).
- **Constraints**:
  - Must run on **CPU** in real time or near real time.
  - Must work on **Windows** with minimal dependency friction.
  - Should produce **auditable logs and snapshots** that can be reviewed and exported.
  - Should support both:
    - A **CLI workflow** for scripting and batch runs.
    - A **web UI** with live video, history, and analytics visualizations.

**Non‑goals**:
- It is not a fully trained scene classification system.
- It does not do long‑term tracking or object detection; it focuses on **change detection**.

---

## 2. High‑level architecture

The system has three main layers:

- **Detector core (`model.py`)**
  - Stateless API: `DomainShiftDetector.step(frame, now=None) -> Optional[event_dict]`.
  - Maintains internal EMAs and logic to decide when an event fires.
- **Runtimes**
  - **CLI (`main.py`)**: runs the detector in a loop, logs to JSONL, writes snapshots, optional OpenCV window.
  - **Web server (`webapp.py`)**: Flask app with:
    - MJPEG video streaming.
    - Server‑Sent Events (SSE) for event stream.
    - A worker thread that runs the detector.
    - Endpoints for CSV and JSON export, snapshot serving.
- **Frontend (`templates/index.html`, `static/main.js`)**
  - **Monitor tab**: Live preview + simple history table.
  - **Analytics tab**: Dashboard‑style visualizer with summary stats, a score‑over‑time chart, and an interactive timeline with thumbnails.

Outputs:

- `out/events.jsonl`: JSONL log with a single `run_start` header and per‑event lines.
- `out/snapshots/`: JPEG snapshots (start, periodic, and per‑event).
- `GET /download.csv`: CSV export for spreadsheet analysis.
- `GET /events.json`: JSON API powering the Analytics visualizer.

---

## 3. Detector design and parameter choices

### 3.1 DetectorConfig

`DetectorConfig` (in `model.py`) controls how sensitive the detector is and how it behaves over time:

- `resize = (640, 360)`
  - **Why**: This is a good compromise between:
    - Enough resolution for histogram / edge patterns and CNN embeddings.
    - Low CPU cost per frame.
  - 640×360 is a common 16:9 size and scales well from HD sources.

- `window_seconds = 3.0`
  - Controls the **EMA time constant**: how quickly the detector’s “baseline” adapts.
  - We compute `alpha = 1 − exp(−dt / window_seconds)` per frame.
  - **Why 3 seconds**:
    - Fast enough to adapt to slow lighting drift or camera auto‑exposure.
    - Slow enough that true sudden changes (cuts, big motion) stand out as spikes vs the EMA.

- `threshold` (CLI default `0.6`, web default `0.08` in `webapp.WorkerConfig`)
  - The **score threshold** above which we consider the frame “suspicious”.
  - Different entry points use slightly different defaults; they are intended to be tuned:
    - CLI: more conservative by default, assuming higher variance sources.
    - Web: more sensitive to show more events in a demo/monitoring context.
  - The user can override via `--threshold` in CLI or via code/config changes for web.

- `debounce`
  - Number of **consecutive frames** that must exceed `threshold` before firing an event.
  - CLI default: `8`. Web default: `3`.
  - **Why**:
    - Single‑frame spikes (compression artifacts, noise) should not trigger events.
    - 8 frames at ~25 fps ≈ 0.3 s of sustained change; 3 frames ≈ 0.12 s.  
    - The web UI uses a smaller debounce to be more responsive; CLI defaults can be stricter.

- `cooldown_seconds = 1.0`
  - Minimum real‑time seconds between events.
  - **Why**:
    - Prevents spamming multiple events for a single transition.
    - 1 second is short enough to capture closely spaced changes while filtering noise.

- `weights = (1.0, 0.6, 0.6, 1.0)` for `(hist_l1, brightness_diff, edge_diff, ssim_dist)`
  - Histogram and SSIM are given higher weight because:
    - They represent **structural and color distribution** changes.
  - Brightness and edge density are lower weight:
    - They are more sensitive to mild camera noise and automatic exposure.
  - These values were chosen experimentally to balance:
    - Sensitivity to real content changes.
    - Robustness against minor flicker/noise.

- `use_cnn = True`, `cnn_weight = 1.0`
  - Adds a **semantic component** using MobileNetV2 embeddings.
  - `cnn_weight` controls how much the semantic distance influences the total score.
  - **Why**:
    - CNN features help distinguish “real content change” from pure lighting shifts.
    - Weight 1.0 is a reasonable starting point; users can increase it if they trust semantic cues more (at some CPU cost) or set `--no-cnn` to disable it.

### 3.2 Feature design

For each frame, we compute:

- **HS histogram** (`_hist_hs`):
  - 16×16 bins over Hue and Saturation, L1‑normalized.
  - Captures overall color distribution.
  - Why HS: robust to brightness changes; focuses on color content.

- **Brightness (V channel mean)**:
  - Average brightness in [0,1].
  - Separately measures lighting change; useful for the `"lighting_change"` label logic.

- **Edge density** (`_edge_density`):
  - Fraction of pixels marked as edges by Canny.
  - Increases with texture/structure; useful for motion and structure changes.

- **SSIM distance (`1 − SSIM`)**:
  - Structural similarity vs EMA gray image.
  - High when the structure of the scene changes significantly.

- **CNN embedding distance**:
  - MobileNetV2 feature vectors, L2‑normalized.
  - Distance: `1 − cos(embed, ema_embed)` (cosine distance).
  - Adds semantic awareness (e.g., new objects/people).

The **score** is:

\[
\text{score} = w_\text{hist} \cdot \text{hist\_l1} +
               w_b \cdot \text{bright\_d} +
               w_e \cdot \text{edge\_d} +
               w_s \cdot \text{ssim\_dist} +
               \text{cnn\_weight} \cdot \text{cnn\_d}
\]

**Why this hybrid approach**:

- Pure low‑level metrics (histogram, SSIM, edges) are fast and sensitive but can be fooled by lighting/exposure.
- Pure CNN features are expensive and may miss small but important structure changes.
- The hybrid design:
  - Gives fast, cheap signals for obvious shifts.
  - Lets the CNN refine decisions where semantics matter (e.g., same lighting but a new object).

### 3.3 Event logic (debounce + cooldown + categorization)

- **Debounce**:
  - Avoids triggering on one‑frame anomalies.
  - If the score dips below threshold, `exceed_count` resets to zero.

- **Cooldown**:
  - After an event fires, we wait `cooldown_seconds` before allowing another event.
  - This prevents near‑duplicate events in highly dynamic scenes.

- **Categorization** (`_categorize`):
  - Heuristic rules based on feature contributions:
    - Large `bright_d`, but small `hist_l1` and `edge_d` → `"lighting_change"`.
    - Large `ssim_d` and `edge_d`, but small `bright_d` → `"camera_motion"`.
    - Everything else → `"scene_or_object_change"`.
  - **Why heuristics**:
    - Avoids training a classifier; keeps system zero‑shot with interpretable rules.
    - Easy to tune while inspecting components in logs and the new visualizer.

---

## 4. CLI runtime (`main.py`)

### 4.1 Why a CLI

- Enables:
  - Headless servers.
  - Cron jobs or batch processing.
  - Easy experimentation/tuning via flags.

### 4.2 Key parameters and rationale

- `--source` / positional `pos_source`
  - Accepts:
    - File path.
    - RTSP URL.
    - HTTP URL (including YouTube).
    - Numeric webcam index (e.g., `"0"`).
  - **Why**: Provides a single entrypoint for multiple video backends.

- YouTube handling (`is_youtube`, `resolve_youtube_stream`)
  - Uses `yt_dlp` to get a **direct playable URL** to feed OpenCV.
  - Fallback to raw URL on failure.
  - **Why**:
    - OpenCV often struggles with YouTube pages directly; `yt_dlp` normalizes this.

- Timezone options:
  - `--tz`: explicit IANA timezone.
  - `--place`: free‑text location (geocode → lat/lon → timezone).
  - `--auto-tz-ip`: infer from public IP.
  - **Why**:
    - Many cameras are remote; correct local time is important for auditing.
    - We avoid heavy deps like `timezonefinder`, instead using HTTP APIs.

- Output control:
  - `--out`: base directory, default `out/`.
  - `--display`: OpenCV window for debugging/monitoring.
  - `--snapshots`, `--snapshot-interval`, `--save-first`:
    - Provide visual evidence of events over time and at run start.

- CNN options:
  - `--use-cnn/--no-cnn`, `--cnn-weight`.
  - **Why**: Let the user pick trade‑offs between CPU cost and semantic power.

- `--max-frames`:
  - Simple cap for testing and batch jobs.

### 4.3 Logging format

- First line:
  - `"type": "run_start"` with:
    - `time`, `time_iso`, `source`, `threshold`, `window`, `debounce`, `cooldown`, `tz`.
- Event lines:
  - `time`, `time_iso`, `tz`, `label`, `score`, `components`, `threshold`.
- **Why JSONL**:
  - Append‑friendly and streamable.
  - Easy to parse incrementally and friendly to tools like `jq`, pandas, etc.

---

## 5. Web runtime (`webapp.py`)

### 5.1 Why a web UI

- Provides:
  - A **zero‑install** UI (just open the browser).
  - Live preview and control buttons.
  - A place to host richer visualizations (Analytics).

### 5.2 Worker thread and state

- `WorkerConfig`:
  - Holds source, timezone, threshold, debounce, cooldown, CNN flags.
- `ServerState`:
  - `thread`, `stop_evt`, `running`, `tzinfo`, `last_jpeg`, SSE `subscribers`, and cached `run_header`.
  - Methods:
    - `add_subscriber()`, `remove_subscriber()` for SSE.
    - `broadcast(obj)` to push JSON events to all subscribers.

**Why a dedicated worker thread**:

- Keeps the Flask main thread non‑blocking.
- Allows long‑running detection loops without affecting HTTP responsiveness.

### 5.3 Streaming and events

- `/stream` (MJPEG):
  - Serves `state.last_jpeg` in a multipart response, ~10 fps (100 ms sleep).
  - **Why MJPEG**:
    - Simple and widely supported for “fake live” video via `<img src="/stream">`.

- `/events` (SSE):
  - Each connected client gets:
    - `run_start` header (if available).
    - Subsequent event dicts as JSON.
    - Periodic `"ping"` keep‑alives.
  - **Why SSE**:
    - One‑way, server→browser stream is enough; simpler than WebSockets.

### 5.4 Logs, snapshots, and exports

- Same log format as CLI, written to `out/events.jsonl`.
- Snapshots:
  - `start`: first frame of the run.
  - `periodic`: every 10 seconds.
  - Per‑event: `<epoch>_<label>_<score>.jpg`.
- `/snapshots/<filename>`:
  - Serves JPEGs from `out/snapshots/`.
- `/download.csv`:
  - Converts JSONL into a CSV with:
    - Run header row.
    - Event rows, including component columns and snapshot URL.
- `_snapshot_name_for_event(ev)`:
  - Reconstructs canonical names or falls back using time prefix.
  - **Why**:
    - Snapshot naming is deterministic but in practice slight timing differences may occur; this helper makes the mapping robust.

### 5.5 `/events.json` – Analytics API

- Returns:

  - `run`: last `run_start` record (or `null`).
  - `events`: event list with original fields plus:
    - `snapshot`: best snapshot filename.
    - `snapshot_url`: `/snapshots/<name>`.

- **Why**:
  - A structured JSON API is more convenient for the frontend visualizer than parsing JSONL.

---

## 6. Frontend design (`index.html`, `main.js`)

### 6.1 Monitor tab

- **UI elements**:
  - Source input, timezone dropdown, Start/Stop/Download CSV buttons.
  - Live MJPEG video.
  - Simple history table listing events as they arrive.

- **Behavior** (`main.js`):
  - `startRun()`:
    - POST `/start` with `{source, tz}`.
    - Opens SSE connection to `/events`.
    - Populates history table as `run_start` and events arrive.
  - `stopRun()`:
    - POST `/stop`, closes SSE.

**Why keep the Monitor tab simple**:

- It provides a reliable, low‑overhead view of the stream and events, ideal for quick checks and debugging.

### 6.2 Analytics tab

The Analytics tab is built to feel like an **industry‑grade dashboard** while still being lightweight.

- **Summary row**:
  - Cards showing:
    - Total events.
    - Per‑label counts (chips).
    - Time span (first→last event time).
    - Average score.
    - Most recent event label/time.
  - Computed in `recomputeSummary()` from the current `chartData`.

- **Score chart**:
  - Implemented with **Chart.js** (via CDN).
  - Scatter dataset for event scores.
  - Line dataset for threshold.
  - X‑axis: epoch seconds (linear), formatted to local time.
  - Y‑axis: score.
  - Tooltips show score, label, and component breakdown.
  - On click, selects corresponding timeline event.

- **Timeline strip**:
  - Horizontally scrolling row of event tiles.
  - Each tile:
    - Snapshot thumbnail (if available).
    - Label badge (different color per label).
    - Time and score.
  - Click → highlight that tile and center it; also used in conjunction with chart clicks.
  - Keyboard arrows (←/→) move across tiles for accessibility.

- **Label filters**:
  - Three checkboxes for each label type.
  - Affect both chart and timeline by filtering `chartData`.

### 6.3 Performance considerations

- **MAX_VISUAL_EVENTS = 150**:
  - At most 150 events are rendered in chart and timeline.
  - Historical fetch (`/events.json`) only uses last 150 events.
  - Live SSE updates trim older events once the limit is reached.
  - **Why**:
    - Prevents the UI from slowing down or stretching excessively during long runs.

- **Clamped heights**:
  - Chart canvas has `max-height: 260px`.
  - Timeline strip has `max-height: 140px`.
  - Ensures Analytics panel remains compact and scrollable instead of expanding indefinitely.

### 6.4 Live vs historical data flow

- On first open of Analytics:
  - `ensureAnalyticsInitialized()` → `loadAnalyticsData()` → `GET /events.json`.
  - Populates `chartData` and builds the summary, chart, and timeline.
- During a run:
  - The same SSE connection used by Monitor is reused:
    - `run_start` updates metadata.
    - Each event is appended to `chartData`, then summary/chart/timeline update.
  - This gives a unified live experience between Monitor and Analytics.

---

## 7. Parameter tuning and practical guidance

### 7.1 Sensitivity

- **More sensitive (detect more events)**:
  - Lower `threshold` (e.g., 0.05).
  - Lower `debounce` (e.g., 3).
  - Potential trade‑off: more false positives.

- **Less sensitive (fewer, stronger events)**:
  - Increase `threshold`.
  - Increase `debounce`.
  - Potential trade‑off: may miss subtle but real changes.

### 7.2 Semantic vs low‑level weighting

- Increase `cnn_weight`:
  - Stronger emphasis on semantic differences (object/scene changes).
  - Good when low‑level noise (e.g., compression) is high.

- Decrease or disable CNN (`--no-cnn`):
  - Lower CPU, rely primarily on low‑level metrics.
  - Good for constrained devices or when MobileNet features are unnecessary.

### 7.3 EMA window

- Shorter `window_seconds`:
  - Faster adaptation, more sensitive to fast shifts.
  - But can “forget” old baselines too quickly.

- Longer `window_seconds`:
  - Smoother baseline, better at detecting rare large changes.
  - Slower adaptation to genuinely new constant scenes.

---

## 8. Why this system design is robust

1. **Modularity**:
   - Detector core is isolated in `model.py` with a clean `step()` API.
   - Runtimes (CLI/web) and frontend are decoupled; only logs and SSE/JSON are shared.

2. **Graceful degradation**:
   - If `torch/torchvision` are unavailable, CNN is disabled and detector still works.
   - Timezone auto‑detection failures fall back without breaking runs.

3. **Auditability**:
   - Every event is logged with:
     - Time (epoch + ISO).
     - Components and labels.
     - Snapshots mapped via deterministic filenames.
   - CSV and JSON exports make it easy to review and share analyses.

4. **User experience**:
   - Monitor tab: minimal, predictable, no heavy JS.
   - Analytics tab: industry‑style visualizer with modern dark UI, but bounded complexity (event cap, fixed panel sizes).

5. **Extensibility**:
   - Easy to add new labels or tweak heuristics in `_categorize`.
   - Additional visualizations can be layered on top of `/events.json` without changing backends.
   - Multi‑camera or run‑ID support can be added by sharding `out/` and extending the API.

---

## 9. Summary

This project implements a **hybrid, real‑time domain shift detector** for video streams, combining:

- Low‑level image statistics (histogram, brightness, edges, SSIM).
- Optional semantic embeddings from MobileNetV2.
- A robust event logic with EMA, thresholds, debounce, and cooldown.

The system exposes:

- A **CLI** for automation and batch work.
- A **web UI** with:
  - Monitor tab (live view + history).
  - Analytics tab (summary cards, score chart, timeline with thumbnails).

Design decisions and parameter choices are made with:

- CPU efficiency.
- Interpretability.
- Ease of tuning.
- A modern, audit‑friendly UI in mind.

This `DESIGN.md` should give future engineers enough context to safely modify thresholds, add new kinds of visualization, swap in new models (e.g., MobileNetV3 or a small ViT), or extend outputs to new formats and storage backends.

---

## 10. End‑to‑end process flow

This section summarizes the **full lifecycle** from a user starting the system to events appearing in the visualizer and exports.

### 10.1 CLI flow

1. **User runs** `python main.py --source ... [flags]`.
2. `parse_args()`:
   - Resolves `source` (positional overrides `--source`).
   - Reads detection, snapshot, CNN, and timezone flags.
3. **Timezone resolution**:
   - Prefer `--tz` if provided.
   - Else, try `--place` (geocode → timezone).
   - Else, if `--auto-tz-ip`, infer from IP.
   - Else, fall back to local time.
4. **Initialize logging**:
   - Create `out/` and `out/snapshots/` if needed.
   - Open `out/events.jsonl` and write a `run_start` JSON line with config + timezone.
5. **Construct detector**:
   - Build `DetectorConfig` from args.
   - Create `DomainShiftDetector(cfg)`.
6. **Open capture**:
   - If `source` is a digit → webcam index.
   - If YouTube URL → resolve via `yt_dlp` then open.
   - Else → open as URL/path, with FFMPEG fallback.
7. **Frame loop** (until EOF, `q`/Esc, `max-frames`, or error):
   - Read a frame from OpenCV.
   - Call `det.step(frame, now=time.time())`.
   - If `event` is returned:
     - Convert `time` to `time_iso` using configured timezone.
     - Add `tz` field and write event JSON to `events.jsonl`.
     - If snapshots enabled:
       - Save per‑event snapshot as `<epoch>_<label>_<score>.jpg`.
   - If `--snapshot-interval` and enough time passed, save `*_periodic.jpg`.
   - If `--save-first` and not yet saved, save `*_start.jpg`.
   - If `--display`:
     - Draw overlay with `score`, `threshold`, `label`.
     - Show window; allow quitting with `q`/Esc or window close.
   - Check for console `q`/Esc via `msvcrt` on Windows.
8. **Shutdown**:
   - Release capture.
   - Close `events.jsonl`.
   - Destroy any OpenCV windows.

Result: The CLI run leaves behind a **JSONL log** and **snapshots** that can be inspected directly or via the web UI/analytics later.

### 10.2 Web UI – Monitor tab

1. **User runs** `python webapp.py` and opens `http://127.0.0.1:5000`.
2. `GET /`:
   - Renders `index.html` with `us_tzs` for the timezone dropdown.
   - Browser loads `static/main.js`.
3. User selects **source + timezone** and clicks **Start**:
   - `startRun()` sends `POST /start` with `{source, tz}` JSON.
4. `/start` handler:
   - Validates `source` and `tz` (must be from US whitelist or empty).
   - If a worker is already running:
     - Signals its `stop_evt` and joins the thread.
   - Creates new `WorkerConfig` with source, timezone, and default detection params.
   - Spawns a new `Thread(target=detection_worker, ...)`, sets `state.running = True`.
5. `detection_worker`:
   - Ensures `out/` and `out/snapshots/` exist.
   - Opens `out/events.jsonl`, writes a `run_start` header and saves it in `state.run_header`.
   - Broadcasts `run_start` over SSE (`state.broadcast(header)`).
   - Builds `DetectorConfig` (window=3.0 seconds) and `DomainShiftDetector`.
   - Opens capture using the same logic as CLI.
6. **Worker frame loop**:
   - Reads frames until `state.stop_evt` is set, capture fails, or EOF:
     - Calls `det.step(frame, now)`.
     - Builds `overlay` image:
       - Draws score/threshold/label.
       - Draws time with timezone.
     - Encodes overlay as JPEG and sets `state.last_jpeg` (for `/stream`).
     - On first frame, saves a `*_start.jpg` snapshot.
     - Every 10 seconds, saves a `*_periodic.jpg` snapshot.
     - If `event` is returned:
       - Enriches with `time_iso` and `tz` (from worker timezone).
       - Appends JSON line to `events.jsonl` and flushes.
       - Broadcasts event via `state.broadcast(event)` (SSE).
       - Saves per‑event snapshot `<epoch>_<label>_<score>.jpg`.
7. **Live video**:
   - Browser `<img id="video" src="/stream">` continuously pulls frames from `/stream`.
   - `/stream` responds with multipart MJPEG built from `state.last_jpeg`.
8. **Live history table**:
   - When `startRun()` succeeds, it opens an `EventSource('/events')`:
     - On `run_start`, it adds a header row to the history table.
     - On each event, it appends a new row with when/label/score/snapshot link.
9. **Stop**:
   - Clicking Stop triggers `POST /stop` → sets `state.stop_evt` and joins the worker thread.
   - SSE connection is closed by the frontend.

### 10.3 Web UI – Analytics tab

1. User clicks **Analytics** tab:
   - `main.js` hides `#monitorView`, shows `#analyticsView`.
   - On first open, `ensureAnalyticsInitialized()` calls `loadAnalyticsData()`.
2. `loadAnalyticsData()`:
   - Fetches `GET /events.json`.
   - Sets `runHeader` from `data.run` (if present).
   - Builds `chartData` from up to the last 150 events:
     - Adds unique `_id` to each event for timeline selection.
   - Calls `recomputeSummary()`, `buildChart()` (if first time), `updateChart()`, and `buildTimeline()`.
3. **Summary computation**:
   - `recomputeSummary()`:
     - Counts events per label.
     - Finds earliest and latest timestamps for time span.
     - Computes average score.
     - Finds most recent event label/time.
     - Writes these values into the summary cards.
4. **Chart rendering**:
   - `buildChart()` constructs a Chart.js scatter+line chart with:
     - Points: each event’s score at time `time` (seconds since epoch).
     - Line: threshold value over time.
     - Tooltips showing score, label, and component breakdown.
     - Click handler that sets the active timeline tile.
   - `updateChart()` rebuilds datasets based on current `chartData` and label filters.
5. **Timeline rendering**:
   - `buildTimeline()`:
     - Clears the timeline strip.
     - For each filtered event:
       - Creates a tile with snapshot thumbnail (if `snapshot_url`), label badge, time, and score.
       - Attaches click handler to call `setActiveTimelineItem()`.
6. **Live updates during a run**:
   - The same SSE connection used by Monitor keeps Analytics up to date:
     - For each event message:
       - Analytics appends a copy to `chartData` (keeping only the latest 150).
       - Recomputes summary, updates chart, and rebuilds timeline.
7. **Keyboard navigation**:
   - The timeline strip listens for ArrowLeft/ArrowRight:
     - Moves active tile left/right among rendered events.
     - Scrolls the new active tile into view.
8. **Exports**:
   - At any time, user can click **Download CSV**:
     - Browser navigates to `/download.csv`.
     - Server reads `events.jsonl`, resolves snapshots, and returns a CSV file.

Together, these flows ensure that from the moment a user hits **Start**, frames pass through the detector, events are logged and visualized, and evidence is captured in both **machine‑readable** (JSONL/CSV/JSON) and **visual** (snapshots, Analytics UI) forms.

