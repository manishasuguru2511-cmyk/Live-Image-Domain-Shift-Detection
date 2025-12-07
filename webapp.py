from __future__ import annotations
import threading
import queue
import time
import json
import os
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify, send_from_directory
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo

from model import DetectorConfig, DomainShiftDetector

try:
    import yt_dlp  
except Exception:
    yt_dlp = None

US_TZS = [
    "America/New_York",   # Eastern
    "America/Chicago",    # Central
    "America/Denver",     # Mountain
    "America/Los_Angeles",# Pacific
    "America/Phoenix",    # Arizona (no DST)
    "America/Anchorage",  # Alaska
    "Pacific/Honolulu",   # Hawaii
]

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "out"
LOG_PATH = OUT_DIR / "events.jsonl"
SNAP_DIR = OUT_DIR / "snapshots"

app = Flask(__name__)


def is_youtube(url: str) -> bool:
    u = url.lower()
    return ("youtube.com" in u) or ("youtu.be" in u)


def resolve_youtube_stream(url: str) -> str:
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")
    ydl_opts = {"quiet": True, "noplaylist": True, "format": "best", "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    direct = info.get("url")
    if direct:
        return direct
    fmts = info.get("formats") or []
    for f in reversed(fmts):
        prot = (f.get("protocol") or "").lower()
        if any(p in prot for p in ("https", "http", "m3u8")) and f.get("url"):
            return f["url"]
    for f in fmts:
        if f.get("url"):
            return f["url"]
    raise RuntimeError("Could not resolve a playable URL from YouTube")


def open_capture(source: str) -> cv2.VideoCapture:
    src = source.strip()
    
    # Handle webcam index
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index: {src}")
        return cap
    
    # Handle YouTube URLs
    if is_youtube(src):
        try:
            url = resolve_youtube_stream(src)
            print("Resolved YouTube stream URL")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open YouTube stream: {src}")
            return cap
        except Exception as e:
            print(f"Warning: failed to resolve YouTube URL: {e}")
            raise RuntimeError(f"Could not open YouTube URL: {src}. Error: {e}")
    
    # Handle local file paths
    file_path = Path(src)
    if file_path.exists() and file_path.is_file():
        # Absolute or relative path to local file
        abs_path = str(file_path.resolve())
        cap = cv2.VideoCapture(abs_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # Try without CAP_FFMPEG flag
            cap = cv2.VideoCapture(abs_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {abs_path}. Please check if the file exists and is a valid video format.")
        return cap
    
    # Handle URLs (HTTP/HTTPS/RTSP)
    if src.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video URL: {src}")
        return cap
    
    # Try as file path (might be relative or need resolution)
    try:
        # Try resolving as file path
        possible_path = Path(src).resolve()
        if possible_path.exists():
            cap = cv2.VideoCapture(str(possible_path), cv2.CAP_FFMPEG)
            if cap.isOpened():
                return cap
    except Exception:
        pass
    
    # Last attempt: try directly
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {src}. "
            "Please ensure it's a valid webcam index, file path, YouTube URL, or video URL."
        )
    return cap


@dataclass
class WorkerConfig:
    source: str
    tz_name: str
    threshold: float = 0.08
    debounce: int = 3
    cooldown: float = 1.0
    use_cnn: bool = True
    cnn_weight: float = 1.0


class ServerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.running = False
        self.cfg: Optional[WorkerConfig] = None
        self.tzinfo: Optional[ZoneInfo] = None
        self.last_jpeg: Optional[bytes] = None
        self.subscribers: List[queue.Queue[str]] = []
        self.run_header: Optional[dict] = None
        # Video metadata and progress
        self.video_fps: Optional[float] = None
        self.video_frame_count: Optional[int] = None
        self.video_duration: Optional[float] = None
        self.current_frame: int = 0
        self.processing_fps: float = 0.0
        self.is_live_source: bool = True

    def add_subscriber(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
            if self.run_header is not None:
                try:
                    q.put_nowait(json.dumps(self.run_header))
                except Exception:
                    pass
        return q

    def remove_subscriber(self, q: queue.Queue[str]) -> None:
        with self.lock:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    def broadcast(self, obj: dict) -> None:
        data = json.dumps(obj)
        with self.lock:
            for q in list(self.subscribers):
                try:
                    q.put_nowait(data)
                except Exception:
                    try:
                        self.subscribers.remove(q)
                    except Exception:
                        pass


state = ServerState()


def detection_worker(cfg: WorkerConfig, state: ServerState):
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        log_f = open(LOG_PATH, "w", encoding="utf-8")
        start_ts = time.time()
        tzinfo = ZoneInfo(cfg.tz_name) if cfg.tz_name else None
        state.tzinfo = tzinfo
        dt_header = datetime.fromtimestamp(start_ts, tz=tzinfo) if tzinfo else datetime.fromtimestamp(start_ts)
        header = {
            "type": "run_start",
            "time": start_ts,
            "time_iso": dt_header.isoformat(timespec="seconds"),
            "source": cfg.source,
            "threshold": cfg.threshold,
            "window": 3.0,
            "debounce": cfg.debounce,
            "cooldown": cfg.cooldown,
            "tz": cfg.tz_name or "local",
        }
        log_f.write(json.dumps(header) + "\n")
        log_f.flush()
        state.run_header = header
        state.broadcast(header)

        det_cfg = DetectorConfig(
            threshold=cfg.threshold,
            window_seconds=3.0,
            debounce=cfg.debounce,
            cooldown_seconds=cfg.cooldown,
            use_cnn=cfg.use_cnn,
            cnn_weight=cfg.cnn_weight,
        )
        det = DomainShiftDetector(det_cfg)

        try:
            cap = open_capture(cfg.source)
        except Exception as e:
            err = {"type": "error", "message": str(e)}
            state.broadcast(err)
            log_f.write(json.dumps(err) + "\n")
            log_f.close()
            return
        
        if not cap.isOpened():
            err = {"type": "error", "message": f"Could not open source: {cfg.source}"}
            state.broadcast(err)
            log_f.write(json.dumps(err) + "\n")
            log_f.close()
            return

        # Extract video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine if it's a live source (webcam/stream) or file
        is_live = frame_count <= 0 or fps <= 0
        duration = frame_count / fps if not is_live and fps > 0 else None
        
        with state.lock:
            state.video_fps = fps if fps > 0 else None
            state.video_frame_count = frame_count if frame_count > 0 else None
            state.video_duration = duration
            state.is_live_source = is_live
        
        # Broadcast video info
        video_info = {
            "type": "video_info",
            "fps": fps if fps > 0 else None,
            "frame_count": frame_count if frame_count > 0 else None,
            "duration": duration,
            "width": width,
            "height": height,
            "is_live": is_live,
        }
        state.broadcast(video_info)

        last_snapshot = 0.0
        first_saved = False
        frame_num = 0
        start_time = time.time()
        last_fps_time = start_time
        fps_frame_count = 0
        
        # For video files, maintain frame timing
        frame_delay = 1.0 / fps if fps > 0 and not is_live else 0
        last_frame_time = start_time

        while not state.stop_evt.is_set():
            # For video files, respect frame rate timing
            if not is_live and fps > 0:
                elapsed = time.time() - last_frame_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
            
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            
            frame_num += 1
            frame_read_time = time.time()
            
            # Use video timestamp for files, real-time for live sources
            if not is_live and fps > 0:
                video_time = frame_num / fps
                now = start_time + video_time
            else:
                now = frame_read_time
            
            event = det.step(frame, now=now)
            
            # Update performance metrics
            fps_frame_count += 1
            if frame_read_time - last_fps_time >= 1.0:
                processing_fps = fps_frame_count / (frame_read_time - last_fps_time)
                with state.lock:
                    state.processing_fps = processing_fps
                    state.current_frame = frame_num
                fps_frame_count = 0
                last_fps_time = frame_read_time
            
            last_frame_time = frame_read_time

            overlay = frame.copy()
            h, w = frame.shape[:2]
            
            # Enhanced overlay with more information
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            line_height = 25
            y_offset = 25
            
            # Background for text (semi-transparent)
            overlay_alpha = 0.7
            overlay_bg = overlay.copy()
            
            # Score and threshold (with color coding)
            score_color = (0, 255, 0) if det.last_score < det_cfg.threshold else (0, 0, 255)
            score_txt = f"Score: {det.last_score:.3f} | Threshold: {det_cfg.threshold:.3f}"
            cv2.putText(overlay, score_txt, (12, y_offset), font, font_scale, score_color, thickness, cv2.LINE_AA)
            y_offset += line_height
            
            # Label
            label_txt = f"Label: {det.last_label or 'N/A'}"
            cv2.putText(overlay, label_txt, (12, y_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y_offset += line_height
            
            # Time and timezone
            dt_disp = datetime.fromtimestamp(now, tz=tzinfo) if tzinfo else datetime.fromtimestamp(now)
            time_txt = dt_disp.strftime("%Y-%m-%d %H:%M:%S ") + (dt_disp.tzname() or "local")
            cv2.putText(overlay, time_txt, (12, y_offset), font, font_scale * 0.9, (200, 200, 200), thickness - 1, cv2.LINE_AA)
            y_offset += line_height
            
            # Video progress (for files)
            if not is_live and duration is not None:
                current_time = frame_num / fps if fps > 0 else 0
                progress_pct = (frame_num / frame_count * 100) if frame_count > 0 else 0
                progress_txt = f"Progress: {current_time:.1f}s / {duration:.1f}s ({progress_pct:.1f}%) | Frame: {frame_num}/{frame_count}"
                cv2.putText(overlay, progress_txt, (12, y_offset), font, font_scale * 0.85, (150, 200, 255), thickness - 1, cv2.LINE_AA)
                y_offset += line_height
            
            # Performance metrics
            with state.lock:
                proc_fps = state.processing_fps
            if proc_fps > 0:
                perf_txt = f"Processing: {proc_fps:.1f} FPS"
                cv2.putText(overlay, perf_txt, (12, y_offset), font, font_scale * 0.85, (100, 255, 100), thickness - 1, cv2.LINE_AA)

            ok2, jpeg = cv2.imencode('.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok2:
                state.last_jpeg = jpeg.tobytes()

            if not first_saved:
                try:
                    fname = f"{int(now)}_start.jpg"
                    cv2.imwrite(str(SNAP_DIR / fname), frame)
                except Exception:
                    pass
                first_saved = True

            if (now - last_snapshot) >= 10.0:
                try:
                    fname = f"{int(now)}_periodic.jpg"
                    cv2.imwrite(str(SNAP_DIR / fname), frame)
                except Exception:
                    pass
                last_snapshot = now

            if event is not None:
                event_dt = datetime.fromtimestamp(event["time"], tz=tzinfo) if tzinfo else datetime.fromtimestamp(event["time"])
                event["time_iso"] = event_dt.isoformat(timespec="seconds")
                event["tz"] = cfg.tz_name or "local"
                
                # Save snapshot first, then attach URL to event
                snap_name = None
                try:
                    fname = f"{int(event['time'])}_{event['label']}_{event['score']:.2f}.jpg"
                    snap_path = SNAP_DIR / fname
                    cv2.imwrite(str(snap_path), frame)
                    if snap_path.exists():
                        snap_name = fname
                except Exception:
                    pass
                
                # Add snapshot URL to event before broadcasting
                if snap_name:
                    event["snapshot"] = snap_name
                    event["snapshot_url"] = f"/snapshots/{snap_name}"
                else:
                    event["snapshot"] = ""
                    event["snapshot_url"] = ""
                
                log_f.write(json.dumps(event) + "\n")
                log_f.flush()
                state.broadcast(event)

        cap.release()
        log_f.close()
    except Exception as e:
        state.broadcast({"type": "error", "message": str(e)})
    finally:
        with state.lock:
            state.running = False
            state.thread = None
            state.stop_evt.clear()


@app.route("/")
def index():
    return render_template("index.html", us_tzs=US_TZS)


@app.post("/start")
def start():
    data = request.get_json(force=True, silent=True) or {}
    source = (data.get("source") or "").strip()
    tz = (data.get("tz") or "").strip()
    if not source:
        return jsonify({"ok": False, "error": "Missing source"}), 400
    if tz and tz not in US_TZS:
        return jsonify({"ok": False, "error": "Invalid timezone"}), 400

    with state.lock:
        if state.running:
            state.stop_evt.set()
    if state.thread is not None:
        state.thread.join(timeout=3)

    with state.lock:
        state.stop_evt.clear()
        state.running = True
        state.cfg = WorkerConfig(source=source, tz_name=tz)
        state.thread = threading.Thread(target=detection_worker, args=(state.cfg, state), daemon=True)
        state.thread.start()
    return jsonify({"ok": True})


@app.post("/stop")
def stop():
    with state.lock:
        state.stop_evt.set()
    if state.thread is not None:
        state.thread.join(timeout=3)
    return jsonify({"ok": True})


@app.get("/stream")
def stream():
    boundary = b"--frame"

    def gen():
        while True:
            if state.last_jpeg is not None:
                yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + state.last_jpeg + b"\r\n"
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/events")
def events():
    q = state.add_subscriber()

    def gen():
        try:
            while True:
                try:
                    item = q.get(timeout=0.5)
                except queue.Empty:
                    yield "data: {\"type\": \"ping\"}\n\n"
                    continue
                yield "data: " + item + "\n\n"
        finally:
            state.remove_subscriber(q)
    return Response(gen(), mimetype="text/event-stream")


@app.get("/snapshots/<path:filename>")
def get_snapshot(filename: str):
    return send_from_directory(SNAP_DIR, filename, as_attachment=False)


def _snapshot_name_for_event(ev: dict) -> Optional[str]:
    # First, check if snapshot_url is already in the event
    snapshot_url = ev.get("snapshot_url", "")
    if snapshot_url:
        # Extract filename from URL like "/snapshots/filename.jpg"
        if snapshot_url.startswith("/snapshots/"):
            snap_name = snapshot_url.replace("/snapshots/", "")
            snap_path = SNAP_DIR / snap_name
            if snap_path.exists():
                return snap_name
    
    # Check if snapshot field is already set
    snap_name = ev.get("snapshot", "")
    if snap_name:
        snap_path = SNAP_DIR / snap_name
        if snap_path.exists():
            return snap_name
    
    try:
        t = int(ev.get("time", 0))
        label = (ev.get("label") or "").replace("/", "-")
        score = float(ev.get("score", 0.0))
        
        # Strategy 1: Exact match
        name = f"{t}_{label}_{score:.2f}.jpg"
        p = SNAP_DIR / name
        if p.exists():
            return name
        
        # Strategy 2: Match by time prefix and label
        prefix = f"{t}_"
        if label:
            # Try to find snapshot with matching label
            for cand in sorted(SNAP_DIR.glob(prefix + "*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True):
                if label in cand.name:
                    return cand.name
        
        # Strategy 3: Any snapshot with matching time prefix (take most recent)
        candidates = sorted(SNAP_DIR.glob(prefix + "*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0].name
        
        # Strategy 4: Try matching by approximate time (within 5 seconds)
        if t > 0:
            for offset in range(1, 6):  # Check ±1 to ±5 seconds
                for time_offset in [t + offset, t - offset]:
                    prefix_offset = f"{int(time_offset)}_"
                    candidates = sorted(SNAP_DIR.glob(prefix_offset + "*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
                    if candidates:
                        return candidates[0].name
    except Exception:
        pass
    return None


@app.get("/download.csv")
def download_csv():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "run_start":
                    rows.append({
                        "type": "run_start",
                        "time": ev.get("time"),
                        "time_iso": ev.get("time_iso"),
                        "tz": ev.get("tz"),
                        "source": ev.get("source"),
                        "threshold": ev.get("threshold"),
                        "debounce": ev.get("debounce"),
                        "cooldown": ev.get("cooldown"),
                        "window": ev.get("window"),
                        "snapshot": "",
                        "snapshot_url": "",
                        "label": "",
                        "score": "",
                    })
                    continue
                snap_name = _snapshot_name_for_event(ev) or ""
                row = {
                    "type": "event",
                    "time": ev.get("time"),
                    "time_iso": ev.get("time_iso"),
                    "tz": ev.get("tz"),
                    "label": ev.get("label"),
                    "score": ev.get("score"),
                    "hist": ev.get("components", {}).get("hist"),
                    "brightness": ev.get("components", {}).get("brightness"),
                    "edge": ev.get("components", {}).get("edge"),
                    "ssim": ev.get("components", {}).get("ssim"),
                    "cnn": ev.get("components", {}).get("cnn"),
                    "mmd": ev.get("components", {}).get("mmd"),
                    "mahalanobis": ev.get("components", {}).get("mahalanobis"),
                    "snapshot": snap_name,
                    "snapshot_url": (f"/snapshots/{snap_name}" if snap_name else ""),
                }
                rows.append(row)

    buf = io.StringIO()
    fieldnames = [
        "type","time","time_iso","tz","source","threshold","debounce","cooldown","window",
        "label","score","hist","brightness","edge","ssim","cnn","mmd","mahalanobis",
        "snapshot","snapshot_url"
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    data = buf.getvalue()
    buf.close()
    return Response(
        data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=events.csv"},
    )


@app.get("/status")
def status():
    with state.lock:
        return jsonify({
            "ok": True,
            "running": state.running,
            "fps": state.video_fps,
            "frame_count": state.video_frame_count,
            "duration": state.video_duration,
            "current_frame": state.current_frame,
            "processing_fps": state.processing_fps,
            "is_live": state.is_live_source,
            "progress": (state.current_frame / state.video_frame_count * 100) if state.video_frame_count and state.video_frame_count > 0 else None,
        })


@app.get("/events.json")
def events_json():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    run_header = None
    events: List[dict] = []

    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "run_start":
                    run_header = ev
                    continue
                snap_name = _snapshot_name_for_event(ev) or ""
                ev_out = dict(ev)
                ev_out["snapshot"] = snap_name
                ev_out["snapshot_url"] = f"/snapshots/{snap_name}" if snap_name else ""
                events.append(ev_out)

    payload = {
        "ok": True,
        "run": run_header,
        "events": events,
    }
    return jsonify(payload)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
