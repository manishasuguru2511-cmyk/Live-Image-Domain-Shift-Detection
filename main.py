import argparse
import json
import time
from pathlib import Path
import cv2
from datetime import datetime
from zoneinfo import ZoneInfo
import urllib.request
import urllib.parse
try:
    import msvcrt
except Exception:
    msvcrt = None

from model import DetectorConfig, DomainShiftDetector

try:
    import yt_dlp  
except Exception:
    yt_dlp = None

def parse_args():
    p = argparse.ArgumentParser(
        description="Time stamp using light CNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video file/RTSP path/YouTube URL or webcam index (e.g., 0)",
    )
    p.add_argument(
        "pos_source",
        nargs="?",
        default=None,
        help="Optional positional source; if provided, overrides --source",
    )
    p.add_argument("--threshold", type=float, default=0.6, help="Alert threshold")
    p.add_argument("--window", type=float, default=3.0, help="EMA time window (sec)")
    p.add_argument("--debounce", type=int, default=8, help="Frames above threshold to trigger event")
    p.add_argument("--cooldown", type=float, default=1.0, help="Minimum seconds between events")
    p.add_argument("--out", type=str, default="out", help="Output directory for logs and snapshots")
    p.add_argument("--display", action="store_true", help="Show video window with overlays")
    p.add_argument("--snapshots", action="store_true", help="Save snapshot images for each event")
    p.add_argument("--no-snapshots", dest="snapshots", action="store_false")
    p.set_defaults(snapshots=True)
    p.add_argument("--snapshot-interval", type=float, default=0.0, help="Also save snapshot every N seconds (0=off)")
    p.add_argument("--save-first", action="store_true", help="Save a snapshot for the first frame")
    p.add_argument("--use-cnn", action="store_true", help="Use MobileNetV2 features for ML distance (default on)")
    p.add_argument("--no-cnn", dest="use_cnn", action="store_false")
    p.set_defaults(use_cnn=True)
    p.add_argument("--cnn-weight", type=float, default=1.0, help="Weight for CNN distance in score")
    p.add_argument("--tz", type=str, default="", help="IANA timezone (e.g., America/New_York). Empty=local time")
    p.add_argument("--place", type=str, default="", help="Camera location string (e.g., 'Ocean City, MD') to auto-detect tz")
    p.add_argument("--auto-tz-ip", action="store_true", help="Infer timezone from your public IP (no extra deps)")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = infinite)")
    args = p.parse_args()
    if args.pos_source is not None:
        args.source = args.pos_source
    return args

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

def infer_tz_from_ip(timeout: float = 3.0) -> str:
    endpoints = (
        "https://worldtimeapi.org/api/ip",      
        "https://ipapi.co/json/",
        "http://ip-api.com/json",               
    )
    for url in endpoints:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.load(resp)
            tz = (
                data.get("timezone")
                or data.get("timeZone")
                or data.get("tz")
            )
            if isinstance(tz, str) and tz:
                return tz
        except Exception:
            continue
    return ""

def geocode_place_to_latlon(place: str, timeout: float = 5.0):
    if not place:
        return None
    try:
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({
            "q": place,
            "format": "json",
            "limit": 1,
        })
        req = urllib.request.Request(url, headers={
            "User-Agent": "bonds-cli/0.1 (timezone lookup via Nominatim)",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
        if isinstance(payload, list) and payload:
            item = payload[0]
            lat = float(item.get("lat"))
            lon = float(item.get("lon"))
            return (lat, lon)
    except Exception:
        return None
    return None

def infer_tz_from_place(place: str, timeout: float = 5.0) -> str:
    coords = geocode_place_to_latlon(place, timeout=timeout)
    if not coords:
        return ""
    lat, lon = coords
    try:
        url = "https://timeapi.io/api/TimeZone/coordinate?" + urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
        })
        req = urllib.request.Request(url, headers={"User-Agent": "bonds-cli/0.1 (timezone lookup)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        tz = data.get("timeZone") or data.get("timezone") or data.get("tz") or data.get("name")
        if isinstance(tz, str) and tz:
            return tz
    except Exception:
        return ""
    return ""

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
    from pathlib import Path
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
            "Ensure it's a valid webcam index, file path, YouTube URL, or video URL."
        )
    return cap

def main():
    args = parse_args()

    out_dir = Path(args.out)
    snap_dir = out_dir / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "events.jsonl"
    log_f = open(log_path, "w", encoding="utf-8")
    start_ts = time.time()
    tz_name = args.tz.strip() if args.tz else ""
    if not tz_name and getattr(args, "place", ""):
        tz_name = infer_tz_from_place(args.place) or ""
    if not tz_name and getattr(args, "auto_tz_ip", False):
        tz_name = infer_tz_from_ip() or ""
    tzinfo = None
    if tz_name:
        try:
            tzinfo = ZoneInfo(tz_name)
        except Exception:
            print(f"Warning: unknown timezone '{tz_name}', using local time.")
            tz_name = ""
    dt_header = datetime.fromtimestamp(start_ts, tz=tzinfo) if tzinfo else datetime.fromtimestamp(start_ts)
    log_f.write(
        json.dumps(
            {
                "type": "run_start",
                "time": start_ts,
                "time_iso": dt_header.isoformat(timespec="seconds"),
                "source": args.source,
                "threshold": args.threshold,
                "window": args.window,
                "debounce": args.debounce,
                "cooldown": args.cooldown,
                "tz": tz_name or "local",
            }
        )
        + "\n"
    )
    log_f.flush()
    print(f"Output folder: {out_dir.resolve()}")
    print(f"Logging to: {log_path.resolve()}")

    cfg = DetectorConfig(
        threshold=args.threshold,
        window_seconds=args.window,
        debounce=args.debounce,
        cooldown_seconds=args.cooldown,
        use_cnn=args.use_cnn,
        cnn_weight=args.cnn_weight,
    )
    det = DomainShiftDetector(cfg)

    try:
        cap = open_capture(args.source)
    except Exception as e:
        print(f"Error: {e}")
        return 2
    
    if not cap.isOpened():
        print(f"Error: could not open source: {args.source}")
        return 2

    print("Press 'q' to quit (focus video window) or press 'q' in the console.")
    frame_count = 0
    last_print = time.time()
    printed_header = False

    last_snapshot_time = 0.0
    first_saved = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            now = time.time()
            event = det.step(frame, now=now)

            if event is not None:
                event_dt = datetime.fromtimestamp(event["time"], tz=tzinfo) if tzinfo else datetime.fromtimestamp(event["time"])
                event["time_iso"] = event_dt.isoformat(timespec="seconds")
                event["tz"] = tz_name or "local"
                log_f.write(json.dumps(event) + "\n")
                log_f.flush()

                if args.snapshots:
                    fname = f"{int(event['time'])}_{event['label']}_{event['score']:.2f}.jpg"
                    cv2.imwrite(str(snap_dir / fname), frame)
                    last_snapshot_time = now

                print(
                    f"EVENT {event['time_iso']} | label={event['label']} | score={event['score']:.3f} | comp={event['components']}"
                )

            if args.snapshots and args.snapshot_interval > 0 and (now - last_snapshot_time) >= args.snapshot_interval:
                fname = f"{int(now)}_periodic.jpg"
                cv2.imwrite(str(snap_dir / fname), frame)
                last_snapshot_time = now

            if args.snapshots and args.save_first and not first_saved:
                fname = f"{int(now)}_start.jpg"
                cv2.imwrite(str(snap_dir / fname), frame)
                first_saved = True

            if args.display:
                overlay = frame.copy()
                txt = f"score={det.last_score:.2f} thr={cfg.threshold:.2f} label={det.last_label or '-'}"
                cv2.putText(
                    overlay,
                    txt,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if det.last_score < cfg.threshold else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Domain Shift", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                try:
                    if cv2.getWindowProperty("Domain Shift", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    pass

            if msvcrt is not None and msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"q", b"Q", b"\x1b"):
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break

            if not printed_header:
                print("time | score | hist/bright/edge/ssim")
                printed_header = True
            if time.time() - last_print >= 2.0:
                c = det.last_components or {k: 0 for k in ("hist", "brightness", "edge", "ssim")}
                display_dt = datetime.fromtimestamp(now, tz=tzinfo) if tzinfo else datetime.fromtimestamp(now)
                tz_label = (display_dt.tzname() or tz_name or "local")
                print(
                    f".. {display_dt.strftime('%H:%M:%S')} {tz_label} | {det.last_score:.2f} | {c.get('hist',0):.2f}/{c.get('brightness',0):.2f}/{c.get('edge',0):.2f}/{c.get('ssim',0):.2f}"
                )
                last_print = time.time()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        log_f.close()
        if args.display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
