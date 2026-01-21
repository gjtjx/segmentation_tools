import argparse
import base64
import colorsys
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import requests
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


COLOR_CLASSES: List[str] = [
    "white",
    "orange_yellow",
    "red",
    "blue",
    "green",
    "purple",
    "black",
    "pink",
    "brown",
]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    obb8: Optional[Tuple[float, float, float, float, float, float, float, float]] = None
    xywha: Optional[Tuple[float, float, float, float, float]] = None

    def area(self) -> float:
        w = max(0.0, self.x2 - self.x1)
        h = max(0.0, self.y2 - self.y1)
        return w * h


def iter_images(input_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in input_dir.rglob("*"):
            if not p.is_file():
                continue
            if not p.name.startswith("A_"):
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            yield p
    else:
        for p in input_dir.iterdir():
            if not p.is_file():
                continue
            if not p.name.startswith("A_"):
                continue
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            yield p


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def iter_video_frames(video_path: Path, frame_interval: int, start_frame: int, max_frames: Optional[int]):
    """Yield (frame_index, PIL.Image) for sampled frames."""
    if frame_interval <= 0:
        raise ValueError("frame_interval must be >= 1")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    try:
        import cv2
    except Exception as e:
        raise RuntimeError(
            "OpenCV is required for video input. Install opencv-python (and numpy)."
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        frame_index = start_frame
        yielded = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if (frame_index - start_frame) % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_index, Image.fromarray(frame_rgb)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break

            frame_index += 1
    finally:
        cap.release()


def pil_to_data_url_png(img: Image.Image) -> str:
    # Always send PNG to keep decoding consistent.
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def split_2x2(img: Image.Image) -> List[Tuple[Tuple[int, int, int, int], Image.Image]]:
    w, h = img.size
    mx = w // 2
    my = h // 2

    tiles: List[Tuple[Tuple[int, int, int, int], Image.Image]] = []
    boxes = [
        (0, 0, mx, my),
        (mx, 0, w, my),
        (0, my, mx, h),
        (mx, my, w, h),
    ]
    for (l, t, r, b) in boxes:
        # Guard against degenerate images
        if r <= l or b <= t:
            continue
        tiles.append(((l, t, r, b), img.crop((l, t, r, b))))
    return tiles


def call_segment_api(
    session: requests.Session,
    endpoint: str,
    tile_img: Image.Image,
    prompt_text: str,
    timeout_s: float,
    confidence_threshold: float,
    enable_postprocess: bool,
    retries: int,
    retry_sleep_s: float,
) -> dict:
    payload = {
        "image": pil_to_data_url_png(tile_img),
        "prompt_text": prompt_text,
        "mode": "full",
        "confidence_threshold": confidence_threshold,
        "enable_postprocess": enable_postprocess,
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = session.post(endpoint, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(retry_sleep_s)

    raise RuntimeError(f"POST {endpoint} failed after {retries + 1} attempts: {last_err}")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(boxes: Sequence[Box], iou_thresh: float) -> List[Box]:
    if iou_thresh <= 0:
        return list(boxes)

    remaining = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: List[Box] = []

    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [b for b in remaining if iou(best, b) < iou_thresh]

    return kept


def decode_data_url_image(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    from io import BytesIO

    return Image.open(BytesIO(raw))


def order_quad_points_clockwise(pts: "list[list[float]]") -> Tuple[float, float, float, float, float, float, float, float]:
    """Order 4 points as (tl, tr, br, bl) for stable exports."""
    # pts: 4x2
    # Based on sum and diff heuristics.
    import numpy as np

    arr = np.array(pts, dtype=float)
    s = arr.sum(axis=1)
    diff = (arr[:, 0] - arr[:, 1])

    tl = arr[int(s.argmin())]
    br = arr[int(s.argmax())]
    tr = arr[int(diff.argmax())]
    bl = arr[int(diff.argmin())]

    return (float(tl[0]), float(tl[1]), float(tr[0]), float(tr[1]), float(br[0]), float(br[1]), float(bl[0]), float(bl[1]))


def try_compute_obbs_from_tile_mask(
    mask_data_url: Optional[str],
    box_xyxy: Sequence[float],
    tile_left: int,
    tile_top: int,
    full_w: int,
    full_h: int,
) -> Tuple[
    Optional[Tuple[float, float, float, float, float, float, float, float]],
    Optional[Tuple[float, float, float, float, float]],
]:
    if not mask_data_url:
        return None, None

    try:
        import numpy as np
        import cv2
    except Exception:
        return None, None

    try:
        mask_img = decode_data_url_image(mask_data_url).convert("RGBA")
        mask_arr = np.array(mask_img)
        alpha = mask_arr[:, :, 3]
        ys, xs = np.where(alpha > 0)
        if xs.size < 20:
            return None, None

        x1_t, y1_t, x2_t, y2_t = [int(v) for v in box_xyxy]
        full_xs = xs.astype(np.float32) + float(tile_left + x1_t)
        full_ys = ys.astype(np.float32) + float(tile_top + y1_t)

        pts = np.stack([full_xs, full_ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle)
        quad = cv2.boxPoints(rect)  # 4x2

        # Clamp to image bounds
        quad[:, 0] = np.clip(quad[:, 0], 0.0, float(full_w))
        quad[:, 1] = np.clip(quad[:, 1], 0.0, float(full_h))

        obb8 = order_quad_points_clockwise(quad.tolist())

        # Build xywha in full-image pixel coords
        (cx, cy), (rw, rh), angle = rect
        cx = float(clamp(float(cx), 0.0, float(full_w)))
        cy = float(clamp(float(cy), 0.0, float(full_h)))
        rw = float(max(0.0, rw))
        rh = float(max(0.0, rh))
        angle = float(angle)  # degrees from OpenCV, typically in (-90, 0]

        # Normalize so w>=h for more consistent downstream handling
        if rw < rh:
            rw, rh = rh, rw
            angle += 90.0
        # Wrap angle to [-90, 90)
        while angle >= 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0

        xywha = (cx, cy, rw, rh, angle)
        return obb8, xywha
    except Exception:
        return None, None


def estimate_mean_rgb_from_masked_region(region_rgb: Image.Image, mask_rgba: Image.Image) -> Optional[Tuple[float, float, float]]:
    """Estimate mean RGB in region using mask alpha>0.

    region_rgb: RGB image cropped to bbox region.
    mask_rgba: RGBA (or RGB/L) mask image for same bbox region.
    """
    region_rgb = region_rgb.convert("RGB")
    mask_rgba = mask_rgba.convert("RGBA")

    w1, h1 = region_rgb.size
    w2, h2 = mask_rgba.size
    if (w1, h1) != (w2, h2):
        # Best-effort fallback: resize mask to region
        mask_rgba = mask_rgba.resize((w1, h1), resample=Image.NEAREST)

    region_pixels = list(region_rgb.getdata())
    alpha = [a for (_, _, _, a) in mask_rgba.getdata()]

    r_sum = 0
    g_sum = 0
    b_sum = 0
    cnt = 0
    for (r, g, b), a in zip(region_pixels, alpha):
        if a > 0:
            r_sum += r
            g_sum += g
            b_sum += b
            cnt += 1

    if cnt < 10:
        return None

    return (r_sum / cnt, g_sum / cnt, b_sum / cnt)


def classify_color(mean_rgb: Tuple[float, float, float]) -> str:
    r, g, b = mean_rgb
    r1, g1, b1 = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r1, g1, b1)  # h in [0,1)
    h_deg = (h * 360.0) % 360.0

    # Heuristics tuned for bright, solid-colored holds.
    if v < 0.20:
        return "black"
    if s < 0.18 and v > 0.75:
        return "white"
    if 15.0 <= h_deg < 60.0:
        # Brown vs orange/yellow split by brightness
        if v < 0.55:
            return "brown"
        return "orange_yellow"
    if h_deg < 15.0 or h_deg >= 345.0:
        # Red vs pink: pink is lighter and a bit less saturated
        if v > 0.65 and s < 0.65:
            return "pink"
        return "red"
    if 60.0 <= h_deg < 160.0:
        return "green"
    if 160.0 <= h_deg < 260.0:
        return "blue"
    if 260.0 <= h_deg < 330.0:
        return "purple"
    if 330.0 <= h_deg < 345.0:
        return "pink"

    # Fallback based on saturation/value
    if v > 0.70 and s < 0.35:
        return "white"
    return "purple"


def color_name_to_class_id(color_name: str) -> int:
    try:
        return COLOR_CLASSES.index(color_name)
    except ValueError:
        return 0


def prompts_for_color(color_name: str, template: str) -> List[str]:
    """Return one or more prompt strings for a given color class."""
    def fmt(color_word: str) -> str:
        if "{color}" not in template:
            # Back-compat: if user passes a fixed string, just append color
            return f"{template} {color_word}".strip()
        return template.format(color=color_word)

    if color_name == "orange_yellow":
        prompts: List[str] = []
        # Prefer grammatically correct article for the default template
        if template.strip() == "a {color} climbing hold":
            prompts.append("an orange climbing hold")
            prompts.append("a yellow climbing hold")
        else:
            prompts.append(fmt("orange"))
            prompts.append(fmt("yellow"))
        return prompts

    # Use natural words for prompt; keep underscores only for class naming.
    return [fmt(color_name.replace("_", " "))]


def tile_box_to_full_box(
    tile_box_xyxy: Sequence[float],
    tile_left: int,
    tile_top: int,
    full_w: int,
    full_h: int,
    score: float,
    class_id: int,
    obb8: Optional[Tuple[float, float, float, float, float, float, float, float]],
    xywha: Optional[Tuple[float, float, float, float, float]],
) -> Box:
    # Server returns [x_min, y_min, x_max, y_max] where x_max/y_max are pixel indices (inclusive).
    # Convert to continuous coords with x2/y2 as exclusive for stable math.
    x1_t, y1_t, x2_t, y2_t = [float(v) for v in tile_box_xyxy]

    x1 = x1_t + tile_left
    y1 = y1_t + tile_top

    # Inclusive -> exclusive
    x2 = (x2_t + 1.0) + tile_left
    y2 = (y2_t + 1.0) + tile_top

    x1 = clamp(x1, 0.0, float(full_w))
    y1 = clamp(y1, 0.0, float(full_h))
    x2 = clamp(x2, 0.0, float(full_w))
    y2 = clamp(y2, 0.0, float(full_h))

    # Ensure valid ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return Box(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=float(score),
        class_id=int(class_id),
        obb8=obb8,
        xywha=xywha,
    )


def box_to_yolo_line(box: Box, class_id: int, img_w: int, img_h: int, include_score: bool) -> str:
    bw = max(0.0, box.x2 - box.x1)
    bh = max(0.0, box.y2 - box.y1)
    cx = box.x1 + bw / 2.0
    cy = box.y1 + bh / 2.0

    # Normalize
    cxn = cx / img_w if img_w > 0 else 0.0
    cyn = cy / img_h if img_h > 0 else 0.0
    bwn = bw / img_w if img_w > 0 else 0.0
    bhn = bh / img_h if img_h > 0 else 0.0

    if include_score:
        return f"{class_id} {cxn:.6f} {cyn:.6f} {bwn:.6f} {bhn:.6f} {box.score:.6f}"

    return f"{class_id} {cxn:.6f} {cyn:.6f} {bwn:.6f} {bhn:.6f}"


def box_to_yolo_obb8_line(box: Box, img_w: int, img_h: int) -> Optional[str]:
    if box.obb8 is None:
        return None
    x1, y1, x2, y2, x3, y3, x4, y4 = box.obb8
    if img_w <= 0 or img_h <= 0:
        return None
    vals = [
        x1 / img_w,
        y1 / img_h,
        x2 / img_w,
        y2 / img_h,
        x3 / img_w,
        y3 / img_h,
        x4 / img_w,
        y4 / img_h,
    ]
    vals = [max(0.0, min(1.0, v)) for v in vals]
    return (
        f"{int(box.class_id)} "
        f"{vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f} "
        f"{vals[4]:.6f} {vals[5]:.6f} {vals[6]:.6f} {vals[7]:.6f}"
    )


def box_to_yolo_xywha_line(box: Box, img_w: int, img_h: int, angle_unit: str) -> Optional[str]:
    if box.xywha is None:
        return None
    if img_w <= 0 or img_h <= 0:
        return None

    cx, cy, w, h, angle_deg = box.xywha
    cxn = max(0.0, min(1.0, cx / img_w))
    cyn = max(0.0, min(1.0, cy / img_h))
    wn = max(0.0, min(1.0, w / img_w))
    hn = max(0.0, min(1.0, h / img_h))

    if angle_unit == "rad":
        angle = angle_deg * math.pi / 180.0
    else:
        angle = angle_deg

    return f"{int(box.class_id)} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f} {angle:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Traverse A_* images, split into 2x2 tiles, call SAM3 service, and export YOLO labels."
    )
    parser.add_argument("input_path", type=str, help="Input folder, image file, or video file")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://192.168.10.12:5002/api/segment/image",
        help="Segment API endpoint (default: http://192.168.10.12:5002/api/segment/image)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a climbing hold",
        help="Text prompt (default: a climbing hold)",
    )
    parser.add_argument("--recursive", action="store_true", default=True, help="Recurse into subfolders (default: on)")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Do not recurse")
    parser.add_argument("--out-dir", type=str, default="yolo_labels", help="Output folder for YOLO .txt files")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id to write (default: 0)")
    parser.add_argument(
        "--color-mode",
        choices=["single", "mask", "prompt"],
        default="single",
        help="single: force all detections to --class-id; mask: classify hold color from returned mask; prompt: run per-color prompts and merge by NMS",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="a {color} climbing hold",
        help="Prompt template used by --color-mode prompt. Must include {color} (default: 'a {color} climbing hold')",
    )
    parser.add_argument(
        "--yolo-format",
        choices=["bbox", "obb8", "xywha"],
        default="bbox",
        help="bbox: axis-aligned YOLO (class cx cy w h). obb8: oriented box 4 points (class x1 y1 ... x4 y4) normalized. xywha: (class cx cy w h angle) with cx/cy/w/h normalized",
    )
    parser.add_argument(
        "--angle-unit",
        choices=["deg", "rad"],
        default="deg",
        help="Angle unit for --yolo-format xywha (default: deg)",
    )
    parser.add_argument(
        "--write-classes",
        action="store_true",
        default=False,
        help="Write classes.txt into out-dir (useful with --color-mode mask)",
    )
    parser.add_argument("--score-thr", type=float, default=0.0, help="Filter detections by score (default: 0.0)")
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.0,
        help="Optional NMS IoU threshold across tiles in full-image coords (0 disables; suggested 0.5)",
    )
    parser.add_argument(
        "--include-score",
        action="store_true",
        default=False,
        help="Append score as 6th column (non-standard YOLO)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds (default: 120)")
    parser.add_argument("--retries", type=int, default=2, help="Retries per tile (default: 2)")
    parser.add_argument("--retry-sleep", type=float, default=1.0, help="Seconds between retries (default: 1.0)")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="confidence_threshold passed to server (default: 0.5)",
    )
    parser.add_argument(
        "--disable-postprocess",
        action="store_true",
        default=False,
        help="Set enable_postprocess=false on server",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not send requests; only list images that would be processed",
    )

    # Video options
    parser.add_argument("--frame-interval", type=int, default=1, help="For video: sample every N frames (default: 1)")
    parser.add_argument("--start-frame", type=int, default=0, help="For video: start frame index (default: 0)")
    parser.add_argument("--max-frames", type=int, default=None, help="For video: maximum sampled frames (default: unlimited)")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        default=True,
        help="For video: save sampled frames as images next to labels (default: on)",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_false",
        dest="save_frames",
        help="For video: do not save sampled frames",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise SystemExit(f"input_path not found: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.color_mode in {"mask", "prompt"} and args.write_classes:
        (out_dir / "classes.txt").write_text("\n".join(COLOR_CLASSES) + "\n", encoding="utf-8")

    # Build processing list: images or video frames
    items: List[Tuple[str, Image.Image]] = []
    if is_video_file(input_path):
        if args.dry_run:
            print(f"Video input: {input_path}")
            print(f"Would sample every {args.frame_interval} frames starting at {args.start_frame}")
            return 0
        for frame_idx, frame_img in iter_video_frames(
            input_path,
            frame_interval=int(args.frame_interval),
            start_frame=int(args.start_frame),
            max_frames=(int(args.max_frames) if args.max_frames is not None else None),
        ):
            key = f"{input_path.stem}_frame{frame_idx:06d}"
            items.append((key, frame_img))

        if not items:
            print(f"No frames read from video: {input_path}")
            return 0
        print(f"Sampled {len(items)} frames from video")
    else:
        # Single image file
        if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTS:
            if args.dry_run:
                print(input_path)
                return 0
            try:
                items.append((input_path.stem, Image.open(input_path).convert("RGB")))
            except Exception as e:
                raise SystemExit(f"Failed to open image: {input_path}: {e}")
        else:
            # Folder of images
            if not input_path.is_dir():
                raise SystemExit(f"input_path must be a folder, image, or video: {input_path}")
            img_paths = sorted(iter_images(input_path, recursive=args.recursive))
            if not img_paths:
                print(f"No A_* images found in: {input_path}")
                return 0

            print(f"Found {len(img_paths)} images")
            if args.dry_run:
                for p in img_paths:
                    print(p)
                return 0

            for p in img_paths:
                try:
                    items.append((p.stem, Image.open(p).convert("RGB")))
                except Exception as e:
                    print(f"[WARN] Failed to open {p}: {e}")
                    continue

    session = requests.Session()

    total_boxes = 0
    for idx, (item_key, img) in enumerate(items, start=1):
        full_w, full_h = img.size
        tiles = split_2x2(img)

        all_boxes: List[Box] = []
        for (l, t, r, b), tile_img in tiles:
            # In prompt mode, call the service once per color prompt; otherwise call once.
            if args.color_mode == "prompt":
                color_prompts: List[Tuple[int, str]] = []
                for cname in COLOR_CLASSES:
                    cid = color_name_to_class_id(cname)
                    for ptxt in prompts_for_color(cname, args.prompt_template):
                        color_prompts.append((cid, ptxt))
            else:
                color_prompts = [(int(args.class_id), args.prompt)]

            for cid, prompt_text in color_prompts:
                try:
                    resp = call_segment_api(
                        session=session,
                        endpoint=args.endpoint,
                        tile_img=tile_img,
                        prompt_text=prompt_text,
                        timeout_s=args.timeout,
                        confidence_threshold=float(args.confidence_threshold),
                        enable_postprocess=not bool(args.disable_postprocess),
                        retries=int(args.retries),
                        retry_sleep_s=float(args.retry_sleep),
                    )
                except Exception as e:
                    print(f"[WARN] Request failed for {item_key} tile ({l},{t},{r},{b}) prompt='{prompt_text}': {e}")
                    continue

                if not isinstance(resp, dict) or not resp.get("success", False):
                    print(f"[WARN] Server returned failure for {item_key} tile ({l},{t},{r},{b}) prompt='{prompt_text}': {resp}")
                    continue

                results = resp.get("results") or []
                for det in results:
                    score = float(det.get("score", 0.0))
                    if score < float(args.score_thr):
                        continue
                    box_xyxy = det.get("box")
                    if not box_xyxy or len(box_xyxy) != 4:
                        continue

                    class_id = int(cid)
                    obb8 = None
                    xywha = None
                    if args.yolo_format in {"obb8", "xywha"}:
                        obb8, xywha = try_compute_obbs_from_tile_mask(
                            mask_data_url=det.get("mask"),
                            box_xyxy=box_xyxy,
                            tile_left=l,
                            tile_top=t,
                            full_w=full_w,
                            full_h=full_h,
                        )
                    if args.color_mode == "mask":
                        try:
                            mask_data = det.get("mask")
                            if mask_data:
                                # bbox coords are inclusive in tile space; build region crop using exclusive right/bottom
                                x1_t, y1_t, x2_t, y2_t = [int(v) for v in box_xyxy]
                                region = tile_img.crop((x1_t, y1_t, x2_t + 1, y2_t + 1))
                                mask_img = decode_data_url_image(mask_data)
                                mean_rgb = estimate_mean_rgb_from_masked_region(region, mask_img)
                                if mean_rgb is not None:
                                    color_name = classify_color(mean_rgb)
                                    class_id = color_name_to_class_id(color_name)
                        except Exception:
                            # Fall back to provided class if any decode/classify issue occurs
                            class_id = int(cid)

                    full_box = tile_box_to_full_box(
                        tile_box_xyxy=box_xyxy,
                        tile_left=l,
                        tile_top=t,
                        full_w=full_w,
                        full_h=full_h,
                        score=score,
                        class_id=class_id,
                        obb8=obb8,
                        xywha=xywha,
                    )
                    # Drop empty boxes
                    if (full_box.x2 - full_box.x1) <= 1.0 or (full_box.y2 - full_box.y1) <= 1.0:
                        continue
                    all_boxes.append(full_box)

        all_boxes = nms(all_boxes, float(args.nms_iou))

        if args.yolo_format == "obb8":
            label_lines: List[str] = []
            for b in all_boxes:
                line = box_to_yolo_obb8_line(b, full_w, full_h)
                if line is not None:
                    label_lines.append(line)
        elif args.yolo_format == "xywha":
            label_lines = []
            for b in all_boxes:
                line = box_to_yolo_xywha_line(b, full_w, full_h, args.angle_unit)
                if line is not None:
                    label_lines.append(line)
        else:
            label_lines = [
                box_to_yolo_line(b, int(b.class_id), full_w, full_h, args.include_score)
                for b in all_boxes
            ]

        out_txt = out_dir / f"{item_key}.txt"
        out_txt.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        total_boxes += len(all_boxes)
        if is_video_file(input_path) and args.save_frames:
            # Save sampled frame image for pairing with YOLO labels
            out_img = out_dir / f"{item_key}.jpg"
            try:
                img.save(out_img, format="JPEG", quality=95)
            except Exception:
                pass

        print(f"[{idx}/{len(items)}] {item_key}: {len(all_boxes)} boxes -> {out_txt}")

    print(f"Done. Wrote labels to: {out_dir} (total boxes: {total_boxes})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
