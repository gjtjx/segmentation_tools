import argparse
from pathlib import Path
import sys
import time
from typing import Any, Optional

import cv2 as _cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as _vision

# OpenCV / MediaPipe 的类型 stub 在部分环境里不完整；这里将其视为 Any，避免编辑器误报。
cv2: Any = _cv2
vision: Any = _vision


class _SimpleLandmark:
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        visibility: float = 1.0,
        presence: float = 1.0,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)
        self.presence = float(presence)


class _SimplePoseResult:
    __slots__ = ("pose_landmarks",)

    def __init__(self, pose_landmarks: list[list[_SimpleLandmark]]):
        self.pose_landmarks = pose_landmarks


MP33_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def _clamp_int_floor(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(np.floor(v)))))


def _clamp_int_ceil(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(np.ceil(v)))))


def _find_person_class_id(names: Any) -> Optional[int]:
    """从 Ultralytics 的 names 中找 person 类别 id。"""
    try:
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == "person":
                    return int(k)
        if isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                if str(v).lower() == "person":
                    return int(i)
    except (TypeError, ValueError):
        return None
    return None


def build_coco33_dict(
    *,
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    pose_result: Optional[Any],
    bbox_xyxy: Optional[tuple[int, int, int, int]] = None,
    score: Optional[float] = None,
    min_visibility: float = 0.0,
) -> dict[str, Any]:
    """生成 COCO 风格标注（但 keypoints 是 33 个，按 MediaPipe BlazePose 顺序）。"""
    categories = [
        {
            "id": 1,
            "name": "person",
            "keypoints": list(MP33_KEYPOINT_NAMES),
            "skeleton": [[a + 1, b + 1] for (a, b) in POSE_CONNECTIONS],
        }
    ]

    annotations: list[dict[str, Any]] = []
    pose_landmarks_list = getattr(pose_result, "pose_landmarks", None) or []
    for ann_id, pose_landmarks in enumerate(pose_landmarks_list, start=1):
        keypoints: list[float] = []
        num_kpts = 0
        xs: list[float] = []
        ys: list[float] = []

        # 确保输出 33 个点（不足补 0，超出截断）
        pose_landmarks = list(pose_landmarks)[: len(MP33_KEYPOINT_NAMES)]
        if len(pose_landmarks) < len(MP33_KEYPOINT_NAMES):
            missing = len(MP33_KEYPOINT_NAMES) - len(pose_landmarks)
            pose_landmarks.extend([_SimpleLandmark(0.0, 0.0, visibility=0.0, presence=0.0)] * missing)

        for lm in pose_landmarks:
            x = float(lm.x) * float(width)
            y = float(lm.y) * float(height)
            v_raw = float(getattr(lm, "visibility", 1.0))
            # COCO: v=0 未标注；v=1 标注但不可见；v=2 可见
            v = 2 if v_raw >= float(min_visibility) else 1
            if v > 0:
                num_kpts += 1
                xs.append(x)
                ys.append(y)
            keypoints.extend([x, y, float(v)])

        if bbox_xyxy is not None:
            x0, y0, x1, y1 = bbox_xyxy
            bbox = [float(x0), float(y0), float(max(0, x1 - x0)), float(max(0, y1 - y0))]
            area = bbox[2] * bbox[3]
        elif xs and ys:
            x0 = float(min(xs))
            y0 = float(min(ys))
            x1 = float(max(xs))
            y1 = float(max(ys))
            bbox = [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]
            area = bbox[2] * bbox[3]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]
            area = 0.0

        if score is None:
            score_val = float(_pose_score(pose_landmarks))
        else:
            score_val = float(score)

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": int(num_kpts),
                "bbox": bbox,
                "area": float(area),
                "iscrowd": 0,
                "score": score_val,
            }
        )

    return {
        "images": [{"id": image_id, "file_name": file_name, "width": int(width), "height": int(height)}],
        "annotations": annotations,
        "categories": categories,
    }


def save_json(path: Path, obj: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yolo_model(yolo_model_path: str) -> Any:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "未安装 ultralytics，无法使用 YOLO 检测。\n"
            "请先执行：pip install ultralytics\n"
            f"原始错误：{e}"
        ) from e

    return YOLO(yolo_model_path)


def yolo_predict_person_boxes(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    conf: float,
    iou: float,
    max_det: int,
    imgsz: int,
) -> list[tuple[int, int, int, int, float]]:
    """返回 person 框列表 (x0,y0,x1,y1,conf)。"""
    names = getattr(model, "names", None)
    person_id = _find_person_class_id(names)

    predict_kwargs: dict[str, Any] = {
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "imgsz": int(imgsz),
        "verbose": False,
    }
    # 如果能确定 person 类别 id，则只跑 person 过滤（加速&减少误检）
    if person_id is not None:
        predict_kwargs["classes"] = [int(person_id)]

    results = model.predict(frame_bgr, **predict_kwargs)
    if not results:
        return []
    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        return []

    xyxy = getattr(boxes, "xyxy", None)
    confs = getattr(boxes, "conf", None)
    clss = getattr(boxes, "cls", None)
    if xyxy is None:
        return []

    try:
        xyxy_np = xyxy.detach().cpu().numpy()
        conf_np = confs.detach().cpu().numpy() if confs is not None else None
        cls_np = clss.detach().cpu().numpy() if clss is not None else None
    except AttributeError:
        # 兼容极端情况（numpy already）
        xyxy_np = np.asarray(xyxy)
        conf_np = np.asarray(confs) if confs is not None else None
        cls_np = np.asarray(clss) if clss is not None else None

    height, width = frame_bgr.shape[:2]
    out: list[tuple[int, int, int, int, float]] = []

    for i in range(int(xyxy_np.shape[0])):
        if cls_np is not None and person_id is not None:
            if int(cls_np[i]) != int(person_id):
                continue
        x0, y0, x1, y1 = xyxy_np[i].tolist()
        # 用 floor/ceil 避免 round 造成框收缩，导致裁剪偏小
        x0i = _clamp_int_floor(x0, 0, width - 1)
        y0i = _clamp_int_floor(y0, 0, height - 1)
        x1i = _clamp_int_ceil(x1, 0, width)
        y1i = _clamp_int_ceil(y1, 0, height)
        if x1i - x0i < 10 or y1i - y0i < 10:
            continue
        c = float(conf_np[i]) if conf_np is not None else 1.0
        out.append((x0i, y0i, x1i, y1i, c))

    # 按置信度降序
    out.sort(key=lambda b: b[4], reverse=True)

    # 额外做一遍严格 NMS（防止极端情况下出现高度重叠框）
    def _iou_xyxy(a: tuple[int, int, int, int, float], b: tuple[int, int, int, int, float]) -> float:
        ax0, ay0, ax1, ay1, _ = a
        bx0, by0, bx1, by1, _ = b
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        inter = iw * ih
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    thr = float(iou)
    if 0.0 <= thr < 1.0 and len(out) > 1:
        kept: list[tuple[int, int, int, int, float]] = []
        for box in out:
            if all(_iou_xyxy(box, k) <= thr for k in kept):
                kept.append(box)
        out = kept
    return out


def draw_boxes_rgb(
    rgb_image: np.ndarray,
    boxes: list[tuple[int, int, int, int, float]],
    *,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    annotated = rgb_image.copy()
    h, w = annotated.shape[:2]
    thickness = max(1, int(round(min(w, h) * 0.003)))

    for x0, y0, x1, y1, c in boxes:
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, thickness=thickness, lineType=cv2.LINE_AA)
        label = f"person {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ty0 = max(0, y0 - th - 6)
        cv2.rectangle(annotated, (x0, ty0), (x0 + tw + 6, ty0 + th + 6), color, thickness=-1)
        cv2.putText(
            annotated,
            label,
            (x0 + 3, ty0 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return annotated


def _scale_pose_result(pose_result: Any, scale_x: float, scale_y: float) -> Any:
    """将姿态结果按比例缩放坐标(用于resize后的crop)"""
    if pose_result is None:
        return None
    
    pose_landmarks_list = getattr(pose_result, "pose_landmarks", None) or []
    if not pose_landmarks_list:
        return pose_result
    
    scaled_all: list[list[_SimpleLandmark]] = []
    for pose_landmarks in pose_landmarks_list:
        scaled: list[_SimpleLandmark] = []
        for lm in pose_landmarks:
            scaled.append(
                _SimpleLandmark(
                    lm.x * scale_x,
                    lm.y * scale_y,
                    z=lm.z,
                    visibility=float(getattr(lm, "visibility", 1.0)),
                    presence=float(getattr(lm, "presence", 1.0)),
                )
            )
        scaled_all.append(scaled)
    
    return _SimplePoseResult(pose_landmarks=scaled_all)


def map_pose_result_from_crop_to_full(
    pose_result: Any,
    *,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    detect_w: int,
    detect_h: int,
    full_w: int,
    full_h: int,
) -> Optional[_SimplePoseResult]:
    """Map pose landmarks from detection image (potentially resized crop) to full image.
    
    MediaPipe returns normalized coords (0-1) relative to the detection image.
    If the crop was resized before detection, we must account for that.
    
    Args:
        crop_x0, crop_y0: Top-left corner of original crop in full image
        crop_w, crop_h: Original crop size in full image (before resize)
        detect_w, detect_h: Actual size of image passed to detector (after resize)
        full_w, full_h: Full image dimensions
    """
    pose_landmarks_list = getattr(pose_result, "pose_landmarks", None) or []
    if not pose_landmarks_list:
        return None

    mapped_all: list[list[_SimpleLandmark]] = []
    sx = crop_w / float(full_w)
    
    # Scale factor from detection image to original crop
    if detect_w <= 0 or detect_h <= 0:
        return None
        
    scale_x = crop_w / float(detect_w)
    scale_y = crop_h / float(detect_h)
    
    for pose_landmarks in pose_landmarks_list:
        mapped: list[_SimpleLandmark] = []
        for lm in pose_landmarks:
            # lm.x/lm.y are normalized (0-1) in detection image space
            # Convert to pixels in detection image
            detect_px = float(lm.x) * detect_w
            detect_py = float(lm.y) * detect_h
            # Scale to original crop pixel coordinates
            crop_px = detect_px * scale_x
            crop_py = detect_py * scale_y
            # Convert to full image normalized coordinates
            fx = (crop_x0 + crop_px) / float(full_w)
            fy = (crop_y0 + crop_py) / float(full_h)
            fz = float(getattr(lm, "z", 0.0)) * sx
            mapped.append(
                _SimpleLandmark(
                    fx,
                    fy,
                    z=fz,
                    visibility=float(getattr(lm, "visibility", 1.0)),
                    presence=float(getattr(lm, "presence", 1.0)),
                )
            )
        mapped_all.append(mapped)

    return _SimplePoseResult(pose_landmarks=mapped_all)


def _bbox_iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _pose_bbox_xyxy(pose: list[_SimpleLandmark], full_w: int, full_h: int) -> Optional[tuple[int, int, int, int]]:
    if not pose:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for lm in pose:
        xs.append(float(lm.x) * float(full_w))
        ys.append(float(lm.y) * float(full_h))
    if not xs or not ys:
        return None
    x0 = int(min(xs))
    y0 = int(min(ys))
    x1 = int(max(xs))
    y1 = int(max(ys))
    return (x0, y0, x1, y1)


def detect_pose_with_yolo_boxes(
    landmarker_image: Any,
    landmarker_video: Optional[Any],
    frame_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int, float]],
    *,
    timestamp_ms: int,
    tile: str,
    max_poses: int,
    pad_ratio: float,
    running_mode: str,
    max_crop_size: int = 1280,
    min_crop_size: int = 0,
    yolo_pose_iou: float = 0.0,
    yolo_fallback_fullframe: bool = False,
    yolo_fallback_iou: float = 0.05,
    debug: bool = False,
    debug_frames: int = 10,
    save_debug_crops: bool = False,
    debug_crop_dir: str = "",
    frame_index: int = 0,
) -> Optional[Any]:
    """对每个 person box 做 pose，合并并取 top-N。
    
    NOTE: Crop检测强制使用IMAGE模式，因为VIDEO模式需要连续的时序帧，
    而crop区域在不同box/帧之间变化很大，会导致tracking失效。
    
    Args:
        landmarker_image: IMAGE模式的landmarker，用于crop检测
        landmarker_video: VIDEO模式的landmarker，用于fullframe fallback（可选）
    """
    h, w = frame_rgb.shape[:2]
    # 每个检测框保留一条最优姿态，避免全局 top-N 导致漏检
    collected_by_box: dict[int, tuple[float, list[_SimpleLandmark]]] = {}
    full_frame_result: Optional[Any] = None

    for box_idx, (x0, y0, x1, y1, _c) in enumerate(boxes):
        # 给 box 加一点 padding，避免框太紧导致 pose 检测失败
        pad_ratio_f = max(0.0, float(pad_ratio))
        bw = max(x1 - x0, 1)
        bh = max(y1 - y0, 1)
        pad_x = int(round(bw * pad_ratio_f))
        pad_y = int(round(bh * pad_ratio_f))
        x0p = max(0, x0 - pad_x)
        y0p = max(0, y0 - pad_y)
        x1p = min(w, x1 + pad_x)
        y1p = min(h, y1 + pad_y)

        crop = frame_rgb[y0p:y1p, x0p:x1p]
        if crop.size == 0:
            if debug:
                print(f"[DBG] frame={frame_index} box={box_idx} SKIP: crop is empty")
            continue
        crop_h = max(y1p - y0p, 1)
        crop_w = max(x1p - x0p, 1)

        if debug:
            print(f"[DBG] frame={frame_index} box={box_idx} orig_box=[{x0},{y0},{x1},{y1}] "
                  f"padded=[{x0p},{y0p},{x1p},{y1p}] crop_shape={crop.shape}")

        # Resize裁剪图：小框上采样/大框下采样
        crop_for_detect = crop
        scale = 1.0
        if min_crop_size > 0 and (crop_h < min_crop_size or crop_w < min_crop_size):
            scale = max(min_crop_size / crop_w, min_crop_size / crop_h)
        elif max_crop_size > 0 and (crop_h > max_crop_size or crop_w > max_crop_size):
            scale = min(max_crop_size / crop_w, max_crop_size / crop_h)

        if scale != 1.0:
            new_w = int(round(crop_w * scale))
            new_h = int(round(crop_h * scale))
            crop_for_detect = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if debug:
                print(
                    f"[DBG] frame={frame_index} box={box_idx} resized crop: {crop.shape} -> {crop_for_detect.shape}, scale={scale:.2f}"
                )

        # 确保数据连续，避免MediaPipe在处理非连续slice时出现坐标错位
        if not crop_for_detect.flags['C_CONTIGUOUS']:
            crop_for_detect = np.ascontiguousarray(crop_for_detect)

        # 保存调试裁剪图像
        if save_debug_crops and debug_crop_dir:
            crop_filename = f"crop_frame{frame_index:05d}_box{box_idx}.jpg"
            crop_path = Path(debug_crop_dir) / crop_filename
            crop_bgr = cv2.cvtColor(crop_for_detect, cv2.COLOR_RGB2BGR)
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            imwrite_any_path(crop_path, crop_bgr)
            if debug:
                print(f"[DBG] Saved crop to: {crop_filename}")

        # 保证同一帧内多次 detect 的时间戳仍然单调递增
        ts = int(timestamp_ms) * 100 + int(box_idx)

        # 强制使用IMAGE模式检测crop：VIDEO模式需要时序连续性，crop区域变化大会失效
        crop_result = detect_pose_for_frame_rgb(
            landmarker_image,
            crop_for_detect,
            timestamp_base_ms=ts,
            tile=tile,
            num_poses=1,
            running_mode="image",  # 强制IMAGE模式
        )
        num_detected = len(getattr(crop_result, "pose_landmarks", []) or []) if crop_result is not None else 0
        if debug:
            print(f"[DBG] frame={frame_index} box={box_idx} detected {num_detected} pose(s) in crop")

        if (crop_result is None or num_detected == 0) and yolo_fallback_fullframe:
            if full_frame_result is None:
                ts_full = int(timestamp_ms) * 100 + 90
                # 全图回退使用VIDEO模式landmarker（如果提供）
                if landmarker_video is not None:
                    full_frame_result = detect_pose_for_frame_rgb(
                        landmarker_video,
                        frame_rgb,
                        timestamp_base_ms=ts_full,
                        tile=tile,
                        num_poses=max(1, int(max_poses)),
                        running_mode="video",
                    )
                else:
                    full_frame_result = detect_pose_for_frame_rgb(
                        landmarker_image,
                        frame_rgb,
                        timestamp_base_ms=ts_full,
                        tile=tile,
                        num_poses=max(1, int(max_poses)),
                        running_mode="image",
                    )
                if debug:
                    ff_n = len(getattr(full_frame_result, "pose_landmarks", []) or []) if full_frame_result is not None else 0
                    print(f"[DBG] frame={frame_index} fullframe detected {ff_n} pose(s)")

            if full_frame_result is not None:
                best_pose: Optional[list[_SimpleLandmark]] = None
                best_iou = 0.0
                for pose in getattr(full_frame_result, "pose_landmarks", []) or []:
                    pose_bbox = _pose_bbox_xyxy(pose, w, h)
                    if pose_bbox is None:
                        continue
                    iou = _bbox_iou_xyxy(pose_bbox, (x0, y0, x1, y1))
                    if iou > best_iou:
                        best_iou = iou
                        best_pose = pose

                min_iou = max(float(yolo_pose_iou), float(yolo_fallback_iou))
                if best_pose is not None and best_iou >= min_iou:
                    score = _pose_score(best_pose)
                    prev = collected_by_box.get(box_idx)
                    if prev is None or score > prev[0]:
                        collected_by_box[box_idx] = (score, best_pose)
                continue

        if crop_result is None:
            if debug:
                print(f"[DBG] frame={frame_index} box={box_idx} detect returned None")
            continue

        # Map coordinates from detection image back to full image
        # crop_w, crop_h: 原始裁剪在全图中的尺寸（未resize前）
        # crop_for_detect: 实际传给MediaPipe的图像（可能已resize）
        detect_h_actual, detect_w_actual = crop_for_detect.shape[:2]
        if debug and frame_index < int(debug_frames):
            print(f"[DBG] MAPPING: crop_w={crop_w}, crop_h={crop_h}, detect_w={detect_w_actual}, detect_h={detect_h_actual}, crop_x0={x0p}, crop_y0={y0p}")
        mapped = map_pose_result_from_crop_to_full(
            crop_result,
            crop_x0=x0p,
            crop_y0=y0p,
            crop_w=crop_w,  # 原始crop在全图中的宽度
            crop_h=crop_h,  # 原始crop在全图中的高度  
            detect_w=detect_w_actual,  # detection图像实际宽度(可能被resize)
            detect_h=detect_h_actual,  # detection图像实际高度(可能被resize)
            full_w=w,
            full_h=h,
        )
        if mapped is None:
            continue

        for pose in mapped.pose_landmarks:
            if yolo_pose_iou > 0.0:
                pose_bbox = _pose_bbox_xyxy(pose, w, h)
                if pose_bbox is None:
                    continue
                if _bbox_iou_xyxy(pose_bbox, (x0, y0, x1, y1)) < float(yolo_pose_iou):
                    continue
            score = _pose_score(pose)
            prev = collected_by_box.get(box_idx)
            if prev is None or score > prev[0]:
                collected_by_box[box_idx] = (score, pose)

    if not collected_by_box:
        return None

    # 按检测框顺序返回，确保一框一姿态
    poses = [v[1] for _, v in sorted(collected_by_box.items(), key=lambda t: t[0])]
    return _SimplePoseResult(pose_landmarks=poses)


# 33 点 BlazePose 的连接关系。
# 这里包含：官方骨架连接 + 额外的头部/手部/脚部连线（更利于可视化）。
# 索引定义参考：https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (16, 18),
    (17, 19), (18, 20),
    (19, 21), (20, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),

    # 头部额外连线（使面部轮廓更清晰）
    (1, 4),   # eye_inner L-R
    (2, 5),   # eye center L-R
    (3, 6),   # eye_outer L-R
    (7, 8),   # ear L-R
    (0, 9), (0, 10), (9, 10),  # nose-mouth
    (2, 0), (5, 0),  # eyes to nose

    # 手部额外连线（腕到食指/拇指，形成更完整的手部形状）
    (15, 19), (15, 21), (17, 21),
    (16, 20), (16, 22), (18, 22),

    # 脚部额外连线（补全踝-脚跟-脚尖三角；部分已在官方连接里，这里补齐闭合边）
    (29, 27), (31, 27), (29, 31),
    (30, 28), (32, 28), (30, 32),
)


# 左右关键点索引（BlazePose 33 landmarks）
LEFT_LANDMARKS: frozenset[int] = frozenset({
    1, 2, 3, 7, 9,
    11, 13, 15, 17, 19, 21,
    23, 25, 27, 29, 31,
})
RIGHT_LANDMARKS: frozenset[int] = frozenset({
    4, 5, 6, 8, 10,
    12, 14, 16, 18, 20, 22,
    24, 26, 28, 30, 32,
})


def make_output_path(input_path: Path, out_dir: Path, suffix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 按需求：原始文件名 + vis.mp4（默认 suffix='vis'，不带下划线）
    return out_dir / f"{input_path.stem}{suffix}.mp4"


def make_output_frames_dir(input_dir: Path, out_dir: Path, suffix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / f"{input_dir.name}{suffix}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir


def iter_image_files(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # 默认按文件名排序，适用于 0001.jpg / frame_0001.png 这类序列
    return sorted(files, key=lambda p: p.name)


def imread_any_path(path: Path) -> Optional[np.ndarray]:
    """兼容 Windows 中文路径的读取。

    OpenCV 的 cv2.imread/cv2.imwrite 在部分 Windows 环境下对非 ASCII 路径支持不稳定，
    这里使用 fromfile + imdecode 的方式读取。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except (OSError, ValueError):
        return None


def imwrite_any_path(path: Path, image_bgr: np.ndarray) -> bool:
    """兼容 Windows 中文路径的写入（PNG/JPG）。"""
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"不支持的输出后缀：{suffix}（仅 png/jpg）")

    try:
        ok, buf = cv2.imencode(suffix, image_bgr)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except (OSError, ValueError):
        return False


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> Any:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Windows 上常见可用的 MP4 fourcc：mp4v；avc1 取决于系统/编译
    for fourcc_str in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer

    raise RuntimeError(
        f"无法创建 VideoWriter 输出：{output_path}\n"
        "请检查 OpenCV 是否具备 MP4 编码能力（或更换 fourcc / 安装带 FFmpeg 的 OpenCV）。"
    )


def _tile_boxes_2x2(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """返回 2x2 切块的像素坐标框 (x0,y0,x1,y1)，覆盖整幅图。"""
    x_mid = width // 2
    y_mid = height // 2
    return [
        (0, 0, x_mid, y_mid),
        (x_mid, 0, width, y_mid),
        (0, y_mid, x_mid, height),
        (x_mid, y_mid, width, height),
    ]


def _pose_score(pose_landmarks: list[Any]) -> float:
    # 用可见度做一个简单可信度分数；越大越可信。
    if not pose_landmarks:
        return 0.0
    vis = [float(getattr(lm, "visibility", 1.0)) for lm in pose_landmarks]
    return float(np.mean(vis))


def _has_any_pose(result: Optional[Any]) -> bool:
    if result is None:
        return False
    pose_landmarks = getattr(result, "pose_landmarks", None)
    return bool(pose_landmarks)


def detect_pose_for_frame_rgb(
    landmarker: Any,
    frame_rgb: np.ndarray,
    *,
        timestamp_base_ms: int,
    tile: str,
    num_poses: int,
        running_mode: str,
) -> Optional[Any]:
    """对单帧 RGB 图做姿态检测；支持 2x2 tile。

    - tile='none': 直接检测
    - tile='2x2': 切 4 块分别检测，映射回整图坐标后选 top-N
    """
    if running_mode not in {"video", "image"}:
        raise ValueError("running_mode 必须是 'video' 或 'image'")

    def _detect(mp_image: Any, ts_ms: int) -> Any:
        if running_mode == "video":
            return landmarker.detect_for_video(mp_image, int(ts_ms))
        return landmarker.detect(mp_image)

    if tile == "none":
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return _detect(mp_image, int(timestamp_base_ms))

    if tile != "2x2":
        raise ValueError("--tile 仅支持 none 或 2x2")

    height, width = frame_rgb.shape[:2]
    boxes = _tile_boxes_2x2(width, height)

    all_poses: list[tuple[float, list[_SimpleLandmark]]] = []

    # VIDEO mode 下需时间戳单调递增：每帧放大 10 倍再加 tile_idx；IMAGE mode 下不会用到
    base = int(timestamp_base_ms) * 10

    for tile_idx, (x0, y0, x1, y1) in enumerate(boxes):
        crop = frame_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        result = landmarker.detect_for_video(mp_image, base + tile_idx)
        pose_landmarks_list = getattr(result, "pose_landmarks", None) or []
        if not pose_landmarks_list:
            continue

        tile_w = max(x1 - x0, 1)
        tile_h = max(y1 - y0, 1)
        sx = tile_w / float(width)

        for pose_landmarks in pose_landmarks_list:
            mapped: list[_SimpleLandmark] = []
            for lm in pose_landmarks:
                # lm.x/lm.y 是 tile 内归一化坐标
                fx = (x0 + float(lm.x) * tile_w) / float(width)
                fy = (y0 + float(lm.y) * tile_h) / float(height)
                fz = float(getattr(lm, "z", 0.0)) * sx
                mapped.append(
                    _SimpleLandmark(
                        fx,
                        fy,
                        z=fz,
                        visibility=float(getattr(lm, "visibility", 1.0)),
                        presence=float(getattr(lm, "presence", 1.0)),
                    )
                )

            all_poses.append((_pose_score(mapped), mapped))

    if not all_poses:
        return None

    all_poses.sort(key=lambda t: t[0], reverse=True)
    top = [pose for _, pose in all_poses[: max(1, int(num_poses))]]
    return _SimplePoseResult(pose_landmarks=top)


def _to_pixel_coords(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return int(round(x * (width - 1))), int(round(y * (height - 1)))


def draw_pose_landmarks_rgb(
    rgb_image: np.ndarray,
    pose_result: Optional[Any],
    *,
    min_visibility: float = 0.0,
) -> np.ndarray:
    """绘制姿态点与骨架。

    说明：官方 Colab 示例使用 `mediapipe.solutions` + `landmark_pb2` 来绘制。
    但在部分 Windows/Python 3.13 环境里，pip 安装到的 mediapipe 包可能缺少
    `mediapipe.solutions`/`mediapipe.framework` 子模块，因此这里改为纯 OpenCV 绘制，
    以确保脚本可运行。
    """
    annotated = rgb_image.copy()
    if pose_result is None:
        return annotated

    height, width = annotated.shape[:2]

    # 左右不同色（在 RGB 图上直接画，传入的 tuple 即按通道写入）
    # 左侧：青色；右侧：橙色；中轴/跨侧：白色
    left_color = (0, 255, 255)
    right_color = (255, 165, 0)
    center_color = (255, 255, 255)

    def pick_color_for_landmark(index: int) -> tuple[int, int, int]:
        if index in LEFT_LANDMARKS:
            return left_color
        if index in RIGHT_LANDMARKS:
            return right_color
        return center_color

    def pick_color_for_connection(a: int, b: int) -> tuple[int, int, int]:
        a_left = a in LEFT_LANDMARKS
        a_right = a in RIGHT_LANDMARKS
        b_left = b in LEFT_LANDMARKS
        b_right = b in RIGHT_LANDMARKS

        if a_left and b_left:
            return left_color
        if a_right and b_right:
            return right_color
        # 跨左右或含中轴点
        return center_color

    line_thickness = max(1, int(round(min(width, height) * 0.003)))
    point_radius = max(2, int(round(min(width, height) * 0.006)))

    for pose_landmarks in pose_result.pose_landmarks:
        pts: list[Optional[tuple[int, int]]] = [None] * len(pose_landmarks)

        # 先算像素坐标（过滤低可见度）
        first_point = True
        for i, lm in enumerate(pose_landmarks):
            vis = float(getattr(lm, "visibility", 1.0))
            if vis < min_visibility:
                continue
            x = float(lm.x)
            y = float(lm.y)
            # 允许轻微越界（画的时候再 clamp/跳过）
            px, py = _to_pixel_coords(x, y, width, height)
            if first_point and i == 0:
                 # 仅在第一点（鼻子）做一次采样打印
                 # print(f"[VIS] Nose draw at: x={x:.4f}->{px}, y={y:.4f}->{py} (img={width}x{height})")
                 first_point = False
            
            if 0 <= px < width and 0 <= py < height:
                pts[i] = (px, py)

        # 画连线（左右不同颜色）
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts) and pts[a] is not None and pts[b] is not None:
                cv2.line(
                    annotated,
                    pts[a],
                    pts[b],
                    pick_color_for_connection(a, b),
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )

        # 画点（最后画，盖在线上；左右不同颜色）
        for idx, p in enumerate(pts):
            if p is not None:
                cv2.circle(
                    annotated,
                    p,
                    radius=point_radius,
                    color=pick_color_for_landmark(idx),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    return annotated


def process_video(
    input_video: Path,
    output_video: Path,
    model_path: Path,
    stride: int,
    num_poses: int,
    min_pose_detection_confidence: float,
    min_pose_presence_confidence: float,
    min_tracking_confidence: float,
    min_visibility: float,
    tile: str,
    yolo_model: Optional[Any],
    yolo_conf: float,
    yolo_iou: float,
    yolo_max_det: int,
    yolo_imgsz: int,
    yolo_stride: int,
    yolo_pad: float,
    max_crop_size: int,
    yolo_pose_iou: float,
    yolo_fallback_fullframe: bool,
    yolo_fallback_iou: float,
    draw_boxes: bool,
    save_coco_json: bool,
    json_out_dir: Optional[Path],
    debug: bool,
    debug_frames: int,
    save_debug_crops: bool,
    debug_crop_dir: str,
    save_frames: bool,
    frames_out_dir: Optional[Path],
    frame_format: str,
) -> None:
    if stride < 1:
        raise ValueError("--stride 必须 >= 1")
    if not input_video.exists():
        raise FileNotFoundError(f"找不到输入视频：{input_video}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{model_path}\n"
            "请下载 MediaPipe Pose Landmarker 的 .task 模型并通过 --model 指定。"
        )

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{input_video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        raise RuntimeError("读取到的视频宽高无效。")

    writer = create_video_writer(output_video, fps, width, height)

    frames_dir: Optional[Path] = None
    frame_ext = str(frame_format).lower().lstrip(".")
    if frame_ext == "jpeg":
        frame_ext = "jpg"
    if save_frames:
        if frame_ext not in {"png", "jpg"}:
            raise ValueError("--image_format 仅支持 png/jpg")
        frames_dir = frames_out_dir
        if frames_dir is None:
            frames_dir = output_video.parent / f"{output_video.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options_video = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    # YOLO crop需要IMAGE模式的landmarker（VIDEO模式在crop区域变化时会失效）
    options_image = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,  # crop中通常只有一个人
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    last_result: Optional[Any] = None
    last_boxes: list[tuple[int, int, int, int, float]] = []
    frame_index = 0
    t0 = time.time()

    with vision.PoseLandmarker.create_from_options(options_video) as landmarker_video, \
         vision.PoseLandmarker.create_from_options(options_image) as landmarker_image:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((frame_index / fps) * 1000.0)

            if yolo_model is not None:
                if yolo_stride < 1:
                    raise ValueError("--yolo_stride 必须 >= 1")

                if (frame_index % yolo_stride == 0) or (not last_boxes):
                    last_boxes = yolo_predict_person_boxes(
                        yolo_model,
                        frame_bgr,
                        conf=yolo_conf,
                        iou=yolo_iou,
                        max_det=yolo_max_det,
                        imgsz=yolo_imgsz,
                    )

                if debug and frame_index < int(debug_frames):
                    print(f"[DBG] frame={frame_index} yolo_boxes={len(last_boxes)}")

                # YOLO 模式下，stride 仍然生效：每 stride 帧更新 pose；其余帧复用 last_result
                if frame_index % stride == 0:
                    result = detect_pose_with_yolo_boxes(
                        landmarker_image,  # IMAGE模式用于crop
                        landmarker_video,  # VIDEO模式用于fallback
                        frame_rgb,
                        last_boxes,
                        timestamp_ms=timestamp_ms,
                        tile=tile,
                        max_poses=num_poses,
                        pad_ratio=yolo_pad,
                        running_mode="image",
                        max_crop_size=max_crop_size,
                        debug=debug,
                        debug_frames=debug_frames,
                        frame_index=frame_index,
                        save_debug_crops=save_debug_crops,
                        debug_crop_dir=debug_crop_dir,
                        yolo_pose_iou=yolo_pose_iou,
                        yolo_fallback_fullframe=yolo_fallback_fullframe,
                        yolo_fallback_iou=yolo_fallback_iou,
                    )
                    last_result = result if _has_any_pose(result) else None
                    if debug and frame_index < int(debug_frames):
                        poses_n = len(getattr(last_result, "pose_landmarks", []) or []) if last_result is not None else 0
                        print(f"[DBG] frame={frame_index} poses_in_boxes={poses_n}")
            else:
                # 非 YOLO：跳帧推理，使用VIDEO模式
                if frame_index % stride == 0:
                    result = detect_pose_for_frame_rgb(
                        landmarker_video,
                        frame_rgb,
                        timestamp_base_ms=timestamp_ms,
                        tile=tile,
                        num_poses=num_poses,
                        running_mode="video",
                    )
                    last_result = result if _has_any_pose(result) else None

            if debug and frame_index < int(debug_frames):
                if last_result is None:
                    print(f"[DBG] frame={frame_index} pose=None")
                else:
                    poses = getattr(last_result, "pose_landmarks", None) or []
                    if not poses:
                        print(f"[DBG] frame={frame_index} pose_landmarks=[]")
                    else:
                        # 输出归一化坐标范围：正常应大致落在 [0,1]
                        for p_idx, pose in enumerate(poses):
                            xs = [float(lm.x) for lm in pose if hasattr(lm, "x")]
                            ys = [float(lm.y) for lm in pose if hasattr(lm, "y")]
                            if not xs or not ys:
                                print(f"[DBG] frame={frame_index} pose[{p_idx}] empty")
                                continue
                            min_x = min(xs)
                            max_x = max(xs)
                            min_y = min(ys)
                            max_y = max(ys)
                            oor = sum(1 for (x, y) in zip(xs, ys) if (x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0))
                            print(
                                f"[DBG] frame={frame_index} pose[{p_idx}] x=[{min_x:.3f},{max_x:.3f}] y=[{min_y:.3f},{max_y:.3f}] out_of_range={oor}/{min(len(xs), len(ys))}"
                            )

            annotated_rgb = draw_pose_landmarks_rgb(frame_rgb, last_result, min_visibility=min_visibility)
            if yolo_model is not None and draw_boxes and last_boxes:
                annotated_rgb = draw_boxes_rgb(annotated_rgb, last_boxes)

            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            if save_coco_json:
                out_dir_for_json = json_out_dir
                if out_dir_for_json is None:
                    out_dir_for_json = output_video.parent / f"{output_video.stem}_json"
                coco = build_coco33_dict(
                    image_id=int(frame_index),
                    file_name=f"{frame_index:06d}.{frame_ext}" if frames_dir is not None else f"{frame_index:06d}.png",
                    width=int(width),
                    height=int(height),
                    pose_result=last_result,
                    bbox_xyxy=None,
                    min_visibility=min_visibility,
                )
                save_json(out_dir_for_json / f"{frame_index:06d}.json", coco)

            if frames_dir is not None:
                frame_path = frames_dir / f"{frame_index:06d}.{frame_ext}"
                ok_write = imwrite_any_path(frame_path, annotated_bgr)
                if not ok_write:
                    raise RuntimeError(f"写入失败：{frame_path}")
            writer.write(annotated_bgr)

            frame_index += 1
            if frame_index % 100 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                if total_frames > 0:
                    print(f"[{frame_index}/{total_frames}] {frame_index/elapsed:.1f} FPS")
                else:
                    print(f"[{frame_index}] {frame_index/elapsed:.1f} FPS")

    cap.release()
    writer.release()


def process_image_folder(
    input_dir: Path,
    output_frames_dir: Path,
    model_path: Path,
    stride: int,
    fps: float,
    image_format: str,
    num_poses: int,
    min_pose_detection_confidence: float,
    min_pose_presence_confidence: float,
    min_tracking_confidence: float,
    min_visibility: float,
    tile: str,
    yolo_model: Optional[Any],
    yolo_conf: float,
    yolo_iou: float,
    yolo_max_det: int,
    yolo_imgsz: int,
    yolo_stride: int,
    yolo_pad: float,
    max_crop_size: int,
    yolo_pose_iou: float,
    yolo_fallback_fullframe: bool,
    yolo_fallback_iou: float,
    draw_boxes: bool,
    save_coco_json: bool,
    json_out_dir: Optional[Path],
    debug: bool,
    debug_frames: int,
    save_debug_crops: bool,
    debug_crop_dir: str,
) -> None:
    if stride < 1:
        raise ValueError("--stride 必须 >= 1")
    if fps <= 0:
        raise ValueError("--fps 必须 > 0")
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"找不到输入图片文件夹：{input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{model_path}\n"
            "请下载 MediaPipe Pose Landmarker 的 .task 模型并通过 --model 指定。"
        )

    image_files = iter_image_files(input_dir)
    if not image_files:
        raise RuntimeError(f"输入文件夹中未找到图片：{input_dir}")

    output_frames_dir.mkdir(parents=True, exist_ok=True)
    image_format = image_format.lower().lstrip(".")
    if image_format not in {"png", "jpg", "jpeg"}:
        raise ValueError("--image_format 仅支持 png/jpg")
    if image_format == "jpeg":
        image_format = "jpg"

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    last_result: Optional[Any] = None
    last_boxes: list[tuple[int, int, int, int, float]] = []
    t0 = time.time()
    saved = 0
    failed = 0
    last_shape: Optional[tuple[int, int, int]] = None

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_index, img_path in enumerate(image_files):
            frame_bgr = imread_any_path(img_path)
            if frame_bgr is None:
                failed += 1
                print(f"[WARN] 读取失败，跳过：{img_path}")
                continue

            # 若帧尺寸变化，则强制进行一次检测，避免复用旧结果导致骨架整体错位
            shape = frame_bgr.shape
            force_detect = last_shape is None or shape != last_shape
            last_shape = shape

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((frame_index / fps) * 1000.0)

            if yolo_model is not None:
                if yolo_stride < 1:
                    raise ValueError("--yolo_stride 必须 >= 1")

                if force_detect or (frame_index % yolo_stride == 0) or (not last_boxes):
                    last_boxes = yolo_predict_person_boxes(
                        yolo_model,
                        frame_bgr,
                        conf=yolo_conf,
                        iou=yolo_iou,
                        max_det=yolo_max_det,
                        imgsz=yolo_imgsz,
                    )

                if debug and frame_index < int(debug_frames):
                    print(f"[DBG] frame={frame_index} file={img_path.name} yolo_boxes={len(last_boxes)}")

                if force_detect or (frame_index % stride == 0):
                    result = detect_pose_with_yolo_boxes(
                        landmarker,  # IMAGE模式用于crop
                        None,  # 图片序列不需要VIDEO模式fallback
                        frame_rgb,
                        last_boxes,
                        timestamp_ms=timestamp_ms,
                        tile=tile,
                        max_poses=num_poses,
                        pad_ratio=yolo_pad,
                        running_mode="image",
                        max_crop_size=max_crop_size,
                        yolo_pose_iou=yolo_pose_iou,
                        yolo_fallback_fullframe=yolo_fallback_fullframe,
                        yolo_fallback_iou=yolo_fallback_iou,
                        debug=debug,
                        debug_frames=debug_frames,
                        frame_index=frame_index,
                        save_debug_crops=save_debug_crops,
                        debug_crop_dir=debug_crop_dir,
                    )
                    last_result = result if _has_any_pose(result) else None

                    if debug and frame_index < int(debug_frames):
                        poses_n = len(getattr(last_result, "pose_landmarks", []) or []) if last_result is not None else 0
                        print(f"[DBG] frame={frame_index} poses_in_boxes={poses_n}")
            else:
                if force_detect or (frame_index % stride == 0):
                    result = detect_pose_for_frame_rgb(
                        landmarker,
                        frame_rgb,
                        timestamp_base_ms=timestamp_ms,
                        tile=tile,
                        num_poses=num_poses,
                        running_mode="image",
                    )
                    last_result = result if _has_any_pose(result) else None

            annotated_rgb = draw_pose_landmarks_rgb(frame_rgb, last_result, min_visibility=min_visibility)
            if yolo_model is not None and draw_boxes and last_boxes:
                annotated_rgb = draw_boxes_rgb(annotated_rgb, last_boxes)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # 保持原始文件名（stem）；若重名则追加帧号避免覆盖
            out_name = f"{img_path.stem}.{image_format}"
            out_path = output_frames_dir / out_name
            if out_path.exists():
                out_path = output_frames_dir / f"{img_path.stem}_{frame_index:06d}.{image_format}"
            ok = imwrite_any_path(out_path, annotated_bgr)
            if not ok:
                raise RuntimeError(f"写入失败：{out_path}")

            if save_coco_json:
                out_dir_for_json = json_out_dir
                if out_dir_for_json is None:
                    out_dir_for_json = output_frames_dir
                coco = build_coco33_dict(
                    image_id=int(frame_index),
                    file_name=out_path.name,
                    width=int(frame_bgr.shape[1]),
                    height=int(frame_bgr.shape[0]),
                    pose_result=last_result,
                    bbox_xyxy=None,
                    min_visibility=min_visibility,
                )
                save_json(out_dir_for_json / f"{img_path.stem}.json", coco)

            saved += 1

            if (frame_index + 1) % 100 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                print(f"[{frame_index + 1}/{len(image_files)}] {((frame_index + 1)/elapsed):.1f} FPS")

    if failed > 0:
        print(f"[WARN] 读取失败 {failed} 张；成功输出 {saved} 张。")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "使用 MediaPipe Pose Landmarker 对视频/图片序列进行姿态点可视化。\n"
            "- 输入视频：输出 MP4\n"
            "- 输入图片文件夹：输出逐帧可视化图片"
        )
    )
    p.add_argument("--input", "-i", type=str, required=True, help="输入视频路径 或 图片文件夹路径")
    p.add_argument("--out_dir", "-o", type=str, required=True, help="输出目录（自动创建）")
    p.add_argument("--model", "-m", type=str, required=True, help="Pose Landmarker .task 模型路径")
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="跳帧推理步长：每 stride 帧推理一次（默认 1，不跳帧）",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="vis",
        help=(
            "输出后缀（默认 vis）。\n"
            "- 视频：foo.mp4 -> foovis.mp4；如需 foo_vis.mp4 用 --suffix _vis\n"
            "- 图片文件夹：输出目录名为 <folder_name><suffix>"
        ),
    )

    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="图片序列模式下用于生成时间戳的 FPS（默认 30；视频模式下忽略）",
    )
    p.add_argument(
        "--image_format",
        type=str,
        default="png",
        help="图片序列模式输出格式：png 或 jpg（默认 png）；若开启 --save_frames 也用于视频逐帧输出格式",
    )

    p.add_argument(
        "--save_frames",
        action="store_true",
        help="视频模式下同时保存逐帧可视化图片序列（默认关闭）",
    )
    p.add_argument(
        "--frames_out_dir",
        type=str,
        default="",
        help="视频逐帧图片输出目录（默认：<输出mp4_stem>_frames）",
    )

    p.add_argument(
        "--tile",
        type=str,
        default="none",
        choices=["none", "2x2"],
        help="大图推理切块模式：none（默认）或 2x2（四块分别推理后合并）",
    )

    p.add_argument(
        "--yolo_model",
        type=str,
        default="",
        help="可选：Ultralytics YOLO 权重路径（例如 yolo26m.pt）。提供后将先检测 person 框，再在框内做姿态估计。",
    )
    p.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO 置信度阈值（默认 0.25）")
    p.add_argument("--yolo_iou", type=float, default=0.45, help="YOLO NMS IoU 阈值（默认 0.45）")
    p.add_argument("--yolo_imgsz", type=int, default=640, help="YOLO 推理尺寸 imgsz（默认 640）")
    p.add_argument("--yolo_max_det", type=int, default=5, help="每帧最多检测多少个框（默认 5）")
    p.add_argument("--yolo_stride", type=int, default=1, help="YOLO 检测步长：每 N 帧跑一次检测（默认 1）")
    p.add_argument(
        "--yolo_pad",
        type=float,
        default=0.15,
        help="对 person 框四周增加 padding 的比例（默认 0.15），避免框太紧导致姿态检测失败",
    )
    p.add_argument(
        "--yolo_pose_iou",
        type=float,
        default=0.0,
        help="姿态框与YOLO框的最小IoU阈值（默认0，不过滤）。建议 0.1~0.3",
    )
    p.add_argument(
        "--yolo_fallback_fullframe",
        action="store_true",
        help="当框内姿态检测失败时，回退到整图检测并按IoU匹配到YOLO框",
    )
    p.add_argument(
        "--yolo_fallback_iou",
        type=float,
        default=0.05,
        help="整图回退匹配的最小IoU阈值（默认0.05）。建议 0.05~0.2",
    )
    p.add_argument(
        "--max_crop_size",
        type=int,
        default=1280,
        help="YOLO裁剪图最大尺寸（默认1280）。裁剪图超过此尺寸会被resize，0表示不限制",
    )
    p.add_argument(
        "--draw_boxes",
        action="store_true",
        help="在输出图上绘制 YOLO person 框（仅 --yolo_model 启用时有效）",
    )

    p.add_argument(
        "--save_coco_json",
        action="store_true",
        help="保存每帧/每张图的 COCO 风格关键点 JSON（33 点，按 MediaPipe 顺序）",
    )
    p.add_argument(
        "--json_out_dir",
        type=str,
        default="",
        help="JSON 输出目录（默认：图片文件夹模式与可视化图同目录；视频模式为 <output_mp4_stem>_json）",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="打印调试信息（前若干帧显示 YOLO 框数量与框内 pose 数量）",
    )
    p.add_argument(
        "--debug_frames",
        type=int,
        default=10,
        help="调试输出前多少帧（默认 10）",
    )
    p.add_argument(
        "--save_debug_crops",
        action="store_true",
        help="保存 YOLO 裁剪后的人体框图像，用于调试姿态检测问题",
    )
    p.add_argument(
        "--debug_crop_dir",
        type=str,
        default="",
        help="调试裁剪图像保存目录（默认：输出目录下的 debug_crops 子目录）",
    )

    p.add_argument("--num_poses", type=int, default=1, help="最多检测人数（默认 1）")
    p.add_argument("--min_pose_detection_confidence", type=float, default=0.0)
    p.add_argument("--min_pose_presence_confidence", type=float, default=0.0)
    p.add_argument("--min_tracking_confidence", type=float, default=0.0)
    p.add_argument(
        "--min_visibility",
        type=float,
        default=0.0,
        help="绘制时过滤低可见度关键点（默认 0.0 不过滤；建议 0.5）",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    model_path = Path(args.model)

    print(f"Input : {input_path}")
    print(f"Model : {model_path}")
    print(f"Stride: {args.stride}")
    print(f"Tile  : {args.tile}")

    yolo_model = None
    if str(args.yolo_model).strip():
        yolo_model = load_yolo_model(str(args.yolo_model).strip())
        print(f"YOLO  : {args.yolo_model}")

    json_out_dir = Path(args.json_out_dir) if str(args.json_out_dir).strip() else None

    frames_out_dir = Path(args.frames_out_dir) if str(args.frames_out_dir).strip() else None

    # 设置调试裁剪目录
    debug_crop_dir = ""
    if args.save_debug_crops:
        if str(args.debug_crop_dir).strip():
            debug_crop_dir = str(args.debug_crop_dir).strip()
        else:
            debug_crop_dir = str(out_dir / "debug_crops")

    if input_path.exists() and input_path.is_dir():
        output_frames_dir = make_output_frames_dir(input_path, out_dir, args.suffix)
        print(f"Output: {output_frames_dir}")
        print(f"FPS   : {args.fps} (for folder)")
        print(f"Fmt   : {args.image_format}")

        process_image_folder(
            input_dir=input_path,
            output_frames_dir=output_frames_dir,
            model_path=model_path,
            stride=args.stride,
            fps=args.fps,
            image_format=args.image_format,
            num_poses=args.num_poses,
            min_pose_detection_confidence=args.min_pose_detection_confidence,
            min_pose_presence_confidence=args.min_pose_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            min_visibility=args.min_visibility,
            tile=args.tile,
            yolo_model=yolo_model,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_max_det=args.yolo_max_det,
            yolo_imgsz=args.yolo_imgsz,
            yolo_stride=args.yolo_stride,
            yolo_pad=args.yolo_pad,
            max_crop_size=args.max_crop_size,
            yolo_pose_iou=args.yolo_pose_iou,
            yolo_fallback_fullframe=args.yolo_fallback_fullframe,
            yolo_fallback_iou=args.yolo_fallback_iou,
            draw_boxes=args.draw_boxes,
            save_coco_json=args.save_coco_json,
            json_out_dir=json_out_dir,
            debug=args.debug,
            debug_frames=args.debug_frames,
            save_debug_crops=args.save_debug_crops,
            debug_crop_dir=debug_crop_dir,
        )
    else:
        output_video = make_output_path(input_path, out_dir, args.suffix)
        print(f"Output: {output_video}")

        process_video(
            input_video=input_path,
            output_video=output_video,
            model_path=model_path,
            stride=args.stride,
            num_poses=args.num_poses,
            min_pose_detection_confidence=args.min_pose_detection_confidence,
            min_pose_presence_confidence=args.min_pose_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            min_visibility=args.min_visibility,
            tile=args.tile,
            yolo_model=yolo_model,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_max_det=args.yolo_max_det,
            yolo_imgsz=args.yolo_imgsz,
            yolo_stride=args.yolo_stride,
            yolo_pad=args.yolo_pad,
            max_crop_size=args.max_crop_size,
            yolo_pose_iou=args.yolo_pose_iou,
            yolo_fallback_fullframe=args.yolo_fallback_fullframe,
            yolo_fallback_iou=args.yolo_fallback_iou,
            draw_boxes=args.draw_boxes,
            save_coco_json=args.save_coco_json,
            json_out_dir=json_out_dir,
            debug=args.debug,
            debug_frames=args.debug_frames,
            save_debug_crops=args.save_debug_crops,
            debug_crop_dir=debug_crop_dir,
            save_frames=bool(args.save_frames),
            frames_out_dir=frames_out_dir,
            frame_format=str(args.image_format),
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
