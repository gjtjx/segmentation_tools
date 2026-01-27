import argparse
import json
import math
from pathlib import Path
from collections import deque
from typing import Any, Optional

import cv2 as _cv2
import numpy as np

cv2: Any = _cv2


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


class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return x

        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def _alpha(self, t_e, cutoff):
        r = 2 * math.pi * cutoff
        # Alpha = te / (te + tau) where tau = 1 / (2*pi*cutoff)
        #       = te / (te + 1/r)
        #       = (te * r) / (te * r + 1)
        return (r * t_e) / (1.0 + r * t_e)


class PoseSmoother:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, disable_value_check=False):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.disable_value_check = disable_value_check
        # filters[pose_idx][landmark_idx][coord_idx] -> OneEuroFilter
        self.filters: dict[int, dict[int, dict[int, OneEuroFilter]]] = {}
        self.history: dict[int, dict[int, float]] = {} # Track timestamp per pose

    def update(self, pose_result: Optional[_SimplePoseResult], timestamp: float) -> Optional[_SimplePoseResult]:
        if pose_result is None or not pose_result.pose_landmarks:
            return pose_result
        
        smoothed_poses = []
        for p_idx, landmarks in enumerate(pose_result.pose_landmarks):
            if p_idx not in self.filters:
                self.filters[p_idx] = {}
            
            smoothed_landmarks = []
            for lm_idx, lm in enumerate(landmarks):
                if lm_idx not in self.filters[p_idx]:
                    self.filters[p_idx][lm_idx] = {}

                # Coords to smooth: x, y, z
                coords = [lm.x, lm.y, lm.z]
                smoothed_coords = []
                
                # Check visibility reset
                # If visibility jumps from low to high, we might want to reset filter to avoid trail
                # But for now let's keep it simple.
                
                for c_idx, val in enumerate(coords):
                    if c_idx not in self.filters[p_idx][lm_idx]:
                        self.filters[p_idx][lm_idx][c_idx] = OneEuroFilter(
                            timestamp, val, 
                            min_cutoff=self.min_cutoff, 
                            beta=self.beta, 
                            d_cutoff=self.d_cutoff
                        )
                        smoothed_coords.append(val)
                    else:
                        f = self.filters[p_idx][lm_idx][c_idx]
                        new_val = f(timestamp, val)
                        smoothed_coords.append(new_val)
                
                smoothed_landmarks.append(_SimpleLandmark(
                    x=smoothed_coords[0],
                    y=smoothed_coords[1],
                    z=smoothed_coords[2],
                    visibility=lm.visibility,
                    presence=lm.presence
                ))
            smoothed_poses.append(smoothed_landmarks)
            
        return _SimplePoseResult(pose_landmarks=smoothed_poses)


# 33 点 BlazePose 的连接关系（与 pose_landmarker_video_vis.py 一致）
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

    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (0, 9), (0, 10), (9, 10),
    (2, 0), (5, 0),

    (15, 19), (15, 21), (17, 21),
    (16, 20), (16, 22), (18, 22),

    (29, 27), (31, 27), (29, 31),
    (30, 28), (32, 28), (30, 32),
)

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


def imread_any_path(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except (OSError, ValueError):
        return None


def create_video_writer(output_path: Path, fps: float, width: int, height: int) -> Any:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fourcc_str in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer

    raise RuntimeError(
        f"无法创建 VideoWriter 输出：{output_path}\n"
        "请检查 OpenCV 是否具备 MP4 编码能力。"
    )


def _to_pixel_coords(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return int(round(x * (width - 1))), int(round(y * (height - 1)))


def calculate_angle_3pt(a: _SimpleLandmark, b: _SimpleLandmark, c: _SimpleLandmark) -> float:
    """Calculate angle at b given points a, b, c (in degrees, 0-180)"""
    # Vectors ba and bc
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    
    # Normalize
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
        
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clamp for safety
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle_rad)


def draw_angle_gauge_fixed(
    img: np.ndarray,
    rect: tuple[int, int, int, int], # x, y, w, h of the card area
    angle: float,
    label: str,
    vec_a: tuple[float, float], # Vector from B to A (e.g. Center to Top limb)
    vec_c: tuple[float, float]  # Vector from B to C (e.g. Center to Bottom limb)
) -> np.ndarray:
    """Draw a high-end dashboard-style gauge CARD with Two Lines representing the real angle orientation"""
    annotated = img.copy()
    rx, ry, rw, rh = rect
    
    # 1. Card Background
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (30, 30, 30), -1)
    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (60, 60, 60), 1)
    
    # 2. Label (Top-Left)
    font_scale_label = rh / 140.0
    font_thick = 1
    c_label = (200, 200, 200) # Light grey
    
    (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, font_thick)
    pad_x = int(rw * 0.08)
    pad_y = int(rh * 0.08)
    label_y = ry + pad_y + txt_h
    cv2.putText(annotated, label, (rx + pad_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_label, c_label, font_thick, cv2.LINE_AA)
    
    # 3. Value (Top-Right)
    val_int = int(angle)
    val_str = f"{val_int}"
    font_scale_val = rh / 60.0 # Bigger
    font_thick_val = 1
    c_val = (255, 255, 255)
    
    (vw, vh), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale_val, font_thick_val)
    
    # Use smaller padding for value to push it more to the right edge
    pad_x_val = int(rw * 0.05) 
    val_x = rx + rw - pad_x_val - vw
    
    # Align roughly with label vertical center or slightly lower
    val_y = ry + pad_y + vh 
    
    cv2.putText(annotated, val_str, (val_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_val, c_val, font_thick_val, cv2.LINE_AA)
    
    # 4. Angle Visualization (Real Orientation)
    # Center of visualization
    vis_cx = rx + rw // 2
    vis_cy = ry + int(rh * 0.65)
    vis_radius = int(min(rw, rh) * 0.28)
    
    # Normalize vectors
    import math
    def normalize(v):
        norm = math.sqrt(v[0]**2 + v[1]**2)
        if norm < 1e-6: return (0, -1) # Default Up
        return (v[0]/norm, v[1]/norm)
        
    na = normalize(vec_a)
    nc = normalize(vec_c)
    
    p_center = (vis_cx, vis_cy)
    
    # Point A endpoint
    pa = (int(vis_cx + na[0] * vis_radius), int(vis_cy + na[1] * vis_radius))
    # Point C endpoint
    pc = (int(vis_cx + nc[0] * vis_radius), int(vis_cy + nc[1] * vis_radius))
    
    # Colors from attachment
    # Reference: Left Elbow has Blueish (Top) and Reddish (Bottom).
    # Let's standardize: Line A (e.g. arm) -> Blue, Line C (e.g. forearm) -> Red?
    # Or just constant colors.
    c_line_a = (200, 100, 50) # Blue-ish (BGR: Blue=200, G=100, R=50) -> Slate Blue
    c_line_c = (50, 50, 200)   # Red-ish (BGR: Blue=50, G=50, R=200) -> Brick Red
    c_arc = (0, 255, 0)        # Bright Green
    
    # Draw Lines
    cv2.line(annotated, p_center, pa, c_line_a, 3, cv2.LINE_AA)
    cv2.line(annotated, p_center, pc, c_line_c, 3, cv2.LINE_AA)
    
    # Draw Arc
    # Calculate angles
    # atan2(y, x) -> standard math (-pi, pi)
    ang_a_rad = math.atan2(na[1], na[0])
    ang_c_rad = math.atan2(nc[1], nc[0])
    
    ang_a_deg = math.degrees(ang_a_rad)
    ang_c_deg = math.degrees(ang_c_rad)
    
    # OpenCV ellipse draws `angle` to `endAngle` CLOCKWISE ... wait.
    # Docs: "The angles are measured in degrees. 0 degrees corresponds to the X axis direction."
    # "The arc is drawn in clockwise direction" ? 
    # With Y-down image coordinates, standard mathematical angles (Counter Clockwise from X) 
    # mean that positive angle goes Down (Clockwise).
    # So 0=Right, 90=Down, 180=Left, -90=Up.
    # This matches math.degrees(atan2(y,x)) where y is positive down.
    
    # We want to draw the INTERIOR angle.
    # calculate difference
    diff = (ang_c_deg - ang_a_deg) % 360
    
    # We constructed the scalar `angle` using calculation (usually interior angle <= 180).
    # If the vector diff matches `angle`, draw from a to c.
    # If vector diff is 360 - angle, then draw from c to a.
    
    arc_rad = int(vis_radius * 0.6)
    
    # Check if diff is roughly equal to our computed scalar angle
    draw_start = ang_a_deg
    draw_end = ang_c_deg
    
    if abs(diff - angle) > 5.0 and abs(diff - angle) < 355.0:
        # The direct path A->C is NOT the interior angle (it's the reflex 360-angle)
        # So we draw C->A
        # But wait, ellipse draws from startAngle to endAngle clockwise?
        # If we want C->A clockwise, we set start=C, end=A.
        # But maybe C->A clockwise is the REFLEX angle?
        # Let's verify:
        # If A=0 (Right), C=90 (Down). Angle=90.
        # diff (C-A) = 90. Matches angle.
        # Clockwise A->C (0 to 90) COVERS the angle 90. Correct.
        # If A=0, C=-90 (Up). Angle=90.
        # diff (C-A) = -90 = 270. Reflex.
        # So A->C clockwise covers 270. Wrong.
        # We want C->A clockwise (-90 to 0). Covers 90. Correct.
        
        # So if diff (Clockwise distance A->C) > 180, swap.
        if diff > 180:
             draw_start = ang_c_deg
             draw_end = ang_a_deg
    else:
        # diff matches angle (so A->C is interior and clockwise)
        # Or diff is very small (angle near 0).
        pass
        
    # Draw arc
    # Note: cv2.ellipse takes startAngle, endAngle. It draws from start to end CLOCKWISE.
    # Handling the wrap-around? OpenCV handles it if start > end? No.
    # Need to handle relative or allow end > start + 360?
    # Usually it draws start -> end.
    
    if draw_end < draw_start:
        draw_end += 360
        
    # Draw logic:
    # If we decided to draw from A to C: start=A, end=C.
    # If we decided to draw from C to A: start=C, end=A.
    # Above logic `if diff > 180` handles generic A->C clockwise check.
    
    # However, `diff` was (ang_c - ang_a) % 360. This is always positive 0..360.
    # It represents the clockwise sweep from A to C.
    # If this sweep > 180, it means the interior angle is on the other side (C to A).
    # So if diff > 180, start=C, end=A.
    # else start=A, end=C.
    
    if diff > 180:
         s, e = ang_c_deg, ang_a_deg
    else:
         s, e = ang_a_deg, ang_c_deg
         
    # Normalize for elliptical drawing
    if e < s: e += 360
    
    cv2.ellipse(annotated, p_center, (arc_rad, arc_rad), 0, s, e, c_arc, 2, cv2.LINE_AA)
    
    # Draw Pivot
    cv2.circle(annotated, p_center, 3, (0, 255, 0), -1, cv2.LINE_AA) # Green filled
    cv2.circle(annotated, p_center, 4, (0, 200, 0), 1, cv2.LINE_AA) # Border
    
    return annotated


def draw_limb_trails(
    img: np.ndarray,
    limb_histories: dict[int, list[tuple[float, float]]],
    width: int,
    height: int
) -> np.ndarray:
    """绘制四肢末端轨迹"""
    annotated = img.copy()
    
    # 定义四肢配置：Keypoint Index, Color (BGR)
    # L_Index(19), R_Index(20), L_Foot(31), R_Foot(32)
    # Colors: Cyan, Orange, Green, Magenta
    limbs_config = {
        19: ((255, 255, 0), "L-Hand"),    # Cyan (actually Yellow in my previous fix, kept Yellow for left)
        20: ((0, 165, 255), "R-Hand"),    # Orange
        31: ((0, 255, 0),   "L-Foot"),    # Green
        32: ((255, 0, 255), "R-Foot")     # Magenta
    }

    base_thick = max(1, int(min(width, height) * 0.002))
    
    for lm_idx, (color, _) in limbs_config.items():
        if lm_idx not in limb_histories or len(limb_histories[lm_idx]) < 2:
            continue
            
        history = limb_histories[lm_idx]
        pts = []
        for x, y in history:
            px, py = _to_pixel_coords(x, y, width, height)
            pts.append([px, py])
        
        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
        
        # 描边 + 主色
        cv2.polylines(annotated, [pts_np], isClosed=False, color=(0,0,0), thickness=base_thick+1, lineType=cv2.LINE_AA)
        cv2.polylines(annotated, [pts_np], isClosed=False, color=color, thickness=base_thick, lineType=cv2.LINE_AA)
        
        # 绘制末端点
        last_px, last_py = pts[-1]
        cv2.circle(annotated, (last_px, last_py), base_thick+2, color, -1, cv2.LINE_AA)

    return annotated


def calculate_cog(landmarks: list[_SimpleLandmark]) -> Optional[tuple[float, float]]:
    """简单的人体加权加权重心估算 (Based on Dempster's body segment data roughly)"""
    if not landmarks or len(landmarks) < 33:
        return None

    def get_vec(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y])

    def mid_vec(idx1, idx2):
        return (get_vec(idx1) + get_vec(idx2)) * 0.5

    # 简化的肢体段质心位置与权重
    # Segment Point, Weight
    segments = []
    
    # Head & Neck (~7.9%): 使用鼻子或两耳中点
    # Nose(0)
    segments.append((get_vec(0), 0.079))
    
    # Torso (Trunk) (~48.6%): 肩的中点与髋的中点的中点
    # L_Sh(11), R_Sh(12), L_Hip(23), R_Hip(24)
    center_shoulders = mid_vec(11, 12)
    center_hips = mid_vec(23, 24)
    # 躯干质心通常偏上，但简单取中点
    center_torso = (center_shoulders + center_hips) * 0.5
    segments.append((center_torso, 0.486))

    # Upper Arms (~2.7% each): Shoulder to Elbow
    # L: 11-13, R: 12-14
    segments.append((mid_vec(11, 13), 0.027))
    segments.append((mid_vec(12, 14), 0.027))

    # Forearms (~1.5% each): Elbow to Wrist
    # L: 13-15, R: 14-16
    segments.append((mid_vec(13, 15), 0.015))
    segments.append((mid_vec(14, 16), 0.015))
    
    # Hands (~0.6% each): Wrist (or ~Wrist)
    segments.append((get_vec(15), 0.006))
    segments.append((get_vec(16), 0.006))

    # Thighs (~9.8% each): Hip to Knee
    # L: 23-25, R: 24-26
    segments.append((mid_vec(23, 25), 0.098))
    segments.append((mid_vec(24, 26), 0.098))

    # Shanks/Calves (~4.5% each): Knee to Ankle
    # L: 25-27, R: 26-28
    segments.append((mid_vec(25, 27), 0.045))
    segments.append((mid_vec(26, 28), 0.045))

    # Feet (~1.4% each): Ankle (or Ankle to Toe)
    # L: 27, R: 28
    segments.append((get_vec(27), 0.014))
    segments.append((get_vec(28), 0.014))

    cog = np.zeros(2)
    total_w = 0.0
    for vec, w in segments:
        cog += vec * w
        total_w += w
    
    if total_w <= 0:
        return None
        
    cog /= total_w
    return (cog[0], cog[1])



def draw_cog_dashboard(
    img: np.ndarray,
    cog_offset_history: list[tuple[int, float]], # List of (frame_idx, relative_offset)
    rect: tuple[int, int, int, int], # (x, y, w, h)
) -> np.ndarray:
    """Draw Gravity Center dashboard in the specified rectangle"""
    dash_x, dash_y, dash_w, dash_h = rect
    
    annotated = img.copy()
    
    # Draw Background
    cv2.rectangle(annotated, (dash_x, dash_y), (dash_x + dash_w, dash_y + dash_h), (10, 10, 10), -1)
    # Border
    cv2.rectangle(annotated, (dash_x, dash_y), (dash_x + dash_w, dash_y + dash_h), (50, 50, 50), 1)
    
    # Fonts
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_title = 0.4
    font_scale_tick = 0.35
    
    # 1. Slider (Top part of dashboard)
    slider_h = 40
    slider_y_center = dash_y + 25
    
    # Track line (White)
    # Give side margins
    slider_margin_x = 15
    track_start_x = dash_x + slider_margin_x
    track_end_x = dash_x + dash_w - slider_margin_x
    track_center_x = (track_start_x + track_end_x) // 2
    track_width = track_end_x - track_start_x
    
    cv2.line(annotated, (track_start_x, slider_y_center), (track_end_x, slider_y_center), (150, 150, 150), 2)
    # Ticks
    for t_x in [track_start_x, track_center_x, track_end_x]:
        cv2.line(annotated, (t_x, slider_y_center - 10), (t_x, slider_y_center + 10), (200, 200, 200), 2)
        
    # Title "Gravity Center"
    # Centered or Left? Let's Center it for sidebar.
    title_str = "Gravity Center"
    (tw, th), _ = cv2.getTextSize(title_str, font_face, font_scale_title, 1)
    title_x = dash_x + (dash_w - tw) // 2
    cv2.putText(annotated, title_str, (title_x, dash_y + 15), font_face, font_scale_title, (200, 200, 200), 1, cv2.LINE_AA)
        
    # Current Value Marker (Red dot at center)
    # Note: Image is RGB
    cv2.circle(annotated, (track_center_x, slider_y_center), 4, (255, 0, 0), -1) 
    
    # Get current offset
    curr_offset = 0.0
    if cog_offset_history:
        _, curr_offset = cog_offset_history[-1]
    
    # Scale: +/- 2.0 hip widths
    range_max = 2.0 
    scale_x = (track_width / 2.0) / range_max
    
    slide_px = int(curr_offset * scale_x)
    cog_slider_x = int(track_center_x + slide_px)
    cog_slider_x = max(track_start_x, min(track_end_x, cog_slider_x))
    
    # Draw Green Circle for Current CoG
    cv2.circle(annotated, (cog_slider_x, slider_y_center), 6, (0, 255, 0), 2)
    
    # 2. Scrolling Graph (Bottom part)
    # Area
    # Need space on LEFT for Frame Numbers (Y-Axis Labels)
    # Need space on BOTTOM for X-Axis Labels (-2, -1, 0, 1, 2)
    
    graph_y_start = dash_y + 50
    # Reserve more space at bottom for labels
    margin_bottom = 25
    graph_y_end = dash_y + dash_h - margin_bottom 
    graph_h = graph_y_end - graph_y_start
    
    # Margins inside dashboard
    margin_left_labels = 40 # Increased space for "1500"
    margin_right = 5
    
    # ALIGNMENT FIX:
    # Ensure Graph 0-center aligns with Slider 0-center.
    # Slider center is track_center_x = (track_start + track_end) / 2
    # track_start/end are symmetric in dash_w. So track_center_x approx dash_x + dash_w/2.
    # Graph drawing area must be symmetric around track_center_x to align 0.
    
    center_x_g = track_center_x
    
    # Calculate max possible half-width for graph given margins
    # We are bounded by margin_left_labels on left, and margin_right on right.
    # dist_to_left = center_x_g - (dash_x + margin_left_labels)
    # dist_to_right = (dash_x + dash_w - margin_right) - center_x_g
    # half_width = min(dist_to_left, dist_to_right)
    
    dist_l = center_x_g - (dash_x + margin_left_labels)
    dist_r = (dash_x + dash_w - margin_right) - center_x_g
    graph_half_w = min(dist_l, dist_r)
    
    graph_x = center_x_g - graph_half_w
    graph_w_draw = graph_half_w * 2
    
    # Override safety
    if graph_w_draw < 10: 
        graph_w_draw = 10 
        graph_x = dash_x + 5
    
    # Draw Grid
    # Vertical center line (RGB Red = 255, 0, 0)
    cv2.line(annotated, (center_x_g, graph_y_start), (center_x_g, graph_y_start + graph_h), (255, 0, 0), 2) 
    
    # Horizontal grid lines (background)
    for i in range(5):
        gy = graph_y_start + int(graph_h * i / 4)
        cv2.line(annotated, (graph_x, gy), (graph_x + graph_w_draw, gy), (40, 40, 40), 1)
        
    # Vertical grid lines (background) labels
    # Recalculate scale for graph using new width
    scale_x_graph = (graph_w_draw / 2.0) / range_max
    
    for i in range(-2, 3): # -2, -1, 0, 1, 2
        px = int(center_x_g + i * scale_x_graph)
        
        # Draw Line
        if px >= graph_x and px <= graph_x + graph_w_draw:
             cv2.line(annotated, (px, graph_y_start), (px, graph_y_end), (40, 40, 40), 1)
             
             # Draw Tick on bottom axis
             cv2.line(annotated, (px, graph_y_end), (px, graph_y_end + 3), (150, 150, 150), 1)
             
             # X-Axis Labels (Small)
             label_x = f"{i}"
             (lw, lh), _ = cv2.getTextSize(label_x, font_face, 0.35, 1)
             # Draw below
             cv2.putText(annotated, label_x, (px - lw//2, graph_y_end + 12), font_face, 0.35, (150, 150, 150), 1)

    # Y-Axis Label "Frame" - put at top left corner of graph area
    # Or rotate it? Text "Frame"
    cv2.putText(annotated, "Frame", (dash_x + 2, graph_y_start - 5), font_face, 0.35, (150, 150, 150), 1)
    # X-Axis Label "Offset" - put at bottom right or center
    cv2.putText(annotated, "Offset", (dash_x + dash_w - 40, graph_y_end + 20), font_face, 0.35, (150, 150, 150), 1)

    # 3. Draw Graph Data
    pts_curve = []
    
    history_len = len(cog_offset_history)
    frame_window = 150 
    
    if history_len > 0:
        last_frame_idx = cog_offset_history[-1][0]
        start_frame_view = last_frame_idx - frame_window
        
        # Draw Y-axis ticks (Right Aligned in margin area)
        for i in range(5):
            tick_frame = int(start_frame_view + (i / 4.0) * frame_window)
            if tick_frame < 0: count_str = "0"
            else: count_str = str(tick_frame)
            
            tick_y = int(graph_y_end - (i / 4.0) * graph_h)
            
            # Measure text
            (tw, th), _ = cv2.getTextSize(count_str, font_face, font_scale_tick, 1)
            # Right align to graph_x - 3
            tx = graph_x - 3 - tw
            ty = tick_y + th // 2
            
            cv2.putText(annotated, count_str, (tx, ty), font_face, font_scale_tick, (150, 150, 150), 1)

        points_in_view = [p for p in cog_offset_history if p[0] >= start_frame_view]
        
        if points_in_view:
            for f_idx, offset in points_in_view:
                # X: 
                px = int(center_x_g + offset * scale_x_graph)
                
                # Y: 
                norm_y = (f_idx - start_frame_view) / float(frame_window)
                py = int(graph_y_end - norm_y * graph_h)
                
                if graph_y_start <= py <= graph_y_end:
                    pts_curve.append([px, py])
            
            if len(pts_curve) > 1:
                # Simple clamping:
                pts_clamped = []
                for p in pts_curve:
                    cx = max(graph_x, min(graph_x + graph_w_draw, p[0]))
                    cy = max(graph_y_start, min(graph_y_end, p[1]))
                    pts_clamped.append([cx, cy])
                    
                cv2.polylines(annotated, [np.array(pts_clamped, np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
        
    return annotated


def draw_cog_trail_on_body(
    img: np.ndarray,
    cog_history: list[tuple[float, float]],
    width: int,
    height: int
) -> np.ndarray:
    """Draw CoG point and trail on the body safely"""
    annotated = img.copy()
    if not cog_history:
        return annotated

    # Draw history trail
    pts = []
    for x, y in cog_history:
        px, py = _to_pixel_coords(x, y, width, height)
        pts.append([px, py])
    
    if len(pts) > 1:
        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts_np], False, (0, 255, 255), 2, cv2.LINE_AA) # Yellow trail
        
    # Draw current
    cx, cy = pts[-1]
    # Draw Crosshair
    cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1) # Red center
    cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), 2) # White rim
    cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 1)
    cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 1)
    
    return annotated


def draw_angle_overlay(
    img: np.ndarray,
    center: tuple[int, int],
    angle: float,
    vec_a: tuple[float, float],
    vec_c: tuple[float, float]
) -> np.ndarray:
    """Draw angle visualization directly on the joint (AR style)"""
    annotated = img.copy()
    cx, cy = center
    radius = 30
    
    # 1. Visualization Lines (Semi-transparent look simulation)
    # We can't do alpha blending easily with just opencv drawing primitives efficiently in loop,
    # but we can try drawing minimal clean lines.
    
    import math
    def normalize(v):
        norm = math.sqrt(v[0]**2 + v[1]**2)
        if norm < 1e-6: return (0, -1)
        return (v[0]/norm, v[1]/norm)
    
    na = normalize(vec_a)
    nc = normalize(vec_c)
    
    # Endpoints
    pa = (int(cx + na[0] * radius), int(cy + na[1] * radius))
    pc = (int(cx + nc[0] * radius), int(cy + nc[1] * radius))
    
    # Draw Lines (Thick, distinctive colors)
    cv2.line(annotated, (cx, cy), pa, (255, 100, 0), 2, cv2.LINE_AA) # Blue-ish BGR
    cv2.line(annotated, (cx, cy), pc, (0, 100, 255), 2, cv2.LINE_AA) # Red-ish BGR
    
    # 2. Angle Arc
    # Calculate angles for ellipse
    ang_a_rad = math.atan2(na[1], na[0])
    ang_c_rad = math.atan2(nc[1], nc[0])
    ang_a_deg = math.degrees(ang_a_rad)
    ang_c_deg = math.degrees(ang_c_rad)
    
    # Ensure drawing shortest arc
    diff = (ang_c_deg - ang_a_deg) % 360
    if diff > 180:
        # Swap start/end to draw the smaller angle
        start_angle = ang_c_deg
        end_angle = ang_c_deg + (360 - diff)
    else:
        start_angle = ang_a_deg
        end_angle = ang_c_deg
        
    cv2.ellipse(annotated, (cx, cy), (15, 15), 0, start_angle, end_angle, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 3. Text Value
    # Check text size
    label = str(int(angle))
    font_scale = 0.6
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    
    # Draw Text with outlining for readability against any background
    # Position: Slightly offset from center to avoid covering the joint dot
    tx = cx + 15
    ty = cy - 15
    
    cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA) # Outline
    cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA) # Text
    
    return annotated

def draw_bounding_box(img: np.ndarray, pose: Optional[_SimplePoseResult], width: int, height: int, cog: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Draw dynamic bounding box around the pose"""
    annotated = img.copy()
    if pose is None or not pose.pose_landmarks:
        return annotated
        
    landmarks = pose.pose_landmarks[0]
    
    # Collect valid points
    xs = []
    ys = []
    
    for lm in landmarks:
        if lm.visibility > 0.3: # Only confident points
            xs.append(lm.x * width)
            ys.append(lm.y * height)
            
    if not xs:
        return annotated
        
    min_x, max_x = int(min(xs)), int(max(xs))
    min_y, max_y = int(min(ys)), int(max(ys))
    
    # Calculate box dimensions
    box_w = max_x - min_x
    box_h = max_y - min_y
    
    # Add padding (20% expansion means 10% on each side)
    pad_x = int(max(10, box_w * 0.1)) 
    pad_y = int(max(10, box_h * 0.1))
    
    min_x = max(0, min_x - pad_x)
    min_y = max(0, min_y - pad_y)
    max_x = min(width, max_x + pad_x)
    max_y = min(height, max_y + pad_y)
    
    # Draw Green Box (The "Green Layout")
    # Color: Bright Green (0, 255, 0)
    cv2.rectangle(annotated, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
    
    # Calculate Geometric Center
    # (The Green Crosshair center)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    
    # Draw Green Crosshair (Geometric Center)
    # Using thinner green lines
    cv2.line(annotated, (center_x, min_y), (center_x, max_y), (0, 255, 0), 1)
    cv2.line(annotated, (min_x, center_y), (max_x, center_y), (0, 255, 0), 1)
    
    # Draw Green Dot at Geometric Center
    cv2.circle(annotated, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Draw CoG (Purple Crosshair)
    # The physical balance point
    if cog:
        cog_x, cog_y = cog
        cx_px = int(cog_x * width)
        cy_px = int(cog_y * height)
        
        # Color: Purple (Magenta: 255, 0, 255)
        # In RGB: (255, 0, 255)
        c_cog = (255, 0, 255) 
         
        # Draw Crosshair
        # Extend slightly beyond box or full axes? Let's keep within box for clarity or full vertical.
        # Let's draw full vertical/horizontal within the box limits
        
        # Clamp to box limits for drawing lines
        start_x = max(min_x, 0)
        end_x = min(max_x, width)
        start_y = max(min_y, 0)
        end_y = min(max_y, height)
        
        cv2.line(annotated, (cx_px, start_y), (cx_px, end_y), c_cog, 2)
        cv2.line(annotated, (start_x, cy_px), (end_x, cy_px), c_cog, 2)
        
        # Draw Dot
        cv2.circle(annotated, (cx_px, cy_px), 6, c_cog, -1)
        cv2.circle(annotated, (cx_px, cy_px), 8, (255, 255, 255), 1) # Outline
    
    return annotated

def draw_pose_angles(img: np.ndarray, pose: Optional[_SimplePoseResult], width: int, height: int) -> tuple[np.ndarray, Optional[tuple[int, int, int, int]]]:
    """Draw angles in AR Overlay style directly on body"""
    annotated = img.copy()
    
    # We no longer use a sidebar rect for angles, so return None for rect
    # This signals the CoG drawer to use its fallback position
    
    if pose is None or not pose.pose_landmarks:
        return annotated, None
        
    landmarks = pose.pose_landmarks[0]
    if len(landmarks) < 33:
        return annotated, None

    joints_config = [
        # Label, A, B(Center), C
        ("L.Shldr", 13, 11, 23),
        ("L.Elbow", 11, 13, 15), 
        #("L.Wrist", 13, 15, 19), # Wrists often clutter, optional
        ("L.Hip",   11, 23, 25), 
        ("L.Knee",  23, 25, 27), 
        ("L.Ankel", 25, 27, 31), 
        
        ("R.Shldr", 14, 12, 24), 
        ("R.Elbow", 12, 14, 16), 
        #("R.Wrist", 14, 16, 20), 
        ("R.Hip",   12, 24, 26), 
        ("R.Knee",  24, 26, 28), 
        ("R.Ankel", 26, 28, 32), 
    ]
    
    for i, (label, idx_a, idx_b, idx_c) in enumerate(joints_config):
        a = landmarks[idx_a]
        b = landmarks[idx_b]
        c = landmarks[idx_c]
        
        if b.visibility > 0.5:
            # Calculate pixel coords
            ax, ay = a.x * width, a.y * height
            bx, by = b.x * width, b.y * height
            cx_val, cy_val = c.x * width, c.y * height
            
            # Vector
            v_a = (ax - bx, ay - by)
            v_c = (cx_val - bx, cy_val - by)
            
            angle = calculate_angle_3pt(a, b, c)
            
            annotated = draw_angle_overlay(
                annotated,
                (int(bx), int(by)),
                angle,
                v_a,
                v_c
            )
        
    return annotated, None


def draw_pose_landmarks_rgb(
    rgb_image: np.ndarray,
    pose_result: Optional[Any],
    *,
    min_visibility: float = 0.0,
) -> np.ndarray:
    annotated = rgb_image.copy()
    if pose_result is None:
        return annotated

    height, width = annotated.shape[:2]

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
        return center_color

    line_thickness = max(1, int(round(min(width, height) * 0.003)))
    point_radius = max(2, int(round(min(width, height) * 0.006)))

    for pose_landmarks in pose_result.pose_landmarks:
        pts: list[Optional[tuple[int, int]]] = [None] * len(pose_landmarks)

        for i, lm in enumerate(pose_landmarks):
            vis = float(getattr(lm, "visibility", 1.0))
            if vis < min_visibility:
                continue
            x = float(lm.x)
            y = float(lm.y)
            px, py = _to_pixel_coords(x, y, width, height)
            if 0 <= px < width and 0 <= py < height:
                pts[i] = (px, py)

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


def _load_pose_from_coco_json(
    json_path: Path,
    *,
    target_w: int,
    target_h: int,
    scale_to_background: bool,
) -> Optional[_SimplePoseResult]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    anns = data.get("annotations", []) or []
    if not anns:
        return None

    src_w = None
    src_h = None
    images = data.get("images", []) or []
    if images and isinstance(images, list):
        src_w = images[0].get("width")
        src_h = images[0].get("height")

    if scale_to_background and src_w and src_h:
        sx = float(target_w) / float(src_w)
        sy = float(target_h) / float(src_h)
    else:
        sx = 1.0
        sy = 1.0

    poses: list[list[_SimpleLandmark]] = []
    for ann in anns:
        kpts = ann.get("keypoints", []) or []
        if not kpts:
            continue
        pts: list[_SimpleLandmark] = []
        for i in range(0, len(kpts), 3):
            x = float(kpts[i]) * sx
            y = float(kpts[i + 1]) * sy
            v = int(kpts[i + 2]) if i + 2 < len(kpts) else 0
            if v <= 0:
                vis = 0.0
            elif v == 1:
                vis = 0.5
            else:
                vis = 1.0
            nx = x / float(target_w)
            ny = y / float(target_h)
            pts.append(_SimpleLandmark(nx, ny, visibility=vis, presence=vis))
        poses.append(pts)

    if not poses:
        return None
    return _SimplePoseResult(pose_landmarks=poses)


def _ease(t: float, mode: str) -> float:
    t = max(0.0, min(1.0, float(t)))
    if mode == "linear":
        return t
    if mode == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    return t


def _interp_landmark(a: _SimpleLandmark, b: _SimpleLandmark, t: float, mode: str) -> _SimpleLandmark:
    tt = _ease(t, mode)
    x = a.x + (b.x - a.x) * tt
    y = a.y + (b.y - a.y) * tt
    z = a.z + (b.z - a.z) * tt
    vis = a.visibility + (b.visibility - a.visibility) * tt
    pres = a.presence + (b.presence - a.presence) * tt
    return _SimpleLandmark(x, y, z=z, visibility=vis, presence=pres)


def _interpolate_pose_results(
    a: Optional[_SimplePoseResult],
    b: Optional[_SimplePoseResult],
    *,
    steps: int,
    mode: str,
) -> list[Optional[_SimplePoseResult]]:
    """在两帧姿态之间插值，返回中间 steps 帧（不含端点）。"""
    if steps <= 0:
        return []
    if a is None or b is None:
        # 任一端缺失，直接返回空（由调用方决定是否复用上一帧）
        return []

    poses_a = a.pose_landmarks
    poses_b = b.pose_landmarks
    if not poses_a or not poses_b:
        return []

    num_poses = min(len(poses_a), len(poses_b))
    if num_poses <= 0:
        return []

    out: list[Optional[_SimplePoseResult]] = []
    for i in range(1, steps + 1):
        t = i / float(steps + 1)
        interpolated: list[list[_SimpleLandmark]] = []
        for p_idx in range(num_poses):
            lm_a = poses_a[p_idx]
            lm_b = poses_b[p_idx]
            num_kpts = min(len(lm_a), len(lm_b))
            if num_kpts <= 0:
                continue
            kpts: list[_SimpleLandmark] = []
            for k in range(num_kpts):
                kpts.append(_interp_landmark(lm_a[k], lm_b[k], t, mode))
            interpolated.append(kpts)
        out.append(_SimplePoseResult(pose_landmarks=interpolated) if interpolated else None)
    return out


def iter_json_files(input_dir: Path) -> list[Path]:
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    return sorted(files, key=lambda p: p.name)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="读取姿态点 JSON 序列，叠加到背景图像或视频上并输出视频"
    )
    p.add_argument("--json_dir", "-i", type=str, required=True, help="姿态点 JSON 文件夹")
    
    # Allow either --background (image) or --bg_video (video)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--background", "-b", type=str, help="背景图像路径")
    group.add_argument("--bg_video", "-v", type=str, help="背景视频路径")

    p.add_argument("--out", "-o", type=str, required=True, help="输出视频路径 (.mp4)")
    p.add_argument("--fps", type=float, default=30.0, help="输出视频 FPS（默认 30）")
    p.add_argument(
        "--min_visibility",
        type=float,
        default=0.0,
        help="绘制时过滤低可见度关键点（默认 0.0）",
    )
    p.add_argument(
        "--scale_to_background",
        action="store_true",
        help="根据 JSON 中的 width/height 缩放到背景图尺寸",
    )
    p.add_argument(
        "--insert_frames",
        type=int,
        default=0,
        help="在相邻两帧之间插入多少帧（默认 0）",
    )
    p.add_argument(
        "--interp_mode",
        type=str,
        default="smoothstep",
        choices=["linear", "smoothstep"],
        help="插值方式：linear 或 smoothstep（默认 smoothstep）",
    )
    p.add_argument(
        "--smooth_landmarks",
        action="store_true",
        help="启用 OneEuroFilter 对关键点进行平滑（缓解抖动）",
    )
    p.add_argument(
        "--smooth_min_cutoff",
        type=float,
        default=1.0,
        help="平滑滤波 min_cutoff (默认 1.0)。值越小越平滑但延迟越高；值越大越灵敏。",
    )
    p.add_argument(
        "--smooth_beta",
        type=float,
        default=0.0,
        help="平滑滤波 beta (默认 0.0)。值越大，快速运动时的延迟越小（但可能增加抖动）。",
    )
    p.add_argument(
        "--smooth_d_cutoff",
        type=float,
        default=1.0,
        help="平滑滤波 derivate_cutoff (默认 1.0)。通常不需要修改。",
    )
    p.add_argument(
        "--draw_cog",
        action="store_true",
        help="计算并绘制人体重心及轨迹",
    )
    p.add_argument(
        "--cog_trail_length",
        type=int,
        default=30,
        help="重心轨迹保留帧数（默认 30），0 表示无限长",
    )
    p.add_argument(
        "--draw_angles",
        action="store_true",
        help="在关键关节处绘制角度仪表盘（0-180度）",
    )
    p.add_argument(
        "--draw_limb_trails",
        action="store_true",
        help="绘制手脚末端轨迹（左黄/右橙/左绿/右紫）",
    )
    p.add_argument(
        "--limb_trail_length",
        type=int,
        default=30,
        help="手脚轨迹长度（默认 30）",
    )
    p.add_argument(
        "--draw_ghosts",
        action="store_true",
        help="绘制动作残影（频闪效果）",
    )
    p.add_argument(
        "--ghost_interval",
        type=int,
        default=10,
        help="残影采样间隔帧数（默认 10）",
    )
    p.add_argument(
        "--ghost_step",
        type=int,
        default=5,
        help="每次绘制保留最近 N 个残影（默认 5）",
    )
    p.add_argument(
        "--draw_bbox",
        action="store_true",
        help="绘制人体动态边界框（Bounding Box）及几何中心",
    )
    p.add_argument(
        "--min_y_center",
        type=float,
        default=0.0,
        help="过滤目标中心 Y 坐标小于此值的姿态（0.0-1.0，默认为 0.0 不过滤）。防止处理图像顶部边缘的误检或半身。",
    )
    p.add_argument(
        "--max_y_center",
        type=float,
        default=1.0,
        help="过滤目标中心 Y 坐标大于此值的姿态（0.0-1.0，默认为 1.0 不过滤）。防止处理图像底部边缘的误检或半身。",
    )
    p.add_argument(
        "--keep_largest_target",
        action="store_true",
        help="仅保留画面中面积最大的目标（有助于去除背景杂乱人物）",
    )
    p.add_argument(
        "--keep_highest_target",
        action="store_true",
        help="仅保留画面中最靠近顶端（Y最小）的目标",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    json_dir = Path(args.json_dir)
    out_path = Path(args.out)

    if not json_dir.exists() or not json_dir.is_dir():
        raise FileNotFoundError(f"找不到 JSON 文件夹：{json_dir}")

    # Validate background source
    if args.background:
        bg_path = Path(args.background)
        if not bg_path.exists():
            raise FileNotFoundError(f"找不到背景图像：{bg_path}")
    elif args.bg_video:
        bg_video_path = Path(args.bg_video)
        if not bg_video_path.exists():
            raise FileNotFoundError(f"找不到背景视频：{bg_video_path}")

    if args.fps <= 0:
        raise ValueError("--fps 必须 > 0")
    if args.insert_frames < 0:
        raise ValueError("--insert_frames 必须 >= 0")

    # 允许 -o 传目录或 '.'，自动生成输出文件名
    if out_path.suffix.lower() != ".mp4":
        if out_path.exists() and out_path.is_dir():
            out_dir = out_path
        else:
            # 例如传入 '.' 或不带后缀路径时，视为目录
            out_dir = out_path
        out_path = out_dir / f"{json_dir.name}_overlay.mp4"

    json_files = iter_json_files(json_dir)
    if not json_files:
        raise RuntimeError(f"未找到 JSON 文件：{json_dir}")

    # Initialize Background Source
    cap = None
    bg_static_rgb = None
    
    if args.bg_video:
        if not Path(args.bg_video).exists():
             raise RuntimeError(f"背景视频不存在：{args.bg_video}")
        cap = cv2.VideoCapture(args.bg_video)
        if not cap.isOpened():
             raise RuntimeError(f"无法打开背景视频：{args.bg_video}")
             
        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError(f"背景视频读取失败（首帧）：{args.bg_video}")
        
        # Override dimensions from video
        h, w = frame0.shape[:2]
        bg_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB) # Current frame buffer
        
        # Reset to beginning? No, we just read frame 0 which corresponds to json 0.
        # But loop below will read next frame for next iteration.
        # So we should treat frame0 as the frame for the first json.
        # We need to handle the first iteration carefully or rewind.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    else:
        bg_bgr = imread_any_path(args.background)
        if bg_bgr is None:
            raise RuntimeError(f"读取背景图失败：{args.background}")
        bg_static_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
        h, w = bg_static_rgb.shape[:2]
        bg_rgb = bg_static_rgb.copy() # Current frame buffer

    # Determine Output Dimensions with Dashboard
    out_w = w
    out_h = h
    
    # Calculate default dashboard layout vertical flow for CoG placement
    # Matches draw_pose_angles logic
    _card_w = max(70, int(w * 0.075))
    _dash_w = 2 * _card_w + 5
    _start_x = w - _dash_w - 10
    default_angle_rect = (_start_x, 10, _dash_w, 0)

    # CoG Dashboard moves to sidebar, so no vertical extension needed.
    # if args.draw_cog:
    #    print("CoG Dashboard: ENABLED (Video will be extended vertically)")
    #    out_h += dash_h

    writer = create_video_writer(out_path, float(args.fps), out_w, out_h)

    smoother = None
    if args.smooth_landmarks:
        smoother = PoseSmoother(
            min_cutoff=args.smooth_min_cutoff,
            beta=args.smooth_beta,
            d_cutoff=args.smooth_d_cutoff
        )
        print(f"Smoothing enabled: min_cutoff={args.smooth_min_cutoff}, beta={args.smooth_beta}")

    prev_pose: Optional[_SimplePoseResult] = None
    prev_frame_written = False

    # Assuming constant FPS for filter timestamp
    frame_duration = 1.0 / float(args.fps)
    current_time = 0.0

    cog_history: list[tuple[float, float]] = [] # Real (x, y)
    
    # Store normalized horizontal offset for dashboard
    # (frame_idx, offset_val)
    cog_offset_history: list[tuple[int, float]] = []
    
    # Limb history: key=pixel_idx, val=list of (x,y)
    limb_history: dict[int, list[tuple[float, float]]] = {
        19: [], 20: [], 31: [], 32: []
    }
    
    # Ghosts
    ghost_poses: deque = deque(maxlen=args.ghost_step if args.ghost_step > 0 else 5)
    frame_count = 0

    def write_output_frame(f_rgb, _writer, _args, _w, _h, _cog_hist, _cog_offset_hist, _angle_rect):
        """Helper to append dashboard if needed and write frame"""
        final_frame = f_rgb
        if _args.draw_cog:
            # Draw Trail on Body first
            final_frame = draw_cog_trail_on_body(final_frame, _cog_hist, _w, _h)
            
            # Draw Dashboard in Sidebar (Below Angles)
            # If angles are drawn in overlay mode (angle_rect is None), we place CoG dashboard at bottom center
            
            cog_h = 320 # Standard height
            cog_w = int(_w * 0.3) # 30% of width
            if cog_w < 300: cog_w = 300 # Min width
            if cog_w > _w: cog_w = _w
            
            # Position: Bottom Right with opacity
            cog_x = _w - cog_w - 20
            cog_y = _h - cog_h - 20
            
            # If using Sidebar layout (angle_rect provided)
            if _angle_rect:
                ax, ay, aw, ah = _angle_rect
                cog_x = ax
                cog_y = ay + ah + 10 
                cog_w = aw
                cog_h = 600
                if cog_y + cog_h > _h:
                    cog_h = max(100, _h - cog_y - 10)
            
            # Draw semi-transparent background if floating
            if _angle_rect is None:
                # Create overlay
                overlay = final_frame.copy()
                # Use standard draw
                # We need to make sure draw_cog_dashboard doesn't assume opaque background?
                # It draws a filled rectangle (10,10,10).
                # We can draw it on overlay then blend.
                pass

            # Since draw_cog_dashboard draws a solid background, let's just draw it.
            # If we want transparency, we act on the resulting pixels.
            
            # Only draw if valid rect
            if cog_w > 10 and cog_h > 10:
                if _angle_rect is None:
                     # Render to temp buffer to alpha blend
                     temp = final_frame.copy()
                     temp = draw_cog_dashboard(temp, _cog_offset_hist, (cog_x, cog_y, cog_w, cog_h))
                     alpha = 0.8
                     final_frame = cv2.addWeighted(temp, alpha, final_frame, 1.0 - alpha, 0)
                else:
                     final_frame = draw_cog_dashboard(final_frame, _cog_offset_hist, (cog_x, cog_y, cog_w, cog_h))

        # Convert and write
        f_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        _writer.write(f_bgr)

    for json_path in json_files:
        # Update Background if Video
        if cap is not None:
             ret, frame_bgr = cap.read()
             if ret:
                 bg_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
             else:
                 # Video ended? Keep last frame or loop?
                 # Let's keep last frame (do nothing to bg_rgb)
                 pass
        elif bg_static_rgb is not None:
             bg_rgb = bg_static_rgb.copy() # Reset to clean static BG

        pose = _load_pose_from_coco_json(
            json_path,
            target_w=w,
            target_h=h,
            scale_to_background=bool(args.scale_to_background),
        )

        # ---------------------------------------------------------------------
        # Pre-process: Keep Largest Target Only
        # ---------------------------------------------------------------------
        if args.keep_largest_target and pose and pose.pose_landmarks and len(pose.pose_landmarks) > 0:
            best_idx = -1
            max_area = -1.0
            
            for idx, landmarks in enumerate(pose.pose_landmarks):
                xs = [lm.x for lm in landmarks if lm.visibility > 0.3]
                ys = [lm.y for lm in landmarks if lm.visibility > 0.3]
                
                if xs and ys:
                   w_box = max(xs) - min(xs)
                   h_box = max(ys) - min(ys)
                   area = w_box * h_box
                   if area > max_area:
                       max_area = area
                       best_idx = idx
            
            if best_idx >= 0:
                pose.pose_landmarks = [pose.pose_landmarks[best_idx]]

        # ---------------------------------------------------------------------
        # Pre-process: Keep Highest Target Only
        # ---------------------------------------------------------------------
        if args.keep_highest_target and pose and pose.pose_landmarks and len(pose.pose_landmarks) > 0:
            best_idx = -1
            min_y_center = 99999.0
            
            for idx, landmarks in enumerate(pose.pose_landmarks):
                ys = [lm.y for lm in landmarks if lm.visibility > 0.3]
                if ys:
                   cy = (min(ys) + max(ys)) / 2.0
                   if cy < min_y_center:
                       min_y_center = cy
                       best_idx = idx
            
            if best_idx >= 0:
                pose.pose_landmarks = [pose.pose_landmarks[best_idx]]

        # ---------------------------------------------------------------------
        # Pre-process Filtering: Center Y Threshold
        # 如果目标框（或关键点）的几何中心 Y 坐标小于阈值（例如位于顶部边缘），则将其视为无效姿态
        # ---------------------------------------------------------------------
        if (args.min_y_center > 0.0 or args.max_y_center < 1.0) and pose and pose.pose_landmarks:
            valid_ys = [lm.y for lm in pose.pose_landmarks[0] if lm.visibility > 0.3]
            if valid_ys:
                min_y_val = min(valid_ys)
                max_y_val = max(valid_ys)
                center_y_norm = (min_y_val + max_y_val) / 2.0
                
                if center_y_norm < args.min_y_center:
                    # 过滤掉该目标（过顶）
                    pose = None
                elif center_y_norm > args.max_y_center:
                    # 过滤掉该目标（过底）
                    pose = None
        # ---------------------------------------------------------------------

        if smoother is not None:
             pose = smoother.update(pose, current_time)
        
        current_time += frame_duration
        frame_count += 1
        
        # Ghost update
        if args.draw_ghosts and pose and pose.pose_landmarks:
            # Add to ghost list every interval frames
            if frame_count % args.ghost_interval == 0:
                ghost_poses.append(pose)

        # Calculate CoG if pose exists
        if (args.draw_cog or args.draw_limb_trails) and pose and pose.pose_landmarks:
            landmarks = pose.pose_landmarks[0]
            
            # CoG
            if args.draw_cog:
                cog = calculate_cog(landmarks)
                if cog:
                    cog_history.append(cog)
                    if args.cog_trail_length > 0 and len(cog_history) > args.cog_trail_length:
                        cog_history.pop(0)
                        
                    # Calculate Normalized Horizontal Offset
                    # Reference: Hip Center
                    # Width: Hip Width
                    l_hip = landmarks[23]
                    r_hip = landmarks[24]
                    if l_hip.visibility > 0.3 and r_hip.visibility > 0.3:
                        hip_cx = (l_hip.x + r_hip.x) * 0.5
                        hip_w = abs(l_hip.x - r_hip.x)
                        # Avoid div by zero
                        if hip_w < 0.01: 
                            # Try shoulders
                            l_sh = landmarks[11]
                            r_sh = landmarks[12]
                            hip_w = abs(l_sh.x - r_sh.x)
                            
                        if hip_w < 0.01: hip_w = 0.1 # Fallback
                        
                        # Normalize
                        # offset = (cog.x - hip_center) / hip_width
                        # offset = (cog[0] - hip_cx) / hip_w

                        # New Logic: Reference is Geometric Center of Bounding Box (Green Cross)
                        # This ensures Graph 0 aligns with Visual Green Cross.
                        xs_all = [lm.x * w for lm in landmarks if lm.visibility > 0.3]
                        if xs_all:
                            min_px = min(xs_all)
                            max_px = max(xs_all)
                            box_w_px = max_px - min_px
                            pad_px = max(10, box_w_px * 0.1)
                            
                            min_vis = max(0, min_px - pad_px)
                            max_vis = min(w, max_px + pad_px)
                            
                            center_px = (min_vis + max_vis) / 2.0
                            center_norm = center_px / w
                            
                            # Calculate offset from Geometric Center
                            offset = (cog[0] - center_norm) / hip_w
                        else:
                            # Fallback if no points visible (unlikely if hip is visible)
                            offset = (cog[0] - hip_cx) / hip_w
                        
                        cog_offset_history.append((frame_count, offset))
                    else:
                        # If hips not visible, maybe reuse last offset or 0?
                        # Use 0 for safety or last known
                        last_val = cog_offset_history[-1][1] if cog_offset_history else 0.0
                        cog_offset_history.append((frame_count, last_val))
                        
            # Limbs
            if args.draw_limb_trails:
                for idx in limb_history:
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        limb_history[idx].append((lm.x, lm.y))
                        if args.limb_trail_length > 0 and len(limb_history[idx]) > args.limb_trail_length:
                            limb_history[idx].pop(0)

        elif args.draw_cog and (not pose or not pose.pose_landmarks):
             # If pose missing, maybe keep last CoG or skip? 
             pass

        # 首帧直接写
        if not prev_frame_written:
            # Draw Ghosts first (lowest layer)
            if args.draw_ghosts and ghost_poses:
                # Create semi-transparent overlay
                overlay = bg_rgb.copy()
                for gp in ghost_poses:
                    # Draw ghost with low visibility?
                    # We reuse draw_pose_landmarks_rgb but perhaps we need it to look different (white?)
                    # Or just normal skeleton but faded.
                    # Since draw_pose_landmarks doesn't support custom style per call easily without refactor,
                    # we just draw them and blend.
                    # Ideally ghosts should be single color (e.g. white/grey) to not distract.
                    # For now just draw normal.
                    draw_pose_landmarks_rgb(overlay, gp, min_visibility=args.min_visibility)
                
                # Blend overlay
                # But wait, draw_pose_landmarks_rgb returns new image.
                # We need to accumulate them?
                # Let's iterate:
                acc_overlay = bg_rgb.copy()
                acc_overlay = cv2.addWeighted(acc_overlay, 0.3, bg_rgb, 0.7, 0) # Dim background? No.
                
                # Better approach: Draw all ghosts on a clean copy of BG, then blend with current frame?
                # No, just blend ghosts onto current frame BG.
                ghost_layer = bg_rgb.copy()
                for gp in ghost_poses:
                     ghost_layer = draw_pose_landmarks_rgb(ghost_layer, gp, min_visibility=args.min_visibility)
                
                # Blend ghost layer 50% with BG
                frame_rgb = cv2.addWeighted(bg_rgb, 0.5, ghost_layer, 0.5, 0)
            else:
                frame_rgb = bg_rgb.copy()

            # Draw Current Pose
            if pose is not None:
                frame_rgb = draw_pose_landmarks_rgb(frame_rgb, pose, min_visibility=args.min_visibility)
                
                if args.draw_limb_trails:
                    frame_rgb = draw_limb_trails(frame_rgb, limb_history, w, h)
                
                if args.draw_bbox:
                    # Get current CoG if available
                    current_cog = cog_history[-1] if cog_history else None
                    frame_rgb = draw_bounding_box(frame_rgb, pose, w, h, cog=current_cog)
                
                angle_rect = None
                if args.draw_angles:
                    frame_rgb, angle_rect = draw_pose_angles(frame_rgb, pose, w, h)
            else:
                # Pose filtered out, still draw limb trails?
                # Probably should draw trails but no current pose
                if args.draw_limb_trails:
                    frame_rgb = draw_limb_trails(frame_rgb, limb_history, w, h)
                
                angle_rect = None # No angles for missing pose

            write_output_frame(frame_rgb, writer, args, w, h, cog_history, cog_offset_history, angle_rect)
            
            prev_frame_written = True
            prev_pose = pose
            continue

        # 插值帧
        if args.insert_frames > 0:
            mid_poses = _interpolate_pose_results(
                prev_pose,
                pose,
                steps=int(args.insert_frames),
                mode=str(args.interp_mode),
            )
            # Interpolate CoG for smoother trail visualization during interpolation?
            # Or just repeat last known real CoG?
            # Better to linearly interpolate CoG between prev_cog and current_cog if possible.
            # But we only calculated current_cog. 
            
            # Simple approach: Recalculate CoG for each interpolated pose
            for mid_pose in mid_poses:
                # Interpolated frames also need background logic
                if args.draw_ghosts and ghost_poses:
                     ghost_layer = bg_rgb.copy()
                     for gp in ghost_poses:
                        ghost_layer = draw_pose_landmarks_rgb(ghost_layer, gp, min_visibility=args.min_visibility)
                     frame_rgb = cv2.addWeighted(bg_rgb, 0.4, ghost_layer, 0.6, 0)
                else:
                     frame_rgb = bg_rgb.copy()

                frame_rgb = draw_pose_landmarks_rgb(frame_rgb, mid_pose, min_visibility=args.min_visibility)
                
                if args.draw_bbox:
                    # For interpolated frames, we can try to use last known CoG or None?
                    # Ideally we interpolate CoG too, but for now reuse last known for visual consistency
                    current_cog = cog_history[-1] if cog_history else None
                    frame_rgb = draw_bounding_box(frame_rgb, mid_pose, w, h, cog=current_cog)
                    
                angle_rect = None
                if args.draw_angles:
                    frame_rgb, angle_rect = draw_pose_angles(frame_rgb, mid_pose, w, h)
                
                # Update temporary CoG/Limb trails?
                # For simplicity, we just draw the history as is (without the interpolated point appended)
                # or we calculate mid_cog and draw.
                # Just drawing static history is safer for stability.
                
                if args.draw_limb_trails:
                    frame_rgb = draw_limb_trails(frame_rgb, limb_history, w, h)

                write_output_frame(frame_rgb, writer, args, w, h, cog_history, cog_offset_history, angle_rect)

        # 当前帧
        
        # Draw Ghosts
        if args.draw_ghosts and ghost_poses:
            ghost_layer = bg_rgb.copy()
            for gp in ghost_poses:
                    ghost_layer = draw_pose_landmarks_rgb(ghost_layer, gp, min_visibility=args.min_visibility)
            # Blend
            frame_rgb = cv2.addWeighted(bg_rgb, 0.4, ghost_layer, 0.6, 0)
        else:
            frame_rgb = bg_rgb.copy()

        # Draw Current
        if pose is not None:
            frame_rgb = draw_pose_landmarks_rgb(frame_rgb, pose, min_visibility=args.min_visibility)
            
            if args.draw_limb_trails:
                frame_rgb = draw_limb_trails(frame_rgb, limb_history, w, h)
                
            if args.draw_bbox:
                current_cog = cog_history[-1] if cog_history else None
                frame_rgb = draw_bounding_box(frame_rgb, pose, w, h, cog=current_cog)
                
            angle_rect = None
            if args.draw_angles:
                frame_rgb, angle_rect = draw_pose_angles(frame_rgb, pose, w, h)
        else:
            # Pose filtered out
            if args.draw_limb_trails:
                 frame_rgb = draw_limb_trails(frame_rgb, limb_history, w, h)
            angle_rect = None

        write_output_frame(frame_rgb, writer, args, w, h, cog_history, cog_offset_history, angle_rect)

        prev_pose = pose

    if cap:
        cap.release()
    writer.release()
    print(f"Done. Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
