"""按视频起始时间裁剪生成新视频。

特点
- 时间输入支持：秒数（如 12.5）或 hh:mm:ss(.ms) / mm:ss(.ms)
- 默认优先使用 ffmpeg（如已安装）以保留音频；否则使用 OpenCV（仅视频无音频）

示例
- 裁剪 00:10.0 到 00:35.5：
  python trim_video_by_time.py -i input.mp4 -s 00:10 -e 00:35.5 -o out.mp4

- 从 12.3 秒开始，持续 8 秒：
  python trim_video_by_time.py -i input.mp4 -s 12.3 -d 8 -o out.mp4

注意
- OpenCV 方式通常无法保留音频；如果你需要音频，请安装 ffmpeg（推荐）。
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


@dataclass(frozen=True)
class TimeRange:
    start_s: float
    end_s: Optional[float]


def parse_time_to_seconds(value: str) -> float:
    """解析时间字符串为秒。

    支持格式：
    - "12.34"（秒）
    - "MM:SS" / "MM:SS.mmm"
    - "HH:MM:SS" / "HH:MM:SS.mmm"
    """
    s = str(value).strip()
    if not s:
        raise ValueError("时间不能为空")

    # 纯数字：按秒
    try:
        if ":" not in s:
            sec = float(s)
            if sec < 0:
                raise ValueError
            return sec
    except ValueError as e:
        raise ValueError(f"无法解析时间（秒数）：{value!r}") from e

    parts = s.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"无法解析时间（需要 MM:SS 或 HH:MM:SS）：{value!r}")

    try:
        if len(parts) == 2:
            mm = int(parts[0])
            ss = float(parts[1])
            hh = 0
        else:
            hh = int(parts[0])
            mm = int(parts[1])
            ss = float(parts[2])
    except ValueError as e:
        raise ValueError(f"无法解析时间（包含非法数字）：{value!r}") from e

    if hh < 0 or mm < 0 or ss < 0:
        raise ValueError(f"时间不能为负：{value!r}")
    if ss >= 60 or mm >= 60:
        # 允许超出也能算，但这里直接提示更友好
        raise ValueError(f"分钟/秒请小于 60：{value!r}")

    return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)


def build_time_range(*, start: str, end: str, duration: str) -> TimeRange:
    start_s = parse_time_to_seconds(start)

    end_s: Optional[float] = None
    if end:
        end_s = parse_time_to_seconds(end)
    if duration:
        dur_s = parse_time_to_seconds(duration)
        end_from_dur = start_s + dur_s
        if end_s is None:
            end_s = end_from_dur
        else:
            # 两者都给了：取更早的结束，避免越界
            end_s = min(end_s, end_from_dur)

    if end_s is not None and end_s <= start_s:
        raise ValueError(f"end 必须大于 start：start={start_s:.3f}s end={end_s:.3f}s")

    return TimeRange(start_s=start_s, end_s=end_s)


def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None


def _format_hhmmss(seconds: float) -> str:
    # ffmpeg 的 -ss/-to 支持秒数字符串，这里用更直观的 hh:mm:ss.mmm
    seconds = max(0.0, float(seconds))
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60.0
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def trim_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    time_range: TimeRange,
    *,
    reencode: bool,
) -> None:
    """用 ffmpeg 裁剪，默认可保留音频。"""

    if time_range.end_s is None:
        raise ValueError("ffmpeg 模式需要明确 end 或 duration")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 说明：
    # -ss 放在 -i 前面更快但不够精确；放在 -i 后面更精确。
    # 这里取折中：精确优先。
    start = _format_hhmmss(time_range.start_s)
    end = _format_hhmmss(time_range.end_s)

    if reencode:
        # 兼容性更好，但慢
        args = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            start,
            "-to",
            end,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    else:
        # 快速裁剪：尽量复制码流（通常更快，但边界可能不是帧级精确）
        args = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            start,
            "-to",
            end,
            "-c",
            "copy",
            str(output_path),
        ]

    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg 裁剪失败（exit={proc.returncode}）。\n{msg}")


def create_video_writer(output_path: Path, fps: float, width: int, height: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fourcc_str in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (int(width), int(height)))
        if writer.isOpened():
            return writer

    raise RuntimeError(
        f"无法创建 VideoWriter 输出：{output_path}\n"
        "请检查 OpenCV 是否具备 MP4 编码能力（或安装带 FFmpeg 的 OpenCV）。"
    )


def trim_with_opencv(input_path: Path, output_path: Path, time_range: TimeRange) -> None:
    """用 OpenCV 裁剪（不保留音频）。"""

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("读取到的视频宽高无效。")

    start_frame = int(math.floor(time_range.start_s * fps + 1e-9))
    if start_frame < 0:
        start_frame = 0

    if time_range.end_s is None:
        end_frame = total_frames if total_frames > 0 else None
    else:
        end_frame = int(math.floor(time_range.end_s * fps + 1e-9))
        if end_frame < 0:
            end_frame = 0

    if total_frames > 0:
        start_frame = min(start_frame, max(0, total_frames - 1))
        if end_frame is not None:
            end_frame = min(end_frame, total_frames)

    if end_frame is not None and end_frame <= start_frame:
        cap.release()
        raise ValueError(
            f"裁剪范围为空：start_frame={start_frame} end_frame={end_frame} fps={fps:.3f}"
        )

    # 尝试 seek 到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    writer = create_video_writer(output_path, fps, width, height)

    written = 0
    frame_idx = start_frame

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break

        ok, frame_bgr = cap.read()
        if not ok:
            break

        writer.write(frame_bgr)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()

    if written <= 0:
        raise RuntimeError("没有写出任何帧（可能 start 超过视频长度）。")


def default_output_path(input_path: Path, time_range: TimeRange) -> Path:
    start_tag = f"{time_range.start_s:.3f}s".replace(".", "p")
    if time_range.end_s is None:
        end_tag = "end"
    else:
        end_tag = f"{time_range.end_s:.3f}s".replace(".", "p")
    return input_path.with_name(f"{input_path.stem}_trim_{start_tag}_{end_tag}{input_path.suffix}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="按起始时间裁剪视频并输出新视频")
    p.add_argument("--input", "-i", required=True, help="输入视频路径")
    p.add_argument("--output", "-o", default="", help="输出视频路径（默认自动生成）")

    p.add_argument("--start", "-s", required=True, help="开始时间（秒或 HH:MM:SS.mmm / MM:SS.mmm）")

    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--end", "-e", default="", help="结束时间（同 start 格式）")
    grp.add_argument("--duration", "-d", default="", help="持续时长（同 start 格式）")

    p.add_argument(
        "--method",
        choices=["auto", "ffmpeg", "opencv"],
        default="auto",
        help="裁剪方式：auto（默认）、ffmpeg、opencv",
    )
    p.add_argument(
        "--reencode",
        action="store_true",
        help="ffmpeg 模式下强制重编码（更兼容/更精确，但更慢；默认尝试 copy）",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入视频：{input_path}")

    # 注意：这里 end/duration 二选一，但我们 build_time_range 也支持两者同时为空（表示到结尾）
    end_str = getattr(args, "end", "") or ""
    dur_str = getattr(args, "duration", "") or ""
    time_range = build_time_range(start=str(args.start), end=str(end_str), duration=str(dur_str))

    output_path = Path(args.output) if str(args.output).strip() else default_output_path(input_path, time_range)

    method = str(args.method)
    if method == "auto":
        if _ffmpeg_exists() and time_range.end_s is not None:
            method = "ffmpeg"
        else:
            method = "opencv"

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Start : {time_range.start_s:.3f}s")
    if time_range.end_s is None:
        print("End   : (to end)")
    else:
        print(f"End   : {time_range.end_s:.3f}s")
    print(f"Method: {method}")

    if method == "ffmpeg":
        if not _ffmpeg_exists():
            raise RuntimeError("选择了 ffmpeg 方式，但系统找不到 ffmpeg。请先安装并加入 PATH。")
        trim_with_ffmpeg(input_path, output_path, time_range, reencode=bool(args.reencode))
    elif method == "opencv":
        trim_with_opencv(input_path, output_path, time_range)
        print("NOTE: OpenCV 方式不保留音频。")
    else:
        raise ValueError(f"未知 method：{method}")

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
