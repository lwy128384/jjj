#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤1: 视觉维度分析
  1. 检测教师是否在讲台区域（背景减除法）
  2. 若检测到全屏 PPT，默认教师在讲台区域内

运行方式:
  python step1_visual.py --video D:/video/lesson/example.mp4

输出:
  D:/video/output/example/visual_features.json
"""

import cv2
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# ============================================================
# 默认参数（优先从 config.py 读取）
# ============================================================
try:
    import config as _cfg
    OUTPUT_DIR                 = _cfg.OUTPUT_DIR
    VISUAL_SAMPLE_FPS          = _cfg.VISUAL_SAMPLE_FPS
    PODIUM_REGION              = _cfg.PODIUM_REGION
    TEACHER_PRESENCE_THRESHOLD = _cfg.TEACHER_PRESENCE_THRESHOLD
    BG_INIT_FRAMES             = _cfg.BG_INIT_FRAMES
except ImportError:
    OUTPUT_DIR                 = r"D:\video\output"
    VISUAL_SAMPLE_FPS          = 1
    PODIUM_REGION              = (0.10, 0.25, 0.90, 0.95)
    TEACHER_PRESENCE_THRESHOLD = 0.05
    BG_INIT_FRAMES             = 30


# ============================================================
# 工具函数
# ============================================================

def get_output_dir(video_path, base_output_dir=None):
    base = base_output_dir or OUTPUT_DIR
    name = Path(video_path).stem
    out  = os.path.join(base, name)
    os.makedirs(out, exist_ok=True)
    return out, name


def region_crop(frame, region):
    """按比例裁剪帧 region=(x1,y1,x2,y2) 均为 0~1"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region
    return frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]


def is_fullscreen_ppt(frame):
    """
    粗略判断当前帧是否为全屏 PPT。
    规则：边缘密度较高 + 亮部占比较高 + 低饱和像素占比较高。
    """
    if frame is None or frame.size == 0:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.mean(edges > 0))
    bright_ratio = float(np.mean(gray > 180))
    low_sat_ratio = float(np.mean(hsv[:, :, 1] < 40))
    return edge_ratio > 0.02 and bright_ratio > 0.35 and low_sat_ratio > 0.45


# ============================================================
# 核心分析
# ============================================================

def analyze_video_visual(video_path, output_dir, video_name):
    print(f"\n[步骤1] 视觉分析: {video_name}")
    print(f"  视频路径: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {w}×{h}  {fps:.1f}fps  {duration:.1f}s")

    sample_interval = max(1, int(fps / VISUAL_SAMPLE_FPS))
    total_samples   = int(duration * VISUAL_SAMPLE_FPS)

    # 背景减除器
    bg = cv2.createBackgroundSubtractorMOG2(
        history        = max(50, BG_INIT_FRAMES * sample_interval),
        varThreshold   = 50,
        detectShadows  = False,
    )

    # 结果容器
    teacher_timeline  = []
    slide_transitions = []  # 保留字段兼容下游
    ppt_content       = []  # 保留字段兼容下游
    kernel           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 预热背景模型
    print(f"  预热背景模型（{BG_INIT_FRAMES} 帧）…")
    for k in range(min(BG_INIT_FRAMES, total_samples)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k * sample_interval)
        ret, frm = cap.read()
        if ret:
            bg.apply(frm, learningRate=0.1)

    # 逐帧分析
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with tqdm(total=total_samples, desc="  分析进度", ncols=80, unit="帧") as pbar:
        for idx in range(total_samples):
            ts = round(idx / VISUAL_SAMPLE_FPS, 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx * sample_interval)
            ret, frame = cap.read()
            if not ret:
                break

            # —— 教师存在检测 ——
            podium_crop = region_crop(frame, PODIUM_REGION)
            fg_mask     = bg.apply(podium_crop, learningRate=0.002)
            fg_mask     = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,   kernel)
            fg_mask     = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)
            fg_ratio    = float(np.sum(fg_mask > 127)) / (fg_mask.size + 1e-8)
            full_ppt    = is_fullscreen_ppt(frame)
            in_podium   = bool(fg_ratio > TEACHER_PRESENCE_THRESHOLD or full_ppt)

            teacher_timeline.append({
                "time":        ts,
                "in_podium":   in_podium,
                "motion_ratio": round(fg_ratio, 4),
                "full_screen_ppt": full_ppt,
            })
            pbar.update(1)

    cap.release()

    present_cnt = sum(1 for t in teacher_timeline if t["in_podium"])
    result = {
        "video_name":        video_name,
        "video_path":        str(video_path),
        "fps":               round(fps, 2),
        "duration":          round(duration, 2),
        "total_frames":      total_frames,
        "resolution":        f"{w}x{h}",
        "teacher_timeline":  teacher_timeline,
        "slide_transitions": slide_transitions,
        "ppt_content":       ppt_content,
        "stats": {
            "total_samples":          len(teacher_timeline),
            "teacher_present_count":  present_cnt,
            "teacher_presence_ratio": round(present_cnt / max(len(teacher_timeline), 1), 3),
            "total_slide_changes":    len(slide_transitions),
        },
    }

    out_file = os.path.join(output_dir, "visual_features.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ 视觉分析完成")
    print(f"    教师在讲台: {result['stats']['teacher_presence_ratio']*100:.1f}%")
    print("    说明: 检测到全屏PPT时默认教师在讲台区域内")
    print(f"    输出: {out_file}")
    return result


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="步骤1: 视觉维度分析")
    parser.add_argument("--video",  required=True, help="视频文件路径")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出根目录")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频不存在: {args.video}")
        sys.exit(1)

    out_dir, vname = get_output_dir(args.video, args.output)
    analyze_video_visual(args.video, out_dir, vname)


if __name__ == "__main__":
    main()
