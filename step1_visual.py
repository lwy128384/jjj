#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤1: 视觉维度分析
  1. 检测教师是否在讲台区域（背景减除法）
  2. PPT 区域文字内容识别（EasyOCR）
  3. 幻灯片翻页检测（帧间 SSIM）

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
    PPT_REGION                 = _cfg.PPT_REGION
    PPT_AUTO_REGION            = getattr(_cfg, "PPT_AUTO_REGION", True)
    PPT_REGION_FULLSCREEN      = getattr(_cfg, "PPT_REGION_FULLSCREEN", (0.00, 0.00, 1.00, 0.90))
    PPT_REGION_SPLIT           = getattr(_cfg, "PPT_REGION_SPLIT", PPT_REGION)
    FULLSCREEN_BRIGHT_RATIO    = getattr(_cfg, "FULLSCREEN_BRIGHT_RATIO", 0.35)
    FULLSCREEN_LOW_SAT_RATIO   = getattr(_cfg, "FULLSCREEN_LOW_SAT_RATIO", 0.45)
    FULLSCREEN_EDGE_RATIO      = getattr(_cfg, "FULLSCREEN_EDGE_RATIO", 0.02)
    SLIDE_CHANGE_THRESHOLD     = _cfg.SLIDE_CHANGE_THRESHOLD
    TEACHER_PRESENCE_THRESHOLD = _cfg.TEACHER_PRESENCE_THRESHOLD
    BG_INIT_FRAMES             = _cfg.BG_INIT_FRAMES
    OCR_CONFIDENCE_THRESHOLD   = _cfg.OCR_CONFIDENCE_THRESHOLD
except ImportError:
    OUTPUT_DIR                 = r"D:\video\output"
    VISUAL_SAMPLE_FPS          = 1
    PODIUM_REGION              = (0.38, 0.42, 0.66, 0.94)
    PPT_REGION                 = (0.02, 0.02, 0.98, 0.80)
    PPT_AUTO_REGION            = True
    PPT_REGION_FULLSCREEN      = (0.00, 0.00, 1.00, 0.90)
    PPT_REGION_SPLIT           = PPT_REGION
    FULLSCREEN_BRIGHT_RATIO    = 0.35
    FULLSCREEN_LOW_SAT_RATIO   = 0.45
    FULLSCREEN_EDGE_RATIO      = 0.02
    SLIDE_CHANGE_THRESHOLD     = 0.70
    TEACHER_PRESENCE_THRESHOLD = 0.05
    BG_INIT_FRAMES             = 30
    OCR_CONFIDENCE_THRESHOLD   = 0.40


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


def compute_ssim_fast(img1, img2):
    """轻量 SSIM（缩小到 160×90 后计算）"""
    g1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (160, 90)).astype(float)
    g2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (160, 90)).astype(float)
    mu1, mu2   = g1.mean(), g2.mean()
    s1,  s2    = g1.std(),  g2.std()
    s12 = ((g1 - mu1) * (g2 - mu2)).mean()
    C1, C2     = (0.01*255)**2, (0.03*255)**2
    num  = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den  = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
    return float(num / (den + 1e-8))


def init_ocr():
    """初始化 EasyOCR（首次运行自动下载 ~50 MB 模型）"""
    try:
        import easyocr
        print("  初始化 OCR 引擎（首次运行会下载模型，约 50 MB）…")
        # 'ch_sim' = Simplified Chinese, 'en' = English
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        return reader
    except ImportError:
        print("  警告: EasyOCR 未安装，将跳过 OCR。请运行: pip install easyocr")
        return None
    except Exception as e:
        print(f"  警告: OCR 初始化失败: {e}")
        return None


def run_ocr(reader, frame, region, min_conf):
    """对帧的指定区域做 OCR，返回识别文本"""
    if reader is None:
        return ""
    crop = region_crop(frame, region)
    if crop.size == 0:
        return ""
    try:
        results = reader.readtext(crop, detail=1, paragraph=True)
        return " ".join(t for _, t, c in results if c >= min_conf).strip()
    except Exception:
        return ""


def is_fullscreen_ppt(frame):
    """
    粗略判断全屏 PPT 画面：
    - 亮部占比高
    - 低饱和像素占比高（大面积白底/浅色底）
    - 边缘密度达到一定水平（存在文字线条）
    """
    if frame is None or frame.size == 0:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)

    bright_ratio = float(np.mean(gray > 180))
    low_sat_ratio = float(np.mean(hsv[:, :, 1] < 40))
    edge_ratio = float(np.mean(edges > 0))

    return bool(
        bright_ratio > FULLSCREEN_BRIGHT_RATIO and
        low_sat_ratio > FULLSCREEN_LOW_SAT_RATIO and
        edge_ratio > FULLSCREEN_EDGE_RATIO
    )


def get_active_ppt_region(full_ppt):
    """根据是否全屏 PPT 选择 OCR/SSIM 使用的区域。"""
    if not PPT_AUTO_REGION:
        return PPT_REGION
    return PPT_REGION_FULLSCREEN if full_ppt else PPT_REGION_SPLIT


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

    # OCR
    ocr_reader = init_ocr()

    # 结果容器
    teacher_timeline  = []
    slide_transitions = []
    ppt_content       = []

    prev_frame       = None
    prev_ppt_region  = PPT_REGION
    warned_bad_crop  = False
    slide_idx        = 0
    current_ppt_text = ""
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
            ppt_region  = get_active_ppt_region(full_ppt)
            # 全屏课件时，背景减除对教师检测容易失真，默认按“在讲台”处理，
            # 后续可在步骤5结合静默与多模态规则进一步过滤疑似干扰片段。
            in_podium   = bool(
                fg_ratio > TEACHER_PRESENCE_THRESHOLD or full_ppt
            )

            teacher_timeline.append({
                "time":        ts,
                "in_podium":   in_podium,
                "motion_ratio": round(fg_ratio, 4),
                "full_screen_ppt": full_ppt,
            })

            # —— 幻灯片切换检测 ——
            if prev_frame is not None:
                prev_crop = region_crop(prev_frame, prev_ppt_region)
                curr_crop = region_crop(frame, ppt_region)
                if prev_crop.size > 0 and curr_crop.size > 0:
                    ssim = compute_ssim_fast(prev_crop, curr_crop)
                else:
                    if not warned_bad_crop:
                        print(
                            "  警告: PPT 区域裁剪为空，回退到整帧 SSIM；"
                            f"prev_region={prev_ppt_region}, curr_region={ppt_region}"
                        )
                        warned_bad_crop = True
                    ssim = compute_ssim_fast(prev_frame, frame)
                if ssim < SLIDE_CHANGE_THRESHOLD:
                    slide_idx += 1
                    slide_transitions.append({
                        "time":      ts,
                        "ssim":      round(ssim, 4),
                        "slide_idx": slide_idx,
                    })
                    # 切换时 OCR
                    new_text = run_ocr(ocr_reader, frame, ppt_region,
                                       OCR_CONFIDENCE_THRESHOLD)
                    if new_text:
                        current_ppt_text = new_text
                    ppt_content.append({
                        "time":      ts,
                        "slide_idx": slide_idx,
                        "text":      current_ppt_text,
                    })
            else:
                # 第一帧
                current_ppt_text = run_ocr(ocr_reader, frame, ppt_region,
                                           OCR_CONFIDENCE_THRESHOLD)
                ppt_content.append({"time": 0.0, "slide_idx": 0,
                                    "text": current_ppt_text})

            prev_frame = frame.copy()
            prev_ppt_region = ppt_region
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
    print(f"    幻灯片切换: {result['stats']['total_slide_changes']} 次")
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
