#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤4: 多模态时间对齐
  1. 融合视觉、音频、文本特征到统一时间轴
  2. 生成多模态时间戳索引文件

运行方式:
  python step4_align.py --video D:/video/lesson/example.mp4

输出:
  D:/video/output/example/multimodal_index.json
"""

import os
import sys
import json
import argparse
from bisect import bisect_right
from pathlib import Path

# ============================================================
# 默认参数
# ============================================================
try:
    import config as _cfg
    OUTPUT_DIR                  = _cfg.OUTPUT_DIR
    TIME_RESOLUTION             = _cfg.TIME_RESOLUTION
    NO_SPEECH_PROB_THRESHOLD    = _cfg.NO_SPEECH_PROB_THRESHOLD
    NO_SPEECH_IGNORE_WITH_TEXT  = _cfg.NO_SPEECH_IGNORE_WITH_TEXT
    NO_SPEECH_TEXT_SHORT_LEN    = _cfg.NO_SPEECH_TEXT_SHORT_LEN
    SPEECH_CONFIDENCE_THRESHOLD = _cfg.SPEECH_CONFIDENCE_THRESHOLD
except ImportError:
    OUTPUT_DIR                  = r"D:\video\output"
    TIME_RESOLUTION             = 1.0
    NO_SPEECH_PROB_THRESHOLD    = 0.80
    NO_SPEECH_IGNORE_WITH_TEXT  = True
    NO_SPEECH_TEXT_SHORT_LEN    = 3
    SPEECH_CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# 工具函数
# ============================================================

def get_output_dir(video_path, base_output_dir=None):
    base = base_output_dir or OUTPUT_DIR
    name = Path(video_path).stem
    out  = os.path.join(base, name)
    os.makedirs(out, exist_ok=True)
    return out, name


def _load(path, label):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  已加载 {label}: {path}")
    return data


def load_features(output_dir):
    visual = _load(os.path.join(output_dir, "visual_features.json"), "视觉特征")
    audio  = _load(os.path.join(output_dir, "audio_features.json"),  "音频特征")
    text   = _load(os.path.join(output_dir, "text_features.json"),   "文本特征")
    if audio is None:
        raise FileNotFoundError("音频特征不存在，请先运行步骤2")
    if text is None:
        raise FileNotFoundError("文本特征不存在，请先运行步骤3")
    return visual, audio, text


# ============================================================
# 查找辅助
# ============================================================

def find_latest(items, t, key="time"):
    """在已按 key 排序的列表中找 ≤ t 的最新项"""
    if not items:
        return None
    vals = [item[key] for item in items]
    idx  = bisect_right(vals, t) - 1
    return items[max(0, idx)]


def find_enclosing(items, t, s_key="start", e_key="end"):
    """找包含时刻 t 的片段；没有则返回最近一个"""
    for item in items:
        if item[s_key] <= t <= item[e_key]:
            return item
    if items:
        return min(items, key=lambda x: abs(x[s_key] - t))
    return None


def _is_silence_segment(seg):
    if not seg:
        return True
    no_sp = float(seg.get("no_speech_prob", 1.0) or 1.0)
    if no_sp <= NO_SPEECH_PROB_THRESHOLD:
        return False
    text = str(seg.get("text", "")).strip()
    if not (NO_SPEECH_IGNORE_WITH_TEXT and text):
        return True
    return len(text) <= NO_SPEECH_TEXT_SHORT_LEN


# ============================================================
# 构建时间序列
# ============================================================

def build_timeline(visual, audio, text, duration):
    n_pts = int(duration / TIME_RESOLUTION) + 1

    teacher_tl    = visual.get("teacher_timeline",  []) if visual else []
    slide_trans   = visual.get("slide_transitions", []) if visual else []
    ppt_content   = visual.get("ppt_content",       []) if visual else []
    audio_segs    = audio.get("segments",           [])
    know_segs     = text.get("knowledge_segments",  [])
    know_bounds   = set(text.get("boundaries",      []))

    slide_times   = {st["time"] for st in slide_trans}

    series = []
    for i in range(n_pts):
        t = round(i * TIME_RESOLUTION, 2)

        # —— 视觉 ——
        t_item     = find_latest(teacher_tl, t)
        in_podium  = t_item["in_podium"]    if t_item else True
        motion     = t_item.get("motion_ratio", 0.0) if t_item else 0.0

        ppt_item   = find_latest(ppt_content, t)
        ppt_text   = ppt_item.get("text",      "") if ppt_item else ""
        slide_idx  = ppt_item.get("slide_idx", 0)  if ppt_item else 0
        is_slide   = any(abs(t - st) < TIME_RESOLUTION for st in slide_times)

        # —— 音频 ——
        a_seg      = find_enclosing(audio_segs, t)
        sp_text    = a_seg.get("text",           "")   if a_seg else ""
        speaker    = a_seg.get("speaker",        "")   if a_seg else ""
        sp_conf    = a_seg.get("confidence",     0.0)  if a_seg else 0.0
        no_sp      = a_seg.get("no_speech_prob", 1.0)  if a_seg else 1.0
        is_silence = _is_silence_segment(a_seg)

        # —— 知识点 ——
        k_seg      = find_enclosing(know_segs, t)
        k_id       = k_seg.get("id",    0)      if k_seg else 0
        k_title    = k_seg.get("title", "未知") if k_seg else "未知"
        is_bound   = any(abs(t - b) < TIME_RESOLUTION for b in know_bounds)

        series.append({
            "time":                 t,
            # 视觉
            "teacher_present":      in_podium,
            "motion_ratio":         round(motion, 4),
            "slide_idx":            slide_idx,
            "ppt_text":             ppt_text[:60],
            "is_slide_transition":  is_slide,
            # 音频
            "speech_text":          sp_text[:60],
            "speaker":              speaker,
            "speech_confidence":    round(sp_conf, 4),
            "is_silence":           is_silence,
            # 知识点
            "knowledge_id":         k_id,
            "knowledge_title":      k_title,
            "is_knowledge_boundary": is_bound,
        })

    return series


# ============================================================
# 核心对齐
# ============================================================

def align_features(video_path, output_dir, video_name):
    print(f"\n[步骤4] 多模态对齐: {video_name}")

    visual, audio, text = load_features(output_dir)

    # 推断总时长
    duration = audio.get("total_duration", 0.0)
    if duration <= 0 and audio.get("segments"):
        duration = max(s["end"] for s in audio["segments"])
    if visual and visual.get("duration", 0) > duration:
        duration = visual["duration"]

    print(f"  时长: {duration:.1f}s  分辨率: {TIME_RESOLUTION}s")
    series = build_timeline(visual, audio, text, duration)
    print(f"  时间序列: {len(series)} 个点")

    n = len(series)
    present = sum(1 for p in series if p["teacher_present"])
    silence = sum(1 for p in series if p["is_silence"])

    result = {
        "video_name":         video_name,
        "video_path":         str(video_path),
        "duration":           round(duration, 2),
        "time_resolution":    TIME_RESOLUTION,
        "total_points":       n,
        "time_series":        series,
        "knowledge_segments": text.get("knowledge_segments", []),
        "slide_transitions":  visual.get("slide_transitions", []) if visual else [],
        "stats": {
            "teacher_presence_ratio": round(present / max(n, 1), 3),
            "silence_ratio":          round(silence / max(n, 1), 3),
            "total_knowledge_points": len(text.get("knowledge_segments", [])),
            "total_slide_transitions": len(
                visual.get("slide_transitions", []) if visual else []),
        },
    }

    out_file = os.path.join(output_dir, "multimodal_index.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ 多模态对齐完成")
    print(f"    教师在讲台: {result['stats']['teacher_presence_ratio']*100:.1f}%")
    print(f"    静默比例:   {result['stats']['silence_ratio']*100:.1f}%")
    print(f"    知识点数:   {result['stats']['total_knowledge_points']}")
    print(f"    输出: {out_file}")
    return result


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="步骤4: 多模态时间对齐")
    parser.add_argument("--video",  required=True, help="视频文件路径")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出根目录")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频不存在: {args.video}")
        sys.exit(1)

    out_dir, vname = get_output_dir(args.video, args.output)
    align_features(args.video, out_dir, vname)


if __name__ == "__main__":
    main()
