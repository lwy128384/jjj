#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标注初始化工具：
从 step5 产物 final_index.json 生成可人工二次编辑的 annotation.json。

用法（推荐）:
  python init_annotation.py --video D:/video/lesson/example.mp4

用法（直接指定 final_index）:
  python init_annotation.py --final_index D:/video/output/example/final_index.json \
                            --output D:/video/annotations/example_annotation.json
"""

import os
import json
import math
import argparse
from pathlib import Path

try:
    import config as _cfg
    OUTPUT_DIR = _cfg.OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = r"D:\video\output"


def _safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_time_seconds(v):
    if isinstance(v, str):
        text = v.strip()
        parts = text.split(":")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            h, m, s = [int(p) for p in parts]
            return float(h * 3600 + m * 60 + s)
        return _safe_float(text, None)
    return _safe_float(v, None)


def _sec_to_hms(seconds):
    sec = max(int(seconds), 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02d}:{s:02d}"


def _normalize_interval(start, end):
    s = _parse_time_seconds(start)
    e = _parse_time_seconds(end)
    if s is None or e is None:
        return None
    if e < s:
        s, e = e, s
    s = math.floor(s)
    e = math.ceil(e)
    if e <= s:
        return None
    return s, e


def _from_final_index(final_index):
    items = []

    for clip in final_index.get("clips", []) or []:
        interval = _normalize_interval(clip.get("start"), clip.get("end"))
        if not interval:
            continue
        s, e = interval
        title = str(clip.get("title", "") or "").strip() or "未命名知识点"
        is_interference = (title == "干扰片段")
        items.append((s, e, title, is_interference))

    for seg in final_index.get("removed_segments", []) or []:
        interval = _normalize_interval(seg.get("start"), seg.get("end"))
        if not interval:
            continue
        s, e = interval
        title = str(seg.get("title", "") or "").strip() or "干扰片段"
        items.append((s, e, title, True))

    items.sort(key=lambda x: (x[0], x[1], x[3]))

    annotations = []
    seen = set()
    for s, e, title, is_interference in items:
        key = (s, e, title, is_interference)
        if key in seen:
            continue
        seen.add(key)
        annotations.append({
            "start": _sec_to_hms(s),
            "end": _sec_to_hms(e),
            "title": title,
            "is_interference": bool(is_interference),
        })
    return annotations


def main():
    parser = argparse.ArgumentParser(description="从 final_index 初始化标注文件")
    parser.add_argument("--video", default=None, help="视频路径（用于自动定位 final_index）")
    parser.add_argument("--final_index", default=None, help="final_index.json 路径")
    parser.add_argument("--output", default=None, help="输出 annotation.json 路径")
    parser.add_argument("--annotations_dir", default=r"D:\video\annotations", help="输出目录（未指定 --output 时生效）")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的输出文件")
    args = parser.parse_args()

    if not args.video and not args.final_index:
        raise SystemExit("错误: --video 和 --final_index 至少提供一个")

    video_name = ""
    if args.video:
        video_name = Path(args.video).name
        video_stem = Path(args.video).stem
        final_index_path = args.final_index or os.path.join(OUTPUT_DIR, video_stem, "final_index.json")
    else:
        final_index_path = args.final_index
        video_stem = Path(final_index_path).parent.name

    if not os.path.exists(final_index_path):
        raise SystemExit(f"错误: final_index 不存在: {final_index_path}")

    with open(final_index_path, "r", encoding="utf-8") as f:
        final_index = json.load(f)

    if not video_name:
        video_name = str(final_index.get("video", "") or "").strip() or f"{video_stem}.mp4"

    annotations = _from_final_index(final_index)
    output_data = {
        "video": video_name,
        "annotations": annotations
    }

    if args.output:
        output_path = args.output
    else:
        os.makedirs(args.annotations_dir, exist_ok=True)
        output_path = os.path.join(args.annotations_dir, f"{video_stem}_annotation.json")

    if os.path.exists(output_path) and not args.force:
        raise SystemExit(f"错误: 输出文件已存在，请加 --force 覆盖: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"已生成标注草稿: {output_path}")
    print("请人工复核并微调时间边界后再用于训练。")


if __name__ == "__main__":
    main()
