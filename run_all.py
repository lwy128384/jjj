#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
总流程入口 — 一键运行所有步骤

用法（处理所有 lesson/ 视频）:
  python run_all.py

用法（只处理指定视频）:
  python run_all.py --video D:/video/lesson/example.mp4

用法（从指定步骤开始）:
  python run_all.py --start 2

用法（只运行某一步）:
  python run_all.py --step 3 --video D:/video/lesson/example.mp4
"""

import os
import sys
import glob
import argparse
import traceback
import datetime

# ============================================================
# 读取基础路径
# ============================================================
try:
    import config as _cfg
    LESSON_DIR = _cfg.LESSON_DIR
    OUTPUT_DIR = _cfg.OUTPUT_DIR
except ImportError:
    LESSON_DIR = r"D:\video\lesson"
    OUTPUT_DIR = r"D:\video\output"

# 导入各步骤
from step1_visual import analyze_video_visual, get_output_dir as _get1
from step2_audio  import analyze_video_audio,  get_output_dir as _get2
from step3_text   import analyze_text,          get_output_dir as _get3
from step4_align  import align_features,        get_output_dir as _get4
from step5_fusion import fuse_and_cut,          get_output_dir as _get5


STEP_FUNCS = {
    1: ("视觉分析",   analyze_video_visual, _get1),
    2: ("语音分析",   analyze_video_audio,  _get2),
    3: ("文本分析",   analyze_text,         _get3),
    4: ("多模态对齐", align_features,       _get4),
    5: ("融合剪辑",   fuse_and_cut,         _get5),
}


def run_pipeline(video_path, start_step=1, end_step=5):
    print(f"\n{'='*60}")
    print(f"  视频: {os.path.basename(video_path)}")
    print(f"  时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    results = {}
    for step in range(start_step, end_step + 1):
        label, func, get_dir = STEP_FUNCS[step]
        out_dir, vname = get_dir(video_path, OUTPUT_DIR)
        print(f"\n>>> 步骤{step}: {label}")
        try:
            result = func(video_path, out_dir, vname)
            results[step] = {"status": "ok", "result": result}
            print(f"    步骤{step} 完成 ✓")
        except Exception as e:
            msg = traceback.format_exc()
            print(f"\n  ✗ 步骤{step} 出错:\n{msg}")
            results[step] = {"status": "error", "error": str(e)}
            print(f"  跳过后续步骤")
            break

    return results


def find_videos(lesson_dir):
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.wmv", "*.MP4")
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(lesson_dir, ext)))
    return sorted(set(videos))


def main():
    parser = argparse.ArgumentParser(description="一键运行全流程")
    parser.add_argument("--video",  default=None,
                        help="指定单个视频路径（不填则处理 lesson/ 目录下所有视频）")
    parser.add_argument("--start",  type=int, default=1,
                        help="从第几步开始（1-5，默认 1）")
    parser.add_argument("--step",   type=int, default=None,
                        help="只运行某一步（1-5）")
    args = parser.parse_args()

    if args.step is not None:
        start_step = end_step = args.step
    else:
        start_step = args.start
        end_step   = 5

    # 确定视频列表
    if args.video:
        if not os.path.exists(args.video):
            print(f"错误: 视频不存在: {args.video}")
            sys.exit(1)
        videos = [args.video]
    else:
        if not os.path.isdir(LESSON_DIR):
            print(f"错误: lesson 目录不存在: {LESSON_DIR}")
            print("请先创建目录并放入视频文件。")
            sys.exit(1)
        videos = find_videos(LESSON_DIR)
        if not videos:
            print(f"lesson 目录中未找到视频文件: {LESSON_DIR}")
            print("支持格式: mp4 / avi / mov / mkv / flv / wmv")
            sys.exit(1)
        print(f"找到 {len(videos)} 个视频:\n" +
              "\n".join(f"  {os.path.basename(v)}" for v in videos))

    # 逐个处理
    all_results = {}
    for vpath in videos:
        res = run_pipeline(vpath, start_step, end_step)
        all_results[vpath] = res

    # 汇总报告
    print(f"\n{'='*60}")
    print("  处理汇总")
    print(f"{'='*60}")
    for vpath, res in all_results.items():
        name = os.path.basename(vpath)
        statuses = [f"步骤{s}: {'✓' if v['status']=='ok' else '✗'}"
                    for s, v in res.items()]
        print(f"  {name}  →  {', '.join(statuses)}")
    print()


if __name__ == "__main__":
    main()
