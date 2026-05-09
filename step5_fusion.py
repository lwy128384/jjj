#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤5: 多模态融合与视频剪辑
  1. 确认干扰片段并剔除
  2. 综合知识点边界与干扰标记生成剪辑指令
  3. 输出以知识点命名的独立 MP4 片段
  4. 生成 JSON 格式时间戳索引文件

运行方式:
  python step5_fusion.py --video D:/video/lesson/example.mp4

输出:
  D:/video/output/example/segments/知识点1-xxx.mp4  …
  D:/video/output/example/final_index.json
"""

import os
import sys
import json
import re
import subprocess
import argparse
import datetime
from pathlib import Path

# ============================================================
# 默认参数
# ============================================================
try:
    import config as _cfg
    OUTPUT_DIR                        = _cfg.OUTPUT_DIR
    INTERFERENCE_TEACHER_ABSENT_RATIO = _cfg.INTERFERENCE_TEACHER_ABSENT_RATIO
    INTERFERENCE_LOW_SPEECH_RATIO     = _cfg.INTERFERENCE_LOW_SPEECH_RATIO
    INTERFERENCE_SILENCE_THRESHOLD    = _cfg.INTERFERENCE_SILENCE_THRESHOLD
    INTERFERENCE_MIN_DURATION         = _cfg.INTERFERENCE_MIN_DURATION
    SEGMENT_MIN_DURATION              = _cfg.SEGMENT_MIN_DURATION
    SEGMENT_PADDING                   = _cfg.SEGMENT_PADDING
except ImportError:
    OUTPUT_DIR                        = r"D:\video\output"
    INTERFERENCE_TEACHER_ABSENT_RATIO = 0.70
    INTERFERENCE_LOW_SPEECH_RATIO     = 0.80
    INTERFERENCE_SILENCE_THRESHOLD    = 15.0
    INTERFERENCE_MIN_DURATION         = 5.0
    SEGMENT_MIN_DURATION              = 20.0
    SEGMENT_PADDING                   = 1.0

# Maximum allowed gap (seconds) for merging adjacent model-predicted timestamps.
MODEL_INTERFERENCE_MAX_GAP = 1.5
# Small tolerance for floating-point time comparisons (seconds).
TIME_EPSILON = 1e-6
MODEL_PREDICT_EXCEPTIONS = (
    FileNotFoundError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
)
INTERFERENCE_TITLE = "干扰片段"


# ============================================================
# 工具函数
# ============================================================

def get_output_dir(video_path, base_output_dir=None):
    base = base_output_dir or OUTPUT_DIR
    name = Path(video_path).stem
    out  = os.path.join(base, name)
    os.makedirs(out, exist_ok=True)
    return out, name


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def load_multimodal_index(output_dir):
    p = os.path.join(output_dir, "multimodal_index.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"找不到多模态索引，请先运行步骤4: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_filename(name):
    """去除 Windows 文件名非法字符，并截断"""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    return name[:80].strip()


def normalize_knowledge_title(title, fallback_id=None):
    """
    去除历史命名中的“知识点N-”前缀，保留知识点名称主体。
    """
    raw = str(title or "").strip()
    cleaned = re.sub(r"^\s*知识点\s*\d+\s*[-_—:：]?\s*", "", raw)
    if cleaned:
        return cleaned
    if raw:
        return raw
    if fallback_id is None:
        return "未命名片段"
    return f"片段{int(fallback_id) + 1}"


def _merge_source_labels(prev_source, cur_source):
    labels = str(prev_source).split("+") + str(cur_source).split("+")
    return "+".join(sorted(set([x for x in labels if x])))


# ============================================================
# 干扰检测
# ============================================================

def _merge_interference_ranges(interferences):
    """Merge overlapping interference ranges while preserving reason/source labels."""
    if not interferences:
        return []
    interferences = sorted(interferences, key=lambda x: (x["start"], x["end"]))
    merged = [dict(interferences[0])]
    for cur in interferences[1:]:
        prev = merged[-1]
        # Merge touching/overlapping ranges to avoid fragmented interference clips.
        if cur["start"] <= prev["end"] + TIME_EPSILON:
            prev["end"] = max(prev["end"], cur["end"])
            prev["duration"] = round(prev["end"] - prev["start"], 2)
            prev["reasons"] = list(dict.fromkeys(prev.get("reasons", []) + cur.get("reasons", [])))
            prev["source"] = _merge_source_labels(prev.get("source", ""), cur.get("source", ""))
            prev["teacher_absent_ratio"] = round(max(prev.get("teacher_absent_ratio", 0.0),
                                                     cur.get("teacher_absent_ratio", 0.0)), 3)
            prev["silence_ratio"] = round(max(prev.get("silence_ratio", 0.0),
                                              cur.get("silence_ratio", 0.0)), 3)
        else:
            merged.append(dict(cur))
    return merged


def _build_model_interference_ranges(model_times):
    """
    Convert model-predicted interference timestamps into contiguous ranges.

    Timestamps are considered contiguous when adjacent points are within
    MODEL_INTERFERENCE_MAX_GAP seconds and are merged into the same range.
    """
    if not model_times:
        return []
    ts = sorted(set(round(float(t), 2) for t in model_times))
    ranges = []
    start = ts[0]
    prev = ts[0]
    for t in ts[1:]:
        if t - prev <= MODEL_INTERFERENCE_MAX_GAP:
            prev = t
            continue
        dur = prev - start
        if dur >= INTERFERENCE_MIN_DURATION:
            ranges.append((start, prev))
        start = t
        prev = t
    dur = prev - start
    if dur >= INTERFERENCE_MIN_DURATION:
        ranges.append((start, prev))
    return ranges


def _dynamic_interference_thresholds(multimodal_index):
    """
    动态阈值：基于视频整体统计自动微调，避免固定阈值在不同课堂类型上失效。
    """
    stats = multimodal_index.get("stats", {}) or {}
    teacher_presence_ratio = float(stats.get("teacher_presence_ratio", 0.0) or 0.0)
    silence_ratio = float(stats.get("silence_ratio", 0.0) or 0.0)
    duration = max(float(multimodal_index.get("duration", 0.0) or 0.0), 1.0)
    slide_density = len(multimodal_index.get("slide_transitions", []) or []) / duration

    absent_th = float(INTERFERENCE_TEACHER_ABSENT_RATIO)
    low_speech_th = float(INTERFERENCE_LOW_SPEECH_RATIO)
    silence_sec_th = float(INTERFERENCE_SILENCE_THRESHOLD)

    if teacher_presence_ratio < 0.55:
        absent_th += 0.08
    elif teacher_presence_ratio > 0.90:
        absent_th -= 0.05

    if silence_ratio > 0.35:
        low_speech_th += 0.05
        silence_sec_th += 3.0
    elif silence_ratio < 0.12:
        low_speech_th -= 0.05
        silence_sec_th -= 2.0

    if slide_density > 0.02:
        absent_th += 0.03

    absent_th = min(max(absent_th, 0.50), 0.90)
    low_speech_th = min(max(low_speech_th, 0.60), 0.95)
    silence_sec_th = min(max(silence_sec_th, 8.0), 30.0)
    return absent_th, low_speech_th, silence_sec_th


def detect_interference(multimodal_index, model_times=None):
    """
    干扰规则（满足任一即标记）：
      1. 某知识点段内教师缺席比例 > 阈值
      2. 某知识点段内静默比例 > 阈值
      3. 全局连续静默 > 阈值时长
    """
    series   = multimodal_index["time_series"]
    know_segs = multimodal_index["knowledge_segments"]
    interferences = []
    absent_th, low_speech_th, silence_sec_th = _dynamic_interference_thresholds(multimodal_index)

    # 按知识点段分析
    for ks in know_segs:
        pts = [p for p in series if ks["start"] <= p["time"] <= ks["end"]]
        if not pts:
            continue
        n = len(pts)
        absent_ratio  = sum(1 for p in pts if not p["teacher_present"]) / n
        silence_ratio = sum(1 for p in pts if p["is_silence"]) / n

        reasons = []
        if absent_ratio  > absent_th:
            reasons.append(f"教师缺席 {absent_ratio:.0%}")
        if silence_ratio > low_speech_th:
            reasons.append(f"静默占比 {silence_ratio:.0%}")

        if reasons:
            interferences.append({
                "start":               ks["start"],
                "end":                 ks["end"],
                "duration":            ks["end"] - ks["start"],
                "reasons":             reasons,
                "teacher_absent_ratio": round(absent_ratio,  3),
                "silence_ratio":        round(silence_ratio, 3),
                "source":              "knowledge_segment",
            })

    # 扫描连续静默段
    sil_start = None
    for pt in series:
        if pt["is_silence"]:
            if sil_start is None:
                sil_start = pt["time"]
        else:
            if sil_start is not None:
                sil_dur = pt["time"] - sil_start
                if sil_dur >= silence_sec_th:
                    interferences.append({
                        "start":    sil_start,
                        "end":      pt["time"],
                        "duration": sil_dur,
                        "reasons":  [f"连续静默 {sil_dur:.1f}s"],
                        "teacher_absent_ratio": 0.0,
                        "silence_ratio":        1.0,
                        "source":   "silence_scan",
                    })
                sil_start = None

    # 融合模型预测干扰
    for seg_s, seg_e in _build_model_interference_ranges(model_times or []):
        interferences.append({
            "start": seg_s,
            "end": seg_e,
            "duration": round(seg_e - seg_s, 2),
            "reasons": ["模型预测干扰"],
            "teacher_absent_ratio": 0.0,
            "silence_ratio": 0.0,
            "source": "model",
        })

    return _merge_interference_ranges(interferences)


# ============================================================
# 生成剪辑指令
# ============================================================

def build_edit_commands(know_segs, interferences, duration):
    """过滤干扰、添加缓冲，按知识点输出剪辑片段列表（不做片段合并）"""

    def overlap_ratio(start, end):
        for intf in interferences:
            ovlp = min(end, intf["end"]) - max(start, intf["start"])
            if ovlp > 0 and ovlp / max(end - start, 0.001) > 0.5:
                return ovlp / max(end - start, 0.001)
        return 0.0

    valid = []
    dropped = []
    for ks in know_segs:
        normalized_title = normalize_knowledge_title(ks.get("title", ""), ks.get("id"))
        ovlp_ratio = overlap_ratio(ks["start"], ks["end"])
        if ovlp_ratio > 0.5:
            dropped.append({
                "title": INTERFERENCE_TITLE,
                "start": round(float(ks["start"]), 2),
                "end": round(float(ks["end"]), 2),
                "duration": round(float(ks["end"]) - float(ks["start"]), 2),
                "reasons": [f"与干扰区间重叠 {ovlp_ratio:.0%}"],
                "source": "knowledge_overlap_filter",
                "output_policy": "not_exported",
            })
            continue
        seg_dur = ks["end"] - ks["start"]
        if seg_dur < SEGMENT_MIN_DURATION:
            dropped.append({
                "title": INTERFERENCE_TITLE,
                "start": round(float(ks["start"]), 2),
                "end": round(float(ks["end"]), 2),
                "duration": round(float(seg_dur), 2),
                "reasons": [f"片段过短 {seg_dur:.1f}s < {SEGMENT_MIN_DURATION:.1f}s"],
                "source": "min_duration_filter",
                "output_policy": "not_exported",
            })
            continue
        seg_s = max(0.0,     ks["start"] - SEGMENT_PADDING)
        seg_e = min(duration, ks["end"]   + SEGMENT_PADDING)
        valid.append({
            "original_id": ks["id"],
            "title":       normalized_title,
            "start":       round(seg_s, 2),
            "end":         round(seg_e, 2),
            "duration":    round(seg_e - seg_s, 2),
            "keywords":    ks.get("keywords", []),
        })

    # 重新编号（保持每个知识点一个片段）
    for i, seg in enumerate(valid):
        seg["id"] = i

    return valid, dropped


# ============================================================
# 视频剪切
# ============================================================

def cut_segment(video_path, start, end, out_path):
    """ffmpeg 剪切片段，优先 stream copy（快），失败则重新编码"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 快速模式：stream copy
    cmd_copy = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", str(start), "-to", str(end),
        "-c:v", "copy", "-c:a", "copy",
        "-avoid_negative_ts", "1",
        str(out_path),
    ]
    r = subprocess.run(cmd_copy, capture_output=True, text=True)
    if r.returncode == 0 and os.path.getsize(out_path) > 1024:
        return

    # 重新编码模式（精确但慢）
    print(f"    stream copy 失败，改用重新编码…")
    cmd_enc = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t",  str(round(end - start, 2)),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ]
    r2 = subprocess.run(cmd_enc, capture_output=True, text=True)
    if r2.returncode != 0:
        raise RuntimeError(f"ffmpeg 剪切失败:\n{r2.stderr[-600:]}")


# ============================================================
# 核心融合
# ============================================================

def fuse_and_cut(video_path, output_dir, video_name):
    print(f"\n[步骤5] 多模态融合与剪辑: {video_name}")

    if not check_ffmpeg():
        raise RuntimeError(
            "未找到 ffmpeg！\n"
            "Windows 安装：\n"
            "  1. 下载 https://www.gyan.dev/ffmpeg/builds/\n"
            "  2. 解压到 C:\\ffmpeg\n"
            "  3. 将 C:\\ffmpeg\\bin 加入系统 PATH"
        )

    idx      = load_multimodal_index(output_dir)
    duration = idx["duration"]
    know_segs = idx["knowledge_segments"]
    print(f"  时长: {duration:.1f}s  知识点数: {len(know_segs)}")

    # 干扰检测
    print("  检测干扰片段…")
    model_times = []
    try:
        from train import predict_interference
    except ImportError:
        predict_interference = None

    if predict_interference is not None:
        try:
            model_times = predict_interference(idx)
            if model_times:
                print(f"  已应用训练干扰模型（命中时刻 {len(model_times)} 个）")
        except MODEL_PREDICT_EXCEPTIONS as e:
            print(f"  训练干扰模型调用失败，回退规则检测: {e}")
            model_times = []
    interferences = detect_interference(idx, model_times=model_times)
    print(f"  共 {len(interferences)} 个干扰片段")
    for intf in interferences:
        print(f"    [{intf['start']:.0f}s–{intf['end']:.0f}s]  {', '.join(intf['reasons'])}")

    # 生成剪辑指令
    print("  生成剪辑指令…")
    commands, dropped_as_interference = build_edit_commands(know_segs, interferences, duration)
    print(f"  有效片段: {len(commands)} 个")

    # 剪切视频
    seg_dir = os.path.join(output_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)

    clips = []
    for cmd in commands:
        fname    = sanitize_filename(cmd["title"]) + ".mp4"
        out_path = os.path.join(seg_dir, fname)
        print(f"  剪切 [{cmd['start']:.0f}s–{cmd['end']:.0f}s] → {fname}")
        try:
            cut_segment(video_path, cmd["start"], cmd["end"], out_path)
            file_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            clips.append({
                "id":           cmd["id"],
                "title":        cmd["title"],
                "start":        cmd["start"],
                "end":          cmd["end"],
                "duration":     cmd["duration"],
                "output_file":  out_path,
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "keywords":     cmd["keywords"],
                "status":       "ok",
            })
        except Exception as e:
            print(f"    ✗ 剪切失败: {e}")
            clips.append({
                "id":    cmd["id"],
                "title": cmd["title"],
                "start": cmd["start"],
                "end":   cmd["end"],
                "status": "failed",
                "error":  str(e),
            })

    # 汇总 removed 片段
    removed = []
    for intf in interferences:
        removed.append({
            "title":    INTERFERENCE_TITLE,
            "start":    intf["start"],
            "end":      intf["end"],
            "duration": intf["duration"],
            "reasons":  intf["reasons"],
            "source":   intf.get("source", "rule"),
            "output_policy": "not_exported",
        })
    removed.extend(dropped_as_interference)

    final_index = {
        "video_name":       video_name,
        "video_path":       str(video_path),
        "processed_at":     datetime.datetime.now().isoformat(timespec="seconds"),
        "total_clips":      len(clips),
        "clips":            clips,
        "removed_segments": removed,
        "stats": {
            "total_output_clips":      len([c for c in clips if c.get("status") == "ok"]),
            "total_removed_segments":  len(removed),
            "total_output_duration_s": sum(c.get("duration", 0)
                                           for c in clips if c.get("status") == "ok"),
        },
    }

    out_file = os.path.join(output_dir, "final_index.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_index, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ 剪辑完成")
    print(f"    输出片段: {final_index['stats']['total_output_clips']} 个")
    print(f"    剔除片段: {final_index['stats']['total_removed_segments']} 个")
    print(f"    总输出时长: {final_index['stats']['total_output_duration_s']:.0f}s")
    print(f"    索引文件: {out_file}")
    print(f"    视频目录: {seg_dir}")
    return final_index


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="步骤5: 多模态融合与视频剪辑")
    parser.add_argument("--video",  required=True, help="视频文件路径")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出根目录")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频不存在: {args.video}")
        sys.exit(1)

    out_dir, vname = get_output_dir(args.video, args.output)
    fuse_and_cut(args.video, out_dir, vname)


if __name__ == "__main__":
    main()
