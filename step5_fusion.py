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
import math
import re
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
    INTERFERENCE_NO_KNOWLEDGE_THRESHOLD = _cfg.INTERFERENCE_NO_KNOWLEDGE_THRESHOLD
    INTERFERENCE_NO_KNOWLEDGE_TEXT_SIM_THRESHOLD = _cfg.INTERFERENCE_NO_KNOWLEDGE_TEXT_SIM_THRESHOLD
    INTERFERENCE_QA_PATTERN_ENABLED   = _cfg.INTERFERENCE_QA_PATTERN_ENABLED
    INTERFERENCE_QA_MIN_SWITCHES      = _cfg.INTERFERENCE_QA_MIN_SWITCHES
    INTERFERENCE_QA_MAX_DURATION      = _cfg.INTERFERENCE_QA_MAX_DURATION
    INTERFERENCE_TEACHER_QA_CUES      = _cfg.INTERFERENCE_TEACHER_QA_CUES
    DIARIZATION_STUDENT_CUES          = _cfg.DIARIZATION_STUDENT_CUES
    DIARIZATION_TEACHER_CUES          = _cfg.DIARIZATION_TEACHER_CUES
    SEGMENT_MIN_DURATION              = _cfg.SEGMENT_MIN_DURATION
    SEGMENT_PADDING                   = _cfg.SEGMENT_PADDING
    INTERFERENCE_SEGMENT_TITLE        = _cfg.INTERFERENCE_SEGMENT_TITLE
except ImportError:
    OUTPUT_DIR                        = r"D:\video\output"
    INTERFERENCE_TEACHER_ABSENT_RATIO = 0.70
    INTERFERENCE_LOW_SPEECH_RATIO     = 0.80
    INTERFERENCE_SILENCE_THRESHOLD    = 15.0
    INTERFERENCE_MIN_DURATION         = 5.0
    INTERFERENCE_NO_KNOWLEDGE_THRESHOLD = 30.0
    INTERFERENCE_NO_KNOWLEDGE_TEXT_SIM_THRESHOLD = 0.85
    INTERFERENCE_QA_PATTERN_ENABLED   = True
    INTERFERENCE_QA_MIN_SWITCHES      = 2
    INTERFERENCE_QA_MAX_DURATION      = 60.0
    INTERFERENCE_TEACHER_QA_CUES      = ["你来回答", "这位同学", "请回答", "提问", "谁来说一下", "举手"]
    DIARIZATION_STUDENT_CUES          = ["老师", "请问", "我想问", "是不是", "对吗", "为什么", "怎么"]
    DIARIZATION_TEACHER_CUES          = ["同学们", "我们", "下面", "来看", "总结", "注意"]
    SEGMENT_MIN_DURATION              = 20.0
    SEGMENT_PADDING                   = 1.0
    INTERFERENCE_SEGMENT_TITLE        = "干扰片段"

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
INTERFERENCE_TITLE = INTERFERENCE_SEGMENT_TITLE
ABSENT_THRESHOLD_MIN = 0.50
ABSENT_THRESHOLD_MAX = 0.90
LOW_SPEECH_THRESHOLD_MIN = 0.60
LOW_SPEECH_THRESHOLD_MAX = 0.95
SILENCE_THRESHOLD_MIN = 8.0
SILENCE_THRESHOLD_MAX = 30.0


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


def format_hms_floor_ceil(start_sec, end_sec):
    raw_start = float(start_sec)
    raw_end = float(end_sec)
    if raw_end < raw_start:
        raw_start, raw_end = raw_end, raw_start
    start = math.floor(raw_start)
    end = math.ceil(raw_end)

    def _to_hms(total_seconds):
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h}:{m:02d}:{s:02d}"

    return _to_hms(start), _to_hms(end), int(end - start)


def _format_clip_time_fields(clip):
    start_hms, end_hms, duration_sec = format_hms_floor_ceil(clip["start"], clip["end"])
    return {**clip, "start": start_hms, "end": end_hms, "duration": duration_sec}


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

    absent_th = min(max(absent_th, ABSENT_THRESHOLD_MIN), ABSENT_THRESHOLD_MAX)
    low_speech_th = min(max(low_speech_th, LOW_SPEECH_THRESHOLD_MIN), LOW_SPEECH_THRESHOLD_MAX)
    silence_sec_th = min(max(silence_sec_th, SILENCE_THRESHOLD_MIN), SILENCE_THRESHOLD_MAX)
    return absent_th, low_speech_th, silence_sec_th


def _tokenize_for_jaccard(text):
    if not text:
        return set()
    txt = str(text).lower().strip()
    parts = re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", txt)
    return set(parts)


def _jaccard_similarity(a, b):
    ta = _tokenize_for_jaccard(a)
    tb = _tokenize_for_jaccard(b)
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = max(len(ta | tb), 1)
    return inter / union


def _is_teacher_speaking(pt):
    return (
        bool(pt.get("teacher_present"))
        and not bool(pt.get("is_silence"))
        and str(pt.get("speaker", "")).strip() == "教师"
    )


def _knowledge_key(pt):
    k_id = pt.get("knowledge_id")
    k_title = str(pt.get("knowledge_title", "")).strip()
    if k_id is None:
        k_id = -1
    return k_id, k_title


def _is_no_knowledge_id(k_id):
    if k_id is None:
        return True
    if isinstance(k_id, str):
        s = k_id.strip().lower()
        return s in {"", "0", "none", "null", "未知", "unknown"}
    try:
        return int(k_id) <= 0
    except (TypeError, ValueError):
        return False


def _collect_no_knowledge_progress_interference(series):
    """
    规则4：教师在场连续讲话，但知识点长期不推进（或为空）且语义停滞。
    """
    if not series:
        return []
    segments = []
    start_idx = None
    base_key = None
    text_sims = []
    text_pairs = 0
    prev_text = ""

    def _flush(end_idx):
        nonlocal start_idx, base_key, text_sims, text_pairs, prev_text
        if start_idx is None or end_idx is None or end_idx < start_idx:
            start_idx = None
            base_key = None
            text_sims = []
            text_pairs = 0
            prev_text = ""
            return
        st = float(series[start_idx]["time"])
        ed = float(series[end_idx]["time"])
        dur = ed - st
        avg_sim = (sum(text_sims) / len(text_sims)) if text_sims else 1.0
        k_id = base_key[0] if base_key else None
        no_knowledge = _is_no_knowledge_id(k_id)
        semantic_stagnant = (text_pairs == 0 and no_knowledge) or (avg_sim >= INTERFERENCE_NO_KNOWLEDGE_TEXT_SIM_THRESHOLD)
        if dur >= INTERFERENCE_NO_KNOWLEDGE_THRESHOLD and semantic_stagnant:
            segments.append({
                "start": st,
                "end": ed,
                "duration": round(dur, 2),
                "reasons": [f"教师在场但无知识点推进 {dur:.0f}s"],
                "teacher_absent_ratio": 0.0,
                "silence_ratio": 0.0,
                "source": "no_knowledge_progress",
            })
        start_idx = None
        base_key = None
        text_sims = []
        text_pairs = 0
        prev_text = ""

    for i, pt in enumerate(series):
        if not _is_teacher_speaking(pt):
            _flush(i - 1)
            continue
        cur_key = _knowledge_key(pt)
        cur_text = str(pt.get("speech_text", "") or "").strip()
        if start_idx is None:
            start_idx = i
            base_key = cur_key
            prev_text = cur_text
            continue
        if cur_key != base_key:
            _flush(i - 1)
            start_idx = i
            base_key = cur_key
            prev_text = cur_text
            continue
        if prev_text and cur_text:
            text_sims.append(_jaccard_similarity(prev_text, cur_text))
            text_pairs += 1
        prev_text = cur_text
    _flush(len(series) - 1)
    return segments


def _collect_teacher_student_qa_interference(series):
    """
    规则5：师生说话人切换 + 问答 cue 词命中。
    """
    if not INTERFERENCE_QA_PATTERN_ENABLED or not series:
        return []
    teacher_cues = set(str(c).strip() for c in (list(DIARIZATION_TEACHER_CUES) + list(INTERFERENCE_TEACHER_QA_CUES)) if str(c).strip())
    student_cues = set(str(c).strip() for c in DIARIZATION_STUDENT_CUES if str(c).strip())
    valid_speakers = {"教师", "学生"}
    points = [pt for pt in series if not pt.get("is_silence") and str(pt.get("speaker", "")).strip() in valid_speakers]
    if not points:
        return []

    sessions = []
    cur = [points[0]]
    for pt in points[1:]:
        if float(pt["time"]) - float(cur[-1]["time"]) <= 2.0:
            cur.append(pt)
        else:
            sessions.append(cur)
            cur = [pt]
    sessions.append(cur)

    interferences = []
    for session in sessions:
        if len(session) < 3:
            continue
        start = float(session[0]["time"])
        end = float(session[-1]["time"])
        dur = end - start
        if dur <= 0 or dur > INTERFERENCE_QA_MAX_DURATION:
            continue
        switches = 0
        prev_spk = str(session[0].get("speaker", "")).strip()
        teacher_hit = False
        student_hit = False
        for pt in session:
            spk = str(pt.get("speaker", "")).strip()
            txt = str(pt.get("speech_text", "") or "")
            if any(c in txt for c in teacher_cues):
                teacher_hit = True
            if any(c in txt for c in student_cues):
                student_hit = True
            if spk != prev_spk:
                switches += 1
                prev_spk = spk
        has_t_s_t = False
        for i in range(2, len(session)):
            s0 = str(session[i - 2].get("speaker", "")).strip()
            s1 = str(session[i - 1].get("speaker", "")).strip()
            s2 = str(session[i].get("speaker", "")).strip()
            if s0 == "教师" and s1 == "学生" and s2 == "教师":
                has_t_s_t = True
                break
        if switches >= INTERFERENCE_QA_MIN_SWITCHES and teacher_hit and student_hit and has_t_s_t:
            interferences.append({
                "start": start,
                "end": end,
                "duration": round(dur, 2),
                "reasons": [f"师生问答对话 {switches} 次"],
                "teacher_absent_ratio": 0.0,
                "silence_ratio": 0.0,
                "source": "qa_dialogue",
            })
    return interferences


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

    # 规则4：教师在场但知识点无推进（语义空转）
    interferences.extend(_collect_no_knowledge_progress_interference(series))

    # 规则5：师生问答对话模式
    interferences.extend(_collect_teacher_student_qa_interference(series))

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

    def _decode_subprocess_output(raw):
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    # 快速模式：stream copy
    cmd_copy = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", str(start), "-to", str(end),
        "-c:v", "copy", "-c:a", "copy",
        "-avoid_negative_ts", "1",
        str(out_path),
    ]
    r = subprocess.run(cmd_copy, capture_output=True)
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
    r2 = subprocess.run(cmd_enc, capture_output=True)
    if r2.returncode != 0:
        stderr_text = _decode_subprocess_output(r2.stderr)
        raise RuntimeError(f"ffmpeg 剪切失败:\n{stderr_text[-600:]}")


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

    clips_for_index = [_format_clip_time_fields(c) for c in clips]
    final_index = {
        "video_name":       video_name,
        "video_path":       str(video_path),
        "processed_at":     datetime.datetime.now().isoformat(timespec="seconds"),
        "total_clips":      len(clips),
        "clips":            clips_for_index,
        "removed_segments": removed,
        "stats": {
            "total_output_clips":      len([c for c in clips if c.get("status") == "ok"]),
            "total_removed_segments":  len(removed),
            "total_output_duration_s": sum(c.get("duration", 0)
                                           for c in clips_for_index if c.get("status") == "ok"),
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
