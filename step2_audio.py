#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤2: 语音维度分析
  1. 从视频提取音频（ffmpeg）
  2. 语音转录（faster-whisper，纯 CPU）
  3. 转录文本纠错（专业词典 + 拼音模糊匹配）
  4. 说话人区分（二分类聚类 + 声纹辅助复判）
  5. 置信度标记

运行方式:
  python step2_audio.py --video D:/video/lesson/example.mp4

输出:
  D:/video/output/example/audio_features.json
"""

import os
import sys
import json
import subprocess
import argparse
import tempfile
import re
import numpy as np
from pathlib import Path

# ============================================================
# 默认参数
# ============================================================
try:
    import config as _cfg
    OUTPUT_DIR                  = _cfg.OUTPUT_DIR
    WHISPER_MODEL_SIZE          = _cfg.WHISPER_MODEL_SIZE
    WHISPER_LANGUAGE            = _cfg.WHISPER_LANGUAGE
    WHISPER_BEAM_SIZE           = _cfg.WHISPER_BEAM_SIZE
    DIARIZATION_N_CLUSTERS      = _cfg.DIARIZATION_N_CLUSTERS
    DIARIZATION_TEXT_WEIGHT     = _cfg.DIARIZATION_TEXT_WEIGHT
    DIARIZATION_ACOUSTIC_WEIGHT = _cfg.DIARIZATION_ACOUSTIC_WEIGHT
    DIARIZATION_FILLER_CUES     = _cfg.DIARIZATION_FILLER_CUES
    DIARIZATION_TEACHER_CUES    = _cfg.DIARIZATION_TEACHER_CUES
    DIARIZATION_STUDENT_CUES    = _cfg.DIARIZATION_STUDENT_CUES
    DIARIZATION_SMOOTH_WINDOW   = _cfg.DIARIZATION_SMOOTH_WINDOW
    DIARIZATION_SMOOTH_MAX_DURATION = _cfg.DIARIZATION_SMOOTH_MAX_DURATION
    DIARIZATION_SMOOTH_MIN_NEIGHBORS = _cfg.DIARIZATION_SMOOTH_MIN_NEIGHBORS
    DIARIZATION_TEACHER_PROB_CENTER_QUANTILE = _cfg.DIARIZATION_TEACHER_PROB_CENTER_QUANTILE
    DIARIZATION_TEACHER_PROB_SIGMOID_SCALE = _cfg.DIARIZATION_TEACHER_PROB_SIGMOID_SCALE
    DIARIZATION_TEACHER_PROB_BIAS = _cfg.DIARIZATION_TEACHER_PROB_BIAS
    DIARIZATION_TEACHER_PROB_THRESHOLD = _cfg.DIARIZATION_TEACHER_PROB_THRESHOLD
    DIARIZATION_VOICEPRINT_BORDERLINE_LOW = _cfg.DIARIZATION_VOICEPRINT_BORDERLINE_LOW
    DIARIZATION_VOICEPRINT_BORDERLINE_HIGH = _cfg.DIARIZATION_VOICEPRINT_BORDERLINE_HIGH
    DIARIZATION_VOICEPRINT_TEACHER_PROTO_QUANTILE = _cfg.DIARIZATION_VOICEPRINT_TEACHER_PROTO_QUANTILE
    DIARIZATION_VOICEPRINT_ASSIST_ENABLED = _cfg.DIARIZATION_VOICEPRINT_ASSIST_ENABLED
    DIARIZATION_VOICEPRINT_MIN_TEACHER_SAMPLES = _cfg.DIARIZATION_VOICEPRINT_MIN_TEACHER_SAMPLES
    DIARIZATION_VOICEPRINT_MIN_SEGMENT_DURATION = _cfg.DIARIZATION_VOICEPRINT_MIN_SEGMENT_DURATION
    DIARIZATION_VOICEPRINT_MAX_STUDENT_DURATION = _cfg.DIARIZATION_VOICEPRINT_MAX_STUDENT_DURATION
    DIARIZATION_VOICEPRINT_SIMILARITY_THRESHOLD = _cfg.DIARIZATION_VOICEPRINT_SIMILARITY_THRESHOLD
    DIARIZATION_VOICEPRINT_STUDENT_MARGIN = _cfg.DIARIZATION_VOICEPRINT_STUDENT_MARGIN
    SPEECH_CONFIDENCE_THRESHOLD = _cfg.SPEECH_CONFIDENCE_THRESHOLD
    NO_SPEECH_PROB_THRESHOLD    = _cfg.NO_SPEECH_PROB_THRESHOLD
    NO_SPEECH_IGNORE_WITH_TEXT  = _cfg.NO_SPEECH_IGNORE_WITH_TEXT
    NO_SPEECH_TEXT_SHORT_LEN    = _cfg.NO_SPEECH_TEXT_SHORT_LEN
    STEP2_ENABLE_TEXT_CORRECTION = _cfg.STEP2_ENABLE_TEXT_CORRECTION
    STEP2_TEXT_CORRECTION_TERMS = list(_cfg.STEP2_TEXT_CORRECTION_TERMS)
    STEP2_TEXT_CORRECTION_MIN_CHARS = _cfg.STEP2_TEXT_CORRECTION_MIN_CHARS
    STEP2_TEXT_CORRECTION_MAX_PINYIN_NORM_DIST = _cfg.STEP2_TEXT_CORRECTION_MAX_PINYIN_NORM_DIST
    STEP2_TEXT_CORRECTION_MAX_CHAR_DIST = _cfg.STEP2_TEXT_CORRECTION_MAX_CHAR_DIST
    STEP2_TEXT_CORRECTION_MAX_LENGTH_DIFF = _cfg.STEP2_TEXT_CORRECTION_MAX_LENGTH_DIFF
    STEP2_TEXT_CORRECTION_CHAR_WEIGHT = _cfg.STEP2_TEXT_CORRECTION_CHAR_WEIGHT
    STEP2_SCRIPT_CONVERSION_MODE = _cfg.STEP2_SCRIPT_CONVERSION_MODE
except ImportError:
    OUTPUT_DIR                  = r"D:\video\output"
    WHISPER_MODEL_SIZE          = "base"
    WHISPER_LANGUAGE            = "zh"
    WHISPER_BEAM_SIZE           = 3
    DIARIZATION_N_CLUSTERS      = 2
    DIARIZATION_TEXT_WEIGHT     = 0.38
    DIARIZATION_ACOUSTIC_WEIGHT = 0.62
    DIARIZATION_FILLER_CUES     = ["嗯", "啊", "呃", "这个", "那个", "就是", "那么"]
    DIARIZATION_TEACHER_BASE_CUES = [
        "我们", "下面", "今天", "讲", "来看", "举个例子", "同学们", "回顾",
        "总结", "总之", "注意", "定义", "公式", "原理", "人工智能", "历史",
        "先", "然后", "接下来", "这个问题",
    ]
    DIARIZATION_TEACHER_CUES = DIARIZATION_TEACHER_BASE_CUES + DIARIZATION_FILLER_CUES + ["大家", "注意看", "来看一下"]
    DIARIZATION_STUDENT_CUES    = [
        "老师", "请问", "我想问", "是不是", "对吗", "为什么", "怎么",
        "听不清", "没听懂", "可以再说", "啥意思",
    ]
    DIARIZATION_SMOOTH_WINDOW   = 3
    DIARIZATION_SMOOTH_MAX_DURATION = 4.0
    DIARIZATION_SMOOTH_MIN_NEIGHBORS = 2
    DIARIZATION_TEACHER_PROB_CENTER_QUANTILE = 0.35
    DIARIZATION_TEACHER_PROB_SIGMOID_SCALE = 1.0
    DIARIZATION_TEACHER_PROB_BIAS = 0.30
    DIARIZATION_TEACHER_PROB_THRESHOLD = 0.45
    DIARIZATION_VOICEPRINT_BORDERLINE_LOW = 0.28
    DIARIZATION_VOICEPRINT_BORDERLINE_HIGH = 0.60
    DIARIZATION_VOICEPRINT_TEACHER_PROTO_QUANTILE = 0.65
    DIARIZATION_VOICEPRINT_ASSIST_ENABLED = True
    DIARIZATION_VOICEPRINT_MIN_TEACHER_SAMPLES = 2
    DIARIZATION_VOICEPRINT_MIN_SEGMENT_DURATION = 0.8
    DIARIZATION_VOICEPRINT_MAX_STUDENT_DURATION = 40.0
    DIARIZATION_VOICEPRINT_SIMILARITY_THRESHOLD = 0.82
    DIARIZATION_VOICEPRINT_STUDENT_MARGIN = 0.03
    SPEECH_CONFIDENCE_THRESHOLD = 0.60
    NO_SPEECH_PROB_THRESHOLD    = 0.80
    NO_SPEECH_IGNORE_WITH_TEXT  = True
    NO_SPEECH_TEXT_SHORT_LEN    = 3
    STEP2_ENABLE_TEXT_CORRECTION = True
    STEP2_TEXT_CORRECTION_TERMS = [
        "人工智能", "机器学习", "深度学习", "神经网络", "图灵", "图灵测试",
        "亚里士多德", "算法", "数据集", "低谷", "模型", "训练", "推理",
    ]
    STEP2_TEXT_CORRECTION_MIN_CHARS = 2
    STEP2_TEXT_CORRECTION_MAX_PINYIN_NORM_DIST = 0.22
    STEP2_TEXT_CORRECTION_MAX_CHAR_DIST = 1
    STEP2_TEXT_CORRECTION_MAX_LENGTH_DIFF = 1
    STEP2_TEXT_CORRECTION_CHAR_WEIGHT = 0.05
    STEP2_SCRIPT_CONVERSION_MODE = "t2s"


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


def extract_audio(video_path, wav_path):
    """用 ffmpeg 提取 16 kHz 单声道 PCM WAV"""
    print(f"  提取音频 → {wav_path}")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 提取音频失败:\n{r.stderr[-800:]}")


# ============================================================
# 语音转录
# ============================================================

def transcribe(audio_path):
    """faster-whisper 转录，返回 (segments_list, info)"""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("请安装 faster-whisper:\n  pip install faster-whisper")

    print(f"  加载 Whisper 模型 [{WHISPER_MODEL_SIZE}]（首次运行需下载 ~150 MB）…")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device       = "cpu",
        compute_type = "int8",       # CPU 下 int8 最快
    )

    print("  开始转录…")
    gen, info = model.transcribe(
        str(audio_path),
        language         = WHISPER_LANGUAGE,
        beam_size        = WHISPER_BEAM_SIZE,
        word_timestamps  = False,
        vad_filter       = True,
        vad_parameters   = dict(min_silence_duration_ms=500, speech_pad_ms=200),
    )

    segments = []
    for seg in gen:
        conf = float(1.0 - seg.no_speech_prob)
        segments.append({
            "id":               seg.id,
            "start":            round(float(seg.start), 2),
            "end":              round(float(seg.end),   2),
            "text":             seg.text.strip(),
            "no_speech_prob":   round(float(seg.no_speech_prob), 4),
            "confidence":       round(conf, 4),
            "is_low_confidence": conf < SPEECH_CONFIDENCE_THRESHOLD,
        })

    print(f"  语言: {info.language}  概率: {info.language_probability:.2f}")
    print(f"  转录片段: {len(segments)} 个")
    return segments, info


def _should_keep_speech_segment(seg):
    no_sp = float(seg.get("no_speech_prob", 1.0) or 1.0)
    if no_sp <= NO_SPEECH_PROB_THRESHOLD:
        return True
    text = str(seg.get("text", "")).strip()
    if not (NO_SPEECH_IGNORE_WITH_TEXT and text):
        return False
    return len(text) > NO_SPEECH_TEXT_SHORT_LEN


def normalize_transcription_script(segments):
    mode = str(STEP2_SCRIPT_CONVERSION_MODE or "").strip().lower()
    if mode in ("", "none", "off", "false"):
        return segments, {
            "enabled": False,
            "mode": "none",
            "status": "disabled",
            "attempted_segments": len(segments),
            "changed_segments": 0,
        }
    try:
        import opencc
    except ImportError:
        print(
            "  警告: 缺少 opencc-python-reimplemented，跳过简繁转换\n"
            "    pip install opencc-python-reimplemented"
        )
        return segments, {
            "enabled": False,
            "mode": mode,
            "status": "missing_opencc",
            "attempted_segments": len(segments),
            "changed_segments": 0,
        }
    try:
        converter = opencc.OpenCC(mode)
    except Exception as e:
        print(f"  警告: STEP2_SCRIPT_CONVERSION_MODE={mode!r} 无效，跳过简繁转换（{e}）")
        return segments, {
            "enabled": False,
            "mode": mode,
            "status": "invalid_mode",
            "attempted_segments": len(segments),
            "changed_segments": 0,
        }

    changed_segments = 0
    for seg in segments:
        raw_text = str(seg.get("text", ""))
        if not raw_text:
            continue
        new_text = converter.convert(raw_text)
        if new_text != raw_text:
            seg["text"] = new_text
            changed_segments += 1
    print(f"  文本字形统一({mode}): 命中 {changed_segments} 段")
    return segments, {
        "enabled": True,
        "mode": mode,
        "status": "ok",
        "attempted_segments": len(segments),
        "changed_segments": changed_segments,
    }


# ============================================================
# 转录文本纠错（专业词典 + 拼音模糊匹配）
# ============================================================

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _is_cjk_token(token):
    if not token:
        return False
    return _CJK_RE.fullmatch(token) is not None


def _levenshtein_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            rep = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, rep))
        prev = cur
    return prev[-1]


def _build_term_index(terms, lazy_pinyin):
    cleaned = []
    for t in terms or []:
        s = str(t).strip()
        if not s:
            continue
        if len(s) < STEP2_TEXT_CORRECTION_MIN_CHARS:
            continue
        if not _is_cjk_token(s):
            continue
        cleaned.append(s)
    uniq_terms = sorted(set(cleaned), key=lambda x: (-len(x), x))
    term_items = []
    for term in uniq_terms:
        term_py = " ".join(lazy_pinyin(term, errors="ignore"))
        if not term_py:
            continue
        term_items.append((term, term_py))
    return term_items


def _best_term_match(token, term_items, lazy_pinyin):
    if not _is_cjk_token(token):
        return token, None
    if len(token) < STEP2_TEXT_CORRECTION_MIN_CHARS:
        return token, None
    token_py = " ".join(lazy_pinyin(token, errors="ignore"))
    if not token_py:
        return token, None
    token_len = len(token)
    token_py_len = len(token_py)

    best_term = None
    best_score = float("inf")
    best_info = None
    for term, term_py in term_items:
        if term == token:
            return token, None
        term_len = len(term)
        if abs(term_len - token_len) > STEP2_TEXT_CORRECTION_MAX_LENGTH_DIFF:
            continue
        py_dist = _levenshtein_distance(token_py, term_py)
        py_norm = py_dist / max(token_py_len, len(term_py))
        if py_norm > STEP2_TEXT_CORRECTION_MAX_PINYIN_NORM_DIST:
            continue
        char_dist = _levenshtein_distance(token, term)
        if char_dist > STEP2_TEXT_CORRECTION_MAX_CHAR_DIST:
            continue
        score = py_norm + STEP2_TEXT_CORRECTION_CHAR_WEIGHT * char_dist
        if score < best_score:
            best_score = score
            best_term = term
            best_info = {
                "original": token,
                "corrected": term,
                "pinyin_norm_dist": round(py_norm, 4),
                "char_dist": int(char_dist),
            }
    if best_term is None:
        return token, None
    return best_term, best_info


def correct_transcription_with_pinyin_fuzzy(segments):
    if not STEP2_ENABLE_TEXT_CORRECTION:
        return segments, {"enabled": False, "attempted_segments": 0, "changed_segments": 0, "total_replacements": 0}
    try:
        import jieba
    except ImportError:
        print("  警告: 缺少 jieba，跳过步骤2文本纠错")
        return segments, {"enabled": False, "attempted_segments": len(segments), "changed_segments": 0, "total_replacements": 0}
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        print("  警告: 缺少 pypinyin，跳过步骤2文本纠错")
        return segments, {"enabled": False, "attempted_segments": len(segments), "changed_segments": 0, "total_replacements": 0}

    term_items = _build_term_index(STEP2_TEXT_CORRECTION_TERMS, lazy_pinyin)
    if not term_items:
        return segments, {"enabled": True, "attempted_segments": len(segments), "changed_segments": 0, "total_replacements": 0}

    changed_segments = 0
    total_replacements = 0
    for seg in segments:
        raw_text = str(seg.get("text", ""))
        if not raw_text.strip():
            continue
        tokens = jieba.lcut(raw_text, cut_all=False)
        if not tokens:
            continue
        replacement_logs = []
        new_tokens = []
        for token in tokens:
            corrected, info = _best_term_match(token, term_items, lazy_pinyin)
            new_tokens.append(corrected)
            if info:
                replacement_logs.append(info)
        new_text = "".join(new_tokens)
        if replacement_logs and new_text != raw_text:
            seg["text_raw"] = raw_text
            seg["text"] = new_text
            seg["text_correction"] = replacement_logs
            changed_segments += 1
            total_replacements += len(replacement_logs)

    print(f"  文本纠错: 命中 {changed_segments} 段 / 替换 {total_replacements} 处")
    return segments, {
        "enabled": True,
        "attempted_segments": len(segments),
        "changed_segments": changed_segments,
        "total_replacements": total_replacements,
    }


# ============================================================
# 说话人区分
# ============================================================

def diarize_speakers(audio_path, segments):
    """
    基于声学 + 文本特征的二分类说话人聚类（教师 / 学生）。
    """
    if len(segments) < 2:
        for s in segments:
            s["speaker"] = "教师"
        return segments, ["教师", "学生"]

    print("  说话人区分（二分类聚类 + 文本声学融合）…")

    try:
        import librosa
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  警告: 缺少 librosa 或 scikit-learn，所有片段归为「教师」")
        for s in segments:
            s["speaker"] = "教师"
        return segments, ["教师", "学生"]

    def _safe_zscore(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return arr
        std = np.std(arr)
        if std < 1e-6:
            return np.zeros_like(arr)
        return (arr - np.mean(arr)) / std

    def _normalize_to_prob(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) == 0:
            return arr
        center_q = float(np.clip(DIARIZATION_TEACHER_PROB_CENTER_QUANTILE, 0.05, 0.95))
        center = float(np.quantile(arr, center_q))
        mad = float(np.median(np.abs(arr - center)))
        std = float(np.std(arr))
        spread = mad if mad > 1e-6 else std
        if spread < 1e-6:
            return np.full_like(arr, 0.5, dtype=float)
        scale = float(max(DIARIZATION_TEACHER_PROB_SIGMOID_SCALE, 1e-6))
        logit = ((arr - center) / spread) * scale + float(DIARIZATION_TEACHER_PROB_BIAS)
        logit = np.clip(logit, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-logit))

    def _text_teacher_score(text):
        LENGTH_NORM_FACTOR = 35.0
        MAX_LENGTH_BONUS = 1.6
        MAX_FILLER_BONUS_COUNT = 5
        FILLER_BONUS_WEIGHT = 0.2
        MAX_REPEAT_BONUS_COUNT = 3
        REPEAT_BONUS_WEIGHT = 0.35

        t = (text or "").strip()
        if not t:
            return -0.2
        compact_text = "".join(t.split())
        teacher_hits = sum(1 for cue in DIARIZATION_TEACHER_CUES if cue in t)
        student_hits = sum(1 for cue in DIARIZATION_STUDENT_CUES if cue in t)
        question_hits = t.count("？") + t.count("?")
        filler_hits = sum(compact_text.count(term) for term in DIARIZATION_FILLER_CUES)
        repeated_hits = sum(1 for term in DIARIZATION_FILLER_CUES if term * 2 in compact_text)
        # 教师完整讲解常明显长于学生插话，适当提高长文本加分上限。
        length_bonus = min(len(compact_text) / LENGTH_NORM_FACTOR, MAX_LENGTH_BONUS)
        return (
            teacher_hits * 1.0
            - student_hits * 1.0
            # 教师设问/反问常见，问号仅作弱惩罚，避免系统性压低教师分。
            - question_hits * 0.25
            + length_bonus
            + min(filler_hits, MAX_FILLER_BONUS_COUNT) * FILLER_BONUS_WEIGHT
            + min(repeated_hits, MAX_REPEAT_BONUS_COUNT) * REPEAT_BONUS_WEIGHT
        )

    def _l2_normalize(vec):
        vec = np.asarray(vec, dtype=float)
        nrm = np.linalg.norm(vec)
        if nrm < 1e-8:
            return vec
        return vec / nrm

    def _cosine_sim(a, b):
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an < 1e-8 or bn < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    def _smooth_labels_inplace(segs):
        n = len(segs)
        if n <= 2:
            return
        labels = [s.get("speaker", "教师") for s in segs]
        for i in range(n):
            cur_label = labels[i]
            cur_dur = float(segs[i]["end"] - segs[i]["start"])
            if cur_dur > DIARIZATION_SMOOTH_MAX_DURATION:
                continue
            left = max(0, i - DIARIZATION_SMOOTH_WINDOW)
            right = min(n, i + DIARIZATION_SMOOTH_WINDOW + 1)
            neigh = [labels[j] for j in range(left, right) if j != i]
            if len(neigh) < DIARIZATION_SMOOTH_MIN_NEIGHBORS:
                continue
            teacher_votes = sum(1 for x in neigh if x == "教师")
            student_votes = sum(1 for x in neigh if x == "学生")
            if teacher_votes == student_votes:
                continue
            maj = "教师" if teacher_votes > student_votes else "学生"
            maj_votes = max(teacher_votes, student_votes)
            if maj != cur_label and maj_votes >= DIARIZATION_SMOOTH_MIN_NEIGHBORS:
                labels[i] = maj
        for i, s in enumerate(segs):
            s["speaker"] = labels[i]

    def _voiceprint_relabel_inplace(segs, valid_indices, voiceprints, teacher_probs):
        if not DIARIZATION_VOICEPRINT_ASSIST_ENABLED:
            return 0
        if len(valid_indices) != len(voiceprints) or not valid_indices:
            return 0
        if len(teacher_probs) != len(valid_indices):
            return 0
        low = float(DIARIZATION_VOICEPRINT_BORDERLINE_LOW)
        high = float(DIARIZATION_VOICEPRINT_BORDERLINE_HIGH)
        if low > high:
            print(
                "  警告: DIARIZATION_VOICEPRINT_BORDERLINE_LOW 大于 HIGH，"
                "已自动交换边界值。"
            )
            low, high = high, low

        idx2vp = {vi: _l2_normalize(vp) for vi, vp in zip(valid_indices, voiceprints)}
        idx2prob = {vi: float(tp) for vi, tp in zip(valid_indices, teacher_probs)}
        teacher_candidate_idx = [
            vi for vi in valid_indices if segs[vi].get("speaker") == "教师"
        ]
        teacher_candidate_probs = [idx2prob.get(vi, 0.0) for vi in teacher_candidate_idx]
        proto_q = float(np.clip(DIARIZATION_VOICEPRINT_TEACHER_PROTO_QUANTILE, 0.05, 0.95))
        if teacher_candidate_probs:
            dynamic_high = float(np.quantile(teacher_candidate_probs, proto_q))
            proto_high = min(high, max(low, dynamic_high))
        else:
            proto_high = high
        teacher_vecs = [
            idx2vp[vi]
            for vi in teacher_candidate_idx
            if idx2prob.get(vi, 0.0) >= proto_high
        ]
        if len(teacher_vecs) < DIARIZATION_VOICEPRINT_MIN_TEACHER_SAMPLES and teacher_candidate_idx:
            ranked_teacher_idx = sorted(
                teacher_candidate_idx, key=lambda vi: idx2prob.get(vi, 0.0), reverse=True
            )
            top_n = min(len(ranked_teacher_idx), max(3, DIARIZATION_VOICEPRINT_MIN_TEACHER_SAMPLES))
            teacher_vecs = [idx2vp[vi] for vi in ranked_teacher_idx[:top_n]]
        if len(teacher_vecs) < DIARIZATION_VOICEPRINT_MIN_TEACHER_SAMPLES:
            return 0

        teacher_proto = _l2_normalize(np.mean(teacher_vecs, axis=0))
        student_vecs = [
            idx2vp[vi]
            for vi in valid_indices
            if segs[vi].get("speaker") == "学生"
            and idx2prob.get(vi, 1.0) <= low
        ]
        student_proto = _l2_normalize(np.mean(student_vecs, axis=0)) if student_vecs else None

        changed = 0
        for vi in valid_indices:
            if segs[vi].get("speaker") != "学生":
                continue
            teacher_prob = idx2prob.get(vi, 0.0)
            if teacher_prob < low or teacher_prob > high:
                continue
            seg_dur = float(segs[vi]["end"] - segs[vi]["start"])
            if seg_dur < DIARIZATION_VOICEPRINT_MIN_SEGMENT_DURATION:
                continue
            if DIARIZATION_VOICEPRINT_MAX_STUDENT_DURATION > 0 and seg_dur > DIARIZATION_VOICEPRINT_MAX_STUDENT_DURATION:
                continue

            vp = idx2vp[vi]
            sim_teacher = _cosine_sim(vp, teacher_proto)
            sim_student = _cosine_sim(vp, student_proto) if student_proto is not None else -1.0

            if sim_teacher >= DIARIZATION_VOICEPRINT_SIMILARITY_THRESHOLD and (
                student_proto is None or sim_teacher - sim_student >= DIARIZATION_VOICEPRINT_STUDENT_MARGIN
            ):
                segs[vi]["speaker"] = "教师"
                changed += 1
        return changed

    try:
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        feats, valid_idx = [], []
        durations = []
        acoustic_variance = []
        text_scores_raw = []
        voiceprints = []
        for i, seg in enumerate(segments):
            ss = int(seg["start"] * sr)
            es = int(seg["end"]   * sr)
            chunk = audio[ss:es]
            if len(chunk) < sr * 0.4:          # 太短则跳过
                continue
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
            rms = librosa.feature.rms(y=chunk)[0]
            centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
            delta = librosa.feature.delta(mfcc)
            contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y=chunk)[0]
            mfcc_std = float(np.mean(mfcc.std(axis=1)))
            rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-6))
            centroid_cv = float(np.std(centroid) / (np.mean(centroid) + 1e-6))
            var_score = mfcc_std + rms_cv + centroid_cv

            feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), [var_score]])
            voiceprint_feat = np.concatenate([
                mfcc.mean(axis=1),
                delta.mean(axis=1),
                contrast.mean(axis=1),
                [float(np.mean(zcr)), float(np.mean(rms))]
            ])
            feats.append(feat)
            voiceprints.append(voiceprint_feat)
            valid_idx.append(i)
            durations.append(float(seg["end"] - seg["start"]))
            acoustic_variance.append(var_score)
            text_scores_raw.append(_text_teacher_score(seg.get("text", "")))

        if len(valid_idx) < 2:
            for s in segments:
                s["speaker"] = "教师"
            return segments, ["教师", "学生"]

        X = StandardScaler().fit_transform(np.array(feats))

        n_spk = max(2, int(DIARIZATION_N_CLUSTERS))
        n_spk = min(n_spk, len(valid_idx))

        if n_spk == 1:
            labels = np.zeros(len(valid_idx), dtype=int)
        else:
            labels = AgglomerativeClustering(
                n_clusters=n_spk, metric="euclidean", linkage="ward"
            ).fit_predict(X)

        z_dur = _safe_zscore(durations)
        z_stable = -_safe_zscore(acoustic_variance)   # 越稳定越像教师
        z_text = _safe_zscore(text_scores_raw)
        teacher_scores = (
            DIARIZATION_ACOUSTIC_WEIGHT * (0.5 * z_dur + 0.5 * z_stable)
            + DIARIZATION_TEXT_WEIGHT * z_text
        )

        # 片段级概率：避免“整簇贴标签”导致同一位教师被拆成两簇时整簇误判。
        teacher_probs = _normalize_to_prob(teacher_scores)

        # idx → speaker name map（按片段概率判定，而非按簇整体判定）
        idx2spk = {}
        idx2prob = {}
        for vi, prob in zip(valid_idx, teacher_probs):
            p = float(prob)
            idx2prob[vi] = p
            idx2spk[vi] = "教师" if p >= DIARIZATION_TEACHER_PROB_THRESHOLD else "学生"

        for i, seg in enumerate(segments):
            if i in idx2spk:
                seg["speaker"] = idx2spk[i]
                seg["teacher_prob"] = round(idx2prob[i], 4)
            else:
                # 最近有效帧的说话人
                nearest = min(valid_idx, key=lambda j: abs(j - i))
                seg["speaker"] = idx2spk.get(nearest, "教师")
                seg["teacher_prob"] = round(float(idx2prob.get(nearest, 0.5)), 4)

        _smooth_labels_inplace(segments)
        relabeled = _voiceprint_relabel_inplace(segments, valid_idx, voiceprints, teacher_probs)
        if relabeled > 0:
            print(f"  声纹边界复判: 学生→教师 {relabeled} 段")
            _smooth_labels_inplace(segments)

        speakers = ["教师", "学生"]
        print(f"  说话人: {speakers}")
        return segments, speakers

    except Exception as e:
        print(f"  说话人区分出错（{e}），全部标为「教师」")
        for s in segments:
            s["speaker"] = "教师"
        return segments, ["教师", "学生"]


# ============================================================
# 核心分析
# ============================================================

def analyze_video_audio(video_path, output_dir, video_name):
    print(f"\n[步骤2] 语音分析: {video_name}")

    if not check_ffmpeg():
        raise RuntimeError(
            "未找到 ffmpeg！\n"
            "Windows 安装：\n"
            "  1. 下载 https://www.gyan.dev/ffmpeg/builds/\n"
            "  2. 解压到 C:\\ffmpeg\n"
            "  3. 将 C:\\ffmpeg\\bin 加入系统 PATH"
        )

    tmp_dir  = tempfile.mkdtemp()
    wav_path = os.path.join(tmp_dir, f"{video_name}.wav")

    try:
        extract_audio(video_path, wav_path)

        segments, info = transcribe(wav_path)
        segments, script_stats = normalize_transcription_script(segments)
        segments, correction_stats = correct_transcription_with_pinyin_fuzzy(segments)

        valid_segs = [s for s in segments if _should_keep_speech_segment(s)]
        if valid_segs:
            segments, speakers = diarize_speakers(wav_path, segments)
        else:
            for s in segments:
                s["speaker"] = "教师"
            speakers = ["教师"]

        dur   = float(getattr(info, "duration", 0) or 0)
        if dur == 0 and segments:
            dur = max(s["end"] for s in segments)

        avg_conf = float(np.mean([s["confidence"] for s in segments])) if segments else 0.0

        result = {
            "video_name":          video_name,
            "video_path":          str(video_path),
            "language":            info.language,
            "language_probability": round(float(info.language_probability), 4),
            "total_duration":      round(dur, 2),
            "segments":            segments,
            "speakers":            speakers,
            "stats": {
                "total_segments":       len(segments),
                "valid_segments":       len(valid_segs),
                "total_speech_duration": round(
                    sum(s["end"] - s["start"] for s in valid_segs), 2),
                "avg_confidence":       round(avg_conf, 4),
                "low_confidence_count": sum(1 for s in segments if s["is_low_confidence"]),
                "script_conversion":    script_stats,
                "text_correction":      correction_stats,
            },
        }

        out_file = os.path.join(output_dir, "audio_features.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n  ✓ 语音分析完成")
        print(f"    有效片段: {len(valid_segs)}")
        print(f"    平均置信度: {avg_conf:.2f}")
        print(f"    说话人: {speakers}")
        print(f"    输出: {out_file}")
        return result

    finally:
        for f in [wav_path]:
            try:
                os.remove(f)
            except Exception:
                pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="步骤2: 语音维度分析")
    parser.add_argument("--video",  required=True, help="视频文件路径")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出根目录")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频不存在: {args.video}")
        sys.exit(1)

    out_dir, vname = get_output_dir(args.video, args.output)
    analyze_video_audio(args.video, out_dir, vname)


if __name__ == "__main__":
    main()
