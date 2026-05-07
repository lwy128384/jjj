#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤3: 文本语义分析
  1. 中文分词（jieba）
  2. TF-IDF 向量化
  3. 滑动窗口语义相似度 → 知识点边界检测
  4. 关键词提取与知识点命名

运行方式:
  python step3_text.py --video D:/video/lesson/example.mp4

输出:
  D:/video/output/example/text_features.json
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
# 默认参数
# ============================================================
try:
    import config as _cfg
    OUTPUT_DIR             = _cfg.OUTPUT_DIR
    SEMANTIC_WINDOW_SIZE   = _cfg.SEMANTIC_WINDOW_SIZE
    BOUNDARY_THRESHOLD     = _cfg.BOUNDARY_THRESHOLD
    MIN_KNOWLEDGE_DURATION = _cfg.MIN_KNOWLEDGE_DURATION
    MAX_KNOWLEDGE_DURATION = _cfg.MAX_KNOWLEDGE_DURATION
    TOP_KEYWORDS           = _cfg.TOP_KEYWORDS
    MIN_TEXT_LENGTH        = _cfg.MIN_TEXT_LENGTH
    NO_SPEECH_PROB_THRESHOLD = _cfg.NO_SPEECH_PROB_THRESHOLD
except ImportError:
    OUTPUT_DIR             = r"D:\video\output"
    SEMANTIC_WINDOW_SIZE   = 3
    BOUNDARY_THRESHOLD     = 0.35
    MIN_KNOWLEDGE_DURATION = 45
    MAX_KNOWLEDGE_DURATION = 600
    TOP_KEYWORDS           = 5
    MIN_TEXT_LENGTH        = 5
    NO_SPEECH_PROB_THRESHOLD = 0.50

KNOWLEDGE_TITLE_PREFIX = "知识点"
MODEL_PREDICT_EXCEPTIONS = (
    FileNotFoundError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
)
TIME_EPSILON = 1e-6


# ============================================================
# 工具函数
# ============================================================

def get_output_dir(video_path, base_output_dir=None):
    base = base_output_dir or OUTPUT_DIR
    name = Path(video_path).stem
    out  = os.path.join(base, name)
    os.makedirs(out, exist_ok=True)
    return out, name


def load_audio_features(output_dir):
    p = os.path.join(output_dir, "audio_features.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"请先运行步骤2，找不到: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_visual_features(output_dir):
    p = os.path.join(output_dir, "visual_features.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# NLP 工具
# ============================================================

STOP_WORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "那", "里", "而", "啊", "吧", "呢",
    "嗯", "啦", "对", "所以", "因为", "如果", "但是", "然后", "然而",
    "因此", "这个", "那个", "我们", "他们", "它们", "什么", "怎么",
    "哪里", "哪个", "这里", "那里", "可以", "应该", "需要", "一些",
    "很多", "非常", "比较", "可能", "已经", "通过", "进行", "使用",
    "方面", "方法", "问题", "情况", "内容", "过程", "就是", "其实",
    "现在", "时候", "地方", "东西", "感觉", "知道", "觉得", "认为",
}


def init_jieba():
    try:
        import jieba
        import jieba.analyse
        import logging
        logging.getLogger("jieba").setLevel(logging.ERROR)
        return jieba
    except ImportError:
        raise RuntimeError("请安装 jieba:\n  pip install jieba")


def tokenize(text, jieba):
    words = jieba.cut(text)
    return [w for w in words
            if len(w) > 1 and w not in STOP_WORDS and not w.isdigit()]


def extract_keywords(text, jieba, top_n=5):
    if not text.strip():
        return []
    try:
        return list(jieba.analyse.extract_tags(text, topK=top_n))
    except Exception:
        words = tokenize(text, jieba)
        return [w for w, _ in Counter(words).most_common(top_n)]


# ============================================================
# TF-IDF 向量化
# ============================================================

def build_tfidf(texts, jieba):
    """返回 L2 归一化的 TF-IDF 矩阵和词汇表"""
    tokenized = [tokenize(t, jieba) for t in texts]
    vocab     = sorted({w for toks in tokenized for w in toks})
    if not vocab:
        return np.zeros((len(texts), 1)), vocab

    w2i = {w: i for i, w in enumerate(vocab)}
    n, v = len(tokenized), len(vocab)

    tf = np.zeros((n, v))
    for i, toks in enumerate(tokenized):
        if not toks:
            continue
        cnt = Counter(toks)
        for w, c in cnt.items():
            if w in w2i:
                tf[i, w2i[w]] = c / len(toks)

    df  = np.sum(tf > 0, axis=0)
    idf = np.log((n + 1) / (df + 1)) + 1
    mat = tf * idf

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms, vocab


def cosine_distance(u, v):
    """余弦距离（均已 L2 归一化，直接用点积）"""
    return float(1.0 - np.dot(u, v))


def normalize_boundaries(bounds, t_start, t_end):
    """
    Normalize boundary timestamps.

    Args:
        bounds: Candidate boundary timestamps (seconds).
        t_start: Start timestamp (seconds) for the analyzed range.
        t_end: End timestamp (seconds) for the analyzed range.

    Returns:
        A sorted boundary timestamp list after short-interval filtering and
        long-interval splitting.
    """
    filtered = [round(t_start, 2)]
    for t in sorted(bounds):
        if abs(t - filtered[-1]) <= TIME_EPSILON:
            continue
        dt = t - filtered[-1]
        if dt < MIN_KNOWLEDGE_DURATION:
            continue
        if dt > MAX_KNOWLEDGE_DURATION:
            n = int(dt / MAX_KNOWLEDGE_DURATION)
            step = dt / (n + 1)
            for _ in range(1, n + 1):
                filtered.append(round(filtered[-1] + step, 2))
        filtered.append(round(t, 2))
    if abs(filtered[-1] - round(t_end, 2)) >= TIME_EPSILON:
        filtered.append(round(t_end, 2))
    return filtered


def try_refine_boundaries_with_model(filtered, valid, visual_features):
    """
    Refine semantic boundaries with the trained boundary model when available.
    Falls back to rule-based boundaries when model inference is unavailable.

    Args:
        filtered: Rule-based normalized boundaries.
        valid: Valid ASR segments used for boundary detection.
        visual_features: Optional visual feature dictionary.

    Returns:
        Refined boundary list when model predictions are usable; otherwise the
        original rule-based boundary list.
    """
    try:
        from train import predict_boundaries
        from step4_align import build_timeline
    except ImportError:
        return filtered

    if not valid:
        return filtered

    duration = max(s["end"] for s in valid)
    if visual_features:
        duration = max(duration, float(visual_features.get("duration", 0.0)))

    pseudo_text = {
        "knowledge_segments": [
            {"id": i, "start": s, "end": e, "title": f"{KNOWLEDGE_TITLE_PREFIX}{i+1}"}
            for i, (s, e) in enumerate(zip(filtered[:-1], filtered[1:]))
        ],
        "boundaries": filtered,
    }
    pseudo_audio = {"segments": valid}

    timeline = build_timeline(visual_features, pseudo_audio, pseudo_text, duration)
    try:
        model_times = predict_boundaries({"time_series": timeline})
    except MODEL_PREDICT_EXCEPTIONS as e:
        print(f"  训练边界模型调用失败，回退规则边界: {e}")
        return filtered
    if not model_times:
        return filtered

    t_start = valid[0]["start"]
    t_end = valid[-1]["end"]
    model_bounds = [
        round(t, 2)
        for t in model_times
        if t_start + TIME_EPSILON < t < t_end - TIME_EPSILON
    ]
    if not model_bounds:
        return filtered

    merged_bounds = sorted(set(filtered + model_bounds + [t_start, t_end]))
    refined = normalize_boundaries(merged_bounds, t_start, t_end)
    if len(refined) >= 2:
        print(f"  已应用训练边界模型（新增候选边界 {len(model_bounds)} 个）")
        return refined
    return filtered


# ============================================================
# 边界检测
# ============================================================

def detect_boundaries(segments, jieba, visual_features=None):
    """
    滑动窗口语义边界检测，结合幻灯片切换辅助信号。
    返回知识点段列表。
    """
    valid = [s for s in segments
             if len(s.get("text", "").strip()) >= MIN_TEXT_LENGTH
             and s.get("no_speech_prob", 1.0) < NO_SPEECH_PROB_THRESHOLD]

    if len(valid) < SEMANTIC_WINDOW_SIZE * 2:
        print("  有效语音段不足，整个视频作为一个知识点")
        start = segments[0]["start"] if segments else 0.0
        end   = segments[-1]["end"]  if segments else 0.0
        kws   = extract_keywords(" ".join(s.get("text","") for s in segments), jieba)
        return [{
            "id": 0, "start": start, "end": end,
            "duration": end - start,
            "title": f"知识点1-{'_'.join(kws[:2])}" if kws else "知识点1",
            "keywords": kws, "text_preview": "",
        }]

    texts = [s["text"] for s in valid]
    times = [(s["start"], s["end"]) for s in valid]

    print(f"  TF-IDF 向量化 ({len(texts)} 段)…")
    mat, _ = build_tfidf(texts, jieba)

    W = SEMANTIC_WINDOW_SIZE
    scores = []
    for i in range(W, len(valid) - W):
        L = np.mean(mat[max(0, i-W):i], axis=0)
        R = np.mean(mat[i:min(len(valid), i+W)], axis=0)
        ln = np.linalg.norm(L); rn = np.linalg.norm(R)
        if ln > 0: L /= ln
        if rn > 0: R /= rn
        scores.append({"time": times[i][0], "score": cosine_distance(L, R)})

    # 幻灯片切换辅助候选
    slide_times = set()
    if visual_features:
        for st in visual_features.get("slide_transitions", []):
            slide_times.add(st["time"])

    candidates = []
    for item in scores:
        if item["score"] > BOUNDARY_THRESHOLD:
            candidates.append({"time": item["time"], "score": item["score"],
                                "source": "semantic"})
    for t in slide_times:
        candidates.append({"time": t, "score": 0.50, "source": "slide"})

    # 按时间排序后合并相近候选（间距 < MIN/2）
    candidates.sort(key=lambda x: x["time"])
    merged = []
    for cb in candidates:
        if not merged or cb["time"] - merged[-1]["time"] > MIN_KNOWLEDGE_DURATION / 2:
            merged.append(cb)
        elif cb["score"] > merged[-1]["score"]:
            merged[-1] = cb

    # 构建边界列表
    t_start = valid[0]["start"]
    t_end   = valid[-1]["end"]
    bounds  = [t_start] + [m["time"] for m in merged] + [t_end]

    filtered = normalize_boundaries(bounds, t_start, t_end)
    filtered = try_refine_boundaries_with_model(filtered, valid, visual_features)

    # 生成知识点段
    knowledge_segs = []
    for idx, (seg_s, seg_e) in enumerate(zip(filtered[:-1], filtered[1:])):
        seg_texts = [s["text"] for s in valid
                     if s["start"] >= seg_s and s["end"] <= seg_e]
        combined = " ".join(seg_texts)
        kws  = extract_keywords(combined, jieba, TOP_KEYWORDS)
        title = f"知识点{idx+1}"
        if kws:
            title = f"知识点{idx+1}-{'_'.join(kws[:2])}"

        knowledge_segs.append({
            "id":           idx,
            "start":        round(seg_s, 2),
            "end":          round(seg_e, 2),
            "duration":     round(seg_e - seg_s, 2),
            "title":        title,
            "keywords":     kws,
            "text_preview": combined[:120],
        })

    return knowledge_segs


# ============================================================
# 核心分析
# ============================================================

def analyze_text(video_path, output_dir, video_name):
    print(f"\n[步骤3] 文本语义分析: {video_name}")

    audio   = load_audio_features(output_dir)
    visual  = load_visual_features(output_dir)
    if visual:
        print("  已加载视觉特征（幻灯片切换辅助）")

    jieba = init_jieba()
    print(f"  加载 {len(audio['segments'])} 个语音段")

    segs = detect_boundaries(audio["segments"], jieba, visual)

    bounds = [s["start"] for s in segs]
    if segs:
        bounds.append(segs[-1]["end"])

    result = {
        "video_name":        video_name,
        "video_path":        str(video_path),
        "total_audio_segs":  len(audio["segments"]),
        "knowledge_segments": segs,
        "boundaries":        [round(b, 2) for b in bounds],
        "stats": {
            "total_knowledge_points": len(segs),
            "avg_duration":  round(
                sum(s["duration"] for s in segs) / max(len(segs), 1), 2),
            "total_duration": round(sum(s["duration"] for s in segs), 2),
        },
    }

    out_file = os.path.join(output_dir, "text_features.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ 文本分析完成")
    print(f"    知识点数: {len(segs)}")
    for s in segs:
        print(f"      [{s['start']:6.0f}s – {s['end']:6.0f}s]  {s['title']}")
    print(f"    输出: {out_file}")
    return result


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="步骤3: 文本语义分析")
    parser.add_argument("--video",  required=True, help="视频文件路径")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出根目录")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 视频不存在: {args.video}")
        sys.exit(1)

    out_dir, vname = get_output_dir(args.video, args.output)
    analyze_text(args.video, out_dir, vname)


if __name__ == "__main__":
    main()
