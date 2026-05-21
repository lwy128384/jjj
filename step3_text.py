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
import re
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
    NO_SPEECH_IGNORE_WITH_TEXT = _cfg.NO_SPEECH_IGNORE_WITH_TEXT
    NO_SPEECH_TEXT_SHORT_LEN = _cfg.NO_SPEECH_TEXT_SHORT_LEN
    KEYWORD_TITLE_COUNT    = _cfg.KEYWORD_TITLE_COUNT
    KEYWORD_MIN_DOC_FREQ   = _cfg.KEYWORD_MIN_DOC_FREQ
    KEYWORD_BLACKLIST      = set(_cfg.KEYWORD_BLACKLIST)
    STEP3_DOMAIN_TERMS     = list(_cfg.STEP3_DOMAIN_TERMS)
    STEP3_ENABLE_TEXT_NORMALIZATION = _cfg.STEP3_ENABLE_TEXT_NORMALIZATION
    STEP3_TEXT_REPLACE_MAP = dict(_cfg.STEP3_TEXT_REPLACE_MAP)
except ImportError:
    OUTPUT_DIR             = r"D:\video\output"
    SEMANTIC_WINDOW_SIZE   = 3
    BOUNDARY_THRESHOLD     = 0.35
    MIN_KNOWLEDGE_DURATION = 45
    MAX_KNOWLEDGE_DURATION = 600
    TOP_KEYWORDS           = 5
    MIN_TEXT_LENGTH        = 5
    NO_SPEECH_PROB_THRESHOLD = 0.80
    NO_SPEECH_IGNORE_WITH_TEXT = True
    NO_SPEECH_TEXT_SHORT_LEN = 3
    KEYWORD_TITLE_COUNT    = 2
    KEYWORD_MIN_DOC_FREQ   = 2
    KEYWORD_BLACKLIST      = {
        "这个", "那个", "就是", "然后", "所以", "我们", "你们", "他们",
        "可以", "应该", "需要", "东西", "问题", "内容", "方面", "方法",
    }
    STEP3_DOMAIN_TERMS     = [
        "人工智能", "机器学习", "深度学习", "神经网络", "图灵测试", "亚里士多德",
    ]
    STEP3_ENABLE_TEXT_NORMALIZATION = True
    STEP3_TEXT_REPLACE_MAP = {
        "人工质能": "人工智能",
        "运遇": "机遇",
        "亚丽师多德": "亚里士多德",
        "低谱": "低谷",
        "图林": "图灵",
        "這個": "这个",
        "那個": "那个",
        "一個": "一个",
        "我們": "我们",
        "你們": "你们",
        "他們": "他们",
        "什麼": "什么",
        "怎麼": "怎么",
        "為什麼": "为什么",
        "哪裡": "哪里",
        "進行": "进行",
        "發生": "发生",
        "紅顏色": "红颜色",
        "規定": "规定",
    }

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
    "這個", "那個", "一個", "我們", "你們", "他們", "什麼", "怎麼", "為什麼", "哪裡",
}


def normalize_text(text):
    if not text:
        return ""
    out = str(text)
    if STEP3_ENABLE_TEXT_NORMALIZATION and STEP3_TEXT_REPLACE_MAP:
        for src in sorted(STEP3_TEXT_REPLACE_MAP, key=len, reverse=True):
            dst = STEP3_TEXT_REPLACE_MAP[src]
            if src:
                out = out.replace(src, dst)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _is_semantic_valid_segment(seg):
    normalized_text = normalize_text(seg.get("text", ""))
    if len(normalized_text) < MIN_TEXT_LENGTH:
        return False
    no_sp = float(seg.get("no_speech_prob", 1.0) or 1.0)
    if no_sp <= NO_SPEECH_PROB_THRESHOLD:
        return True
    if not (NO_SPEECH_IGNORE_WITH_TEXT and normalized_text):
        return False
    return len(normalized_text) > NO_SPEECH_TEXT_SHORT_LEN


def init_jieba():
    try:
        import jieba
        import logging
        logging.getLogger("jieba").setLevel(logging.ERROR)
        for term in STEP3_DOMAIN_TERMS:
            if term:
                jieba.add_word(term)
        return jieba
    except ImportError:
        raise RuntimeError("请安装 jieba:\n  pip install jieba")


def is_valid_token(token):
    if not token:
        return False
    if len(token) <= 1:
        return False
    if token.isdigit():
        return False
    if token in STOP_WORDS or token in KEYWORD_BLACKLIST:
        return False
    if re.fullmatch(r"[\W_]+", token):
        return False
    return True


def tokenize(text, jieba):
    norm = normalize_text(text)
    words = jieba.cut(norm)
    return [w for w in words if is_valid_token(w)]


def build_knowledge_title(keywords, idx):
    """知识点命名：优先关键词，不再添加“知识点N-”前缀。"""
    kws = []
    for k in (keywords or []):
        token = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", str(k).strip())
        if token:
            kws.append(token)
    if kws:
        return "_".join(kws[:KEYWORD_TITLE_COUNT])
    return f"片段{idx+1}"


def build_idf_stats(texts, jieba):
    n = max(len(texts), 1)
    doc_freq = Counter()
    for t in texts:
        toks = set(tokenize(t, jieba))
        for tok in toks:
            doc_freq[tok] += 1
    idf_map = {w: np.log((n + 1) / (df + 1)) + 1.0 for w, df in doc_freq.items()}
    return idf_map, doc_freq


def extract_keywords(text, jieba, top_n=5, idf_map=None, doc_freq=None):
    norm = normalize_text(text)
    if not norm:
        return []
    words = tokenize(norm, jieba)
    if not words:
        return []

    tf = Counter(words)
    total = len(words)
    scores = {}
    for w, c in tf.items():
        if doc_freq and doc_freq.get(w, 0) < KEYWORD_MIN_DOC_FREQ:
            continue
        score = c / total
        if idf_map:
            score *= idf_map.get(w, 1.0)
        scores[w] = score

    if not scores:
        # 文档频次过滤后为空时回退，避免标题缺失
        scores = {w: c / total for w, c in tf.items()}

    ranked = sorted(scores.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [w for w, _ in ranked[:top_n]]


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
        # Treat near-equal timestamps as duplicates under numeric tolerance.
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
    # Ensure end boundary exists unless it is already present within tolerance.
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
            {"id": i, "start": s, "end": e, "title": build_knowledge_title([], i)}
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
    normalized_segments = []
    for s in segments:
        cp = dict(s)
        cp["text"] = normalize_text(s.get("text", ""))
        normalized_segments.append(cp)

    valid = [s for s in normalized_segments if _is_semantic_valid_segment(s)]

    if len(valid) < SEMANTIC_WINDOW_SIZE * 2:
        print("  有效语音段不足，整个视频作为一个知识点")
        start = segments[0]["start"] if segments else 0.0
        end   = segments[-1]["end"]  if segments else 0.0
        kws   = extract_keywords(" ".join(s.get("text", "") for s in normalized_segments), jieba)
        return [{
            "id": 0, "start": start, "end": end,
            "duration": end - start,
            "title": build_knowledge_title(kws, 0),
            "keywords": kws, "text_preview": "",
        }]

    texts = [s["text"] for s in valid]
    times = [(s["start"], s["end"]) for s in valid]
    idf_map, doc_freq = build_idf_stats(texts, jieba)

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
        kws  = extract_keywords(combined, jieba, TOP_KEYWORDS, idf_map=idf_map, doc_freq=doc_freq)
        title = build_knowledge_title(kws, idx)

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
