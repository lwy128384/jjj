#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练模块 — 利用人工标注数据提升知识点边界检测准确率

标注文件格式 (annotation.json):
  {
    "video": "lesson1.mp4",
    "annotations": [
      {"start": "0:00:00", "end": "0:03:00", "title": "第一章绪论",  "is_interference": false},
      {"start": "0:03:00", "end": "0:03:30", "title": "课间休息",    "is_interference": true},
      {"start": "0:03:30", "end": "0:06:40", "title": "第一节基本概念","is_interference": false}
    ]
  }

用法（训练）:
  python train.py --video D:/video/lesson/example.mp4 \
                  --annotation D:/video/annotations/example_annotation.json

用法（批量训练多个视频）:
  python train.py --annotation_dir D:/video/annotations/

用法（评估已有模型）:
  python train.py --eval --video D:/video/lesson/example.mp4 \
                  --annotation D:/video/annotations/example_annotation.json
"""

import os
import sys
import json
import argparse
import traceback
import glob
import re
import math
import numpy as np
from pathlib import Path

# ============================================================
# 默认参数
# ============================================================
try:
    import config as _cfg
    LESSON_DIR       = _cfg.LESSON_DIR
    OUTPUT_DIR       = _cfg.OUTPUT_DIR
    MODELS_DIR       = _cfg.MODELS_DIR
    TRAIN_MODEL_FILE = _cfg.TRAIN_MODEL_FILE
    TRAIN_TEST_SPLIT = _cfg.TRAIN_TEST_SPLIT
    RANDOM_STATE     = _cfg.RANDOM_STATE
    TIME_RESOLUTION  = _cfg.TIME_RESOLUTION
    BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER = _cfg.BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER
    INTERFERENCE_POSITIVE_WEIGHT_MULTIPLIER = _cfg.INTERFERENCE_POSITIVE_WEIGHT_MULTIPLIER
    INTERFERENCE_NEG_POS_RATIO = _cfg.INTERFERENCE_NEG_POS_RATIO
    BOUNDARY_MODEL_BASE_THRESHOLD = _cfg.BOUNDARY_MODEL_BASE_THRESHOLD
    INTERFERENCE_MODEL_THRESHOLD = _cfg.INTERFERENCE_MODEL_THRESHOLD
    BOUNDARY_POST_MIN_GAP = _cfg.BOUNDARY_POST_MIN_GAP
    BOUNDARY_POST_MAX_GAP = _cfg.BOUNDARY_POST_MAX_GAP
except ImportError:
    LESSON_DIR       = r"D:\video\lesson"
    OUTPUT_DIR       = r"D:\video\output"
    MODELS_DIR       = r"D:\video\models"
    TRAIN_MODEL_FILE = "boundary_model.pkl"
    TRAIN_TEST_SPLIT = 0.20
    RANDOM_STATE     = 42
    TIME_RESOLUTION  = 1.0
    BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER = 3.0
    INTERFERENCE_POSITIVE_WEIGHT_MULTIPLIER = 3.0
    INTERFERENCE_NEG_POS_RATIO = 3
    BOUNDARY_MODEL_BASE_THRESHOLD = 0.50
    INTERFERENCE_MODEL_THRESHOLD = 0.50
    BOUNDARY_POST_MIN_GAP = 30.0
    BOUNDARY_POST_MAX_GAP = 300.0

MIN_BOUNDARY_DYNAMIC_THRESHOLD = 0.35
MAX_BOUNDARY_DYNAMIC_THRESHOLD = 0.75


# ============================================================
# 工具函数
# ============================================================

def get_output_dir(video_path):
    name = Path(video_path).stem
    out  = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    return out, name


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        if default is None:
            return None
        return float(default)


def _parse_time_seconds(v, default=0.0):
    if isinstance(v, str):
        text = v.strip()
        if re.fullmatch(r"\d+:[0-5]\d:[0-5]\d", text):
            h, m, s = text.split(":")
            return float(int(h) * 3600 + int(m) * 60 + int(s))
        return _safe_float(text, default)
    return _safe_float(v, default)


def _merge_intervals(intervals):
    valid = []
    for s, e in intervals:
        s = _safe_float(s, 0.0)
        e = _safe_float(e, 0.0)
        if e < s:
            s, e = e, s
        if e - s <= 1e-6:
            continue
        valid.append((round(s, 2), round(e, 2)))
    if not valid:
        return []
    valid.sort(key=lambda x: (x[0], x[1]))
    merged = [list(valid[0])]
    for s, e in valid[1:]:
        if s <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _in_any_interval(t, intervals):
    for s, e in intervals:
        if s <= t <= e:
            return True
    return False


def _combine_point_text(pt):
    return f"{pt.get('speech_text', '') or ''} {pt.get('ppt_text', '') or ''}".strip()


def normalize_annotations(annotations):
    """
    统一标注结构，兼容“仅知识点标注”场景：
    - 只标知识点：其余时间自动视为干扰候选
    - 显式标注 is_interference 时优先保留
    """
    normalized = []
    for ann in annotations or []:
        s = _parse_time_seconds(ann.get("start"), None)
        e = _parse_time_seconds(ann.get("end"), None)
        if s is None or e is None:
            continue
        if e < s:
            s, e = e, s
        s = math.floor(s)
        e = math.ceil(e)
        if e - s <= 1e-6:
            continue
        is_intf = bool(ann.get("is_interference", False))
        title = str(ann.get("title", "") or "").strip()
        normalized.append({
            "start": s,
            "end": e,
            "title": title,
            "is_interference": is_intf,
        })
    return normalized


# ============================================================
# 特征提取（基于已有的 multimodal_index.json）
# ============================================================

def extract_features_from_index(multimodal_index, window_sec=10):
    """
    从多模态索引中提取每个时间点的特征向量。

    特征维度（每个时间点）:
      0  teacher_present         (0/1)
      1  motion_ratio            (0~1)
      2  is_slide_transition     (0/1)
      3  speech_confidence       (0~1)
      4  is_silence              (0/1)
      5  prev_teacher_present    (滑动窗口前值)
      6  prev_speech_confidence  (滑动窗口前值)
      7  slide_delta             (前后幻灯片差异，0/1)
      8  rolling_silence_ratio   (前 window_sec 秒静默比例)
      9  rolling_teacher_ratio   (前 window_sec 秒教师在场比例)

    标签:
      boundary  : 该时刻是知识点边界 (0/1)
    """
    series = multimodal_index["time_series"]
    n      = len(series)
    W      = window_sec

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

    feats = []
    for i, pt in enumerate(series):
        # 滚动窗口统计
        win = series[max(0, i-W):i+1]
        win_future = series[i:min(n, i+W+1)]
        roll_sil  = sum(1 for p in win if p["is_silence"])   / max(len(win), 1)
        roll_tchr = sum(1 for p in win if p["teacher_present"]) / max(len(win), 1)

        # 前一时刻特征
        prev = series[i-1] if i > 0 else pt
        # 幻灯片跳变
        slide_delta = int(pt["slide_idx"] != prev.get("slide_idx", 0))

        text_now = _combine_point_text(pt)
        text_prev = _combine_point_text(prev)
        text_sim = _jaccard_similarity(text_now, text_prev)

        dur_now = max(_safe_float(pt.get("duration", TIME_RESOLUTION), TIME_RESOLUTION), 1e-3)
        dur_prev = max(_safe_float(prev.get("duration", TIME_RESOLUTION), TIME_RESOLUTION), 1e-3)
        speech_rate_now = len(pt.get("speech_text", "") or "") / dur_now
        speech_rate_prev = len(prev.get("speech_text", "") or "") / dur_prev
        rate_change = abs(speech_rate_now - speech_rate_prev)

        speaker_change = int((pt.get("speaker", "") or "") != (prev.get("speaker", "") or ""))
        roll_sil_future = sum(1 for p in win_future if p["is_silence"]) / max(len(win_future), 1)

        f = [
            int(pt["teacher_present"]),
            pt["motion_ratio"],
            int(pt["is_slide_transition"]),
            pt["speech_confidence"],
            int(pt["is_silence"]),
            int(prev["teacher_present"]),
            prev["speech_confidence"],
            slide_delta,
            roll_sil,
            roll_tchr,
            text_sim,
            rate_change,
            speaker_change,
            roll_sil_future,
        ]
        feats.append(f)

    return np.array(feats, dtype=np.float32)


def label_from_annotation(series, annotations, tolerance_sec=3.0):
    """
    根据标注生成标签数组。
    返回两个数组:
      boundary_labels     : 是否知识点边界 (0/1)
      interference_labels : 是否干扰片段  (0/1)
    """
    times = [pt["time"] for pt in series]
    n     = len(times)

    bound_label  = np.zeros(n, dtype=np.int32)
    interf_label = np.zeros(n, dtype=np.int32)

    normalized = normalize_annotations(annotations)
    knowledge_ranges = _merge_intervals(
        [(a["start"], a["end"]) for a in normalized if not a["is_interference"]]
    )
    explicit_interference_ranges = _merge_intervals(
        [(a["start"], a["end"]) for a in normalized if a["is_interference"]]
    )

    # 边界：知识点片段起止点（兼容只标知识点）
    boundary_points = []
    for ann in normalized:
        if ann["is_interference"]:
            continue
        boundary_points.extend([ann["start"], ann["end"]])
    boundary_points = sorted(set(round(t, 2) for t in boundary_points))

    for b_t in boundary_points:
        for i, t in enumerate(times):
            if abs(t - b_t) <= tolerance_sec:
                bound_label[i] = 1

    # 干扰标签：
    # 1) 显式标注 is_interference=true 的区间
    # 2) 若存在知识点标注，则知识点外时间自动视为干扰（补集）
    for i, t in enumerate(times):
        is_explicit_interference = _in_any_interval(t, explicit_interference_ranges)
        is_in_knowledge = _in_any_interval(t, knowledge_ranges)
        inferred_interference = bool(knowledge_ranges) and not is_in_knowledge
        if is_explicit_interference or inferred_interference:
            interf_label[i] = 1

    return bound_label, interf_label


# ============================================================
# 模型训练
# ============================================================

def balance_dataset(X, y, neg_pos_ratio=3):
    """下采样负样本，缓解极端类别不平衡。"""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y

    n_pos = len(pos_idx)
    # 防御式兜底：即使配置误传 0/负数，也保证最小可用采样比例为 1。
    safe_ratio = int(neg_pos_ratio) if int(neg_pos_ratio) > 0 else 1
    n_neg = min(len(neg_idx), n_pos * safe_ratio)
    sampled_neg = np.random.RandomState(RANDOM_STATE).choice(neg_idx, n_neg, replace=False)
    all_idx = np.concatenate([pos_idx, sampled_neg])
    np.random.RandomState(RANDOM_STATE).shuffle(all_idx)
    return X[all_idx], y[all_idx]


def train_model(X, y, model_type="boundary"):
    """
    使用 RandomForest 训练分类器。
    Returns: trained model, metrics dict
    """
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, f1_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError:
        raise RuntimeError("请安装 scikit-learn:\n  pip install scikit-learn")

    if model_type == "interference":
        X, y = balance_dataset(X, y, neg_pos_ratio=INTERFERENCE_NEG_POS_RATIO)

    classes = np.unique(y)
    if len(classes) < 2:
        raise RuntimeError(f"{model_type} 训练集仅含单一类别，无法训练分类器")

    class_weight_values = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weight_values)}
    if 1 in class_weights:
        if model_type == "boundary":
            class_weights[1] *= float(BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER)
        else:
            class_weights[1] *= float(INTERFERENCE_POSITIVE_WEIGHT_MULTIPLIER)

    def _can_stratify(labels):
        vals, counts = np.unique(labels, return_counts=True)
        return len(vals) > 1 and counts.min() >= 2

    stratify_y = y if _can_stratify(y) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=stratify_y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators  = 200,
        max_depth     = 8,
        class_weight  = class_weights,
        random_state  = RANDOM_STATE,
        n_jobs        = -1,       # 多核 CPU
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    report = classification_report(y_test, y_pred, zero_division=0)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  [{model_type} 模型] 评估报告:")
    print(report)
    print(f"  加权 F1: {f1:.4f}")
    print(f"  类别权重: {class_weights}")

    return clf, scaler, {"f1": round(f1, 4), "report": report}


def save_model(clf, scaler, meta, model_type):
    """保存模型到 MODELS_DIR"""
    try:
        import pickle
        os.makedirs(MODELS_DIR, exist_ok=True)
        fname = f"{model_type}_model.pkl"
        path  = os.path.join(MODELS_DIR, fname)
        with open(path, "wb") as f:
            pickle.dump({"clf": clf, "scaler": scaler, "meta": meta}, f)
        print(f"  模型已保存: {path}")
        return path
    except Exception as e:
        print(f"  模型保存失败: {e}")
        return None


def load_model(model_type):
    """从 MODELS_DIR 加载模型"""
    import pickle
    path = os.path.join(MODELS_DIR, f"{model_type}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["clf"], obj["scaler"]


# ============================================================
# 完整训练流程
# ============================================================

def train_on_video(video_path, annotation_path):
    """
    1. 确保步骤1-4已运行（若 multimodal_index.json 存在则直接使用）
    2. 提取特征
    3. 从标注生成标签
    4. 训练模型
    """
    out_dir, vname = get_output_dir(video_path)
    idx_path = os.path.join(out_dir, "multimodal_index.json")

    # 若索引不存在，先运行步骤1-4
    if not os.path.exists(idx_path):
        print(f"  multimodal_index.json 不存在，将自动运行步骤1-4…")
        from run_all import run_pipeline
        results = run_pipeline(video_path, start_step=1, end_step=4)
        for s, r in results.items():
            if r["status"] != "ok":
                raise RuntimeError(f"步骤{s}失败，无法提取特征")

    print(f"  加载多模态索引: {idx_path}")
    mindex = load_json(idx_path)
    ann    = load_json(annotation_path)

    annotations = ann.get("annotations", [])
    print(f"  标注片段数: {len(annotations)}")

    series = mindex["time_series"]
    X      = extract_features_from_index(mindex)
    b_lbl, i_lbl = label_from_annotation(series, annotations)

    print(f"  特征矩阵: {X.shape}  边界正样本: {b_lbl.sum()}  干扰正样本: {i_lbl.sum()}")
    return X, b_lbl, i_lbl, vname


def aggregate_and_train(all_X, all_b, all_i):
    """聚合多个视频的特征并训练"""
    X = np.vstack(all_X)
    b = np.concatenate(all_b)
    i = np.concatenate(all_i)

    print(f"\n  总样本: {X.shape[0]}  边界正例: {b.sum()}  干扰正例: {i.sum()}")

    results = {}
    if b.sum() >= 2:
        print("\n  训练边界检测模型…")
        clf_b, scaler_b, meta_b = train_model(X, b, "boundary")
        path_b = save_model(clf_b, scaler_b, meta_b, "boundary")
        results["boundary"] = {"path": path_b, "f1": meta_b["f1"]}
    else:
        print("  边界正样本不足（<2），跳过边界模型训练")

    if i.sum() >= 2:
        print("\n  训练干扰检测模型…")
        clf_i, scaler_i, meta_i = train_model(X, i, "interference")
        path_i = save_model(clf_i, scaler_i, meta_i, "interference")
        results["interference"] = {"path": path_i, "f1": meta_i["f1"]}
    else:
        print("  干扰正样本不足（<2），跳过干扰模型训练")

    return results


# ============================================================
# 模型推断（供 step3/step5 调用）
# ============================================================

def predict_boundaries(mindex):
    """
    使用已训练的模型预测边界，返回边界时刻列表。
    若模型不存在则静默返回空列表。
    """
    try:
        clf, scaler = load_model("boundary")
        X = extract_features_from_index(mindex)
        X_s = scaler.transform(X)
        times = [pt["time"] for pt in mindex["time_series"]]
        rule_times = [
            t for t, pt in zip(times, mindex["time_series"])
            if bool(pt.get("is_knowledge_boundary", False))
        ]

        slide_ratio = sum(1 for p in mindex["time_series"] if p.get("is_slide_transition")) / max(len(times), 1)
        silence_ratio = sum(1 for p in mindex["time_series"] if p.get("is_silence")) / max(len(times), 1)
        threshold = float(BOUNDARY_MODEL_BASE_THRESHOLD)
        if slide_ratio >= 0.04:
            threshold += 0.05
        if silence_ratio >= 0.30:
            threshold += 0.03
        if slide_ratio <= 0.01 and silence_ratio <= 0.15:
            threshold -= 0.05
        threshold = min(max(threshold, MIN_BOUNDARY_DYNAMIC_THRESHOLD),
                        MAX_BOUNDARY_DYNAMIC_THRESHOLD)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_s)
            pos_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.zeros(len(times))
            model_times = [t for t, p in zip(times, pos_proba) if float(p) >= threshold]
        else:
            preds = clf.predict(X_s)
            model_times = [t for t, p in zip(times, preds) if p == 1]

        merged_times = sorted(set([round(float(t), 2) for t in (model_times + rule_times)]))
        return post_process_boundaries(
            merged_times,
            mindex["time_series"],
            min_gap=BOUNDARY_POST_MIN_GAP,
            max_gap=BOUNDARY_POST_MAX_GAP,
        )
    except Exception:
        return []


def predict_interference(mindex):
    """使用已训练的模型预测干扰时刻列表"""
    try:
        clf, scaler = load_model("interference")
        X   = extract_features_from_index(mindex)
        X_s = scaler.transform(X)
        times = [pt["time"] for pt in mindex["time_series"]]
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_s)
            pos_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.zeros(len(times))
            return [t for t, p in zip(times, pos_proba) if float(p) >= float(INTERFERENCE_MODEL_THRESHOLD)]
        preds = clf.predict(X_s)
        return [t for t, p in zip(times, preds) if p == 1]
    except Exception:
        return []


def _boundary_score(pt):
    score = 0.0
    if pt.get("is_silence"):
        score += 1.5
    if pt.get("is_slide_transition"):
        score += 1.0
    score += max(0.0, 1.0 - float(pt.get("speech_confidence", 0.0))) * 0.5
    return score


def find_best_split(series, left_t, right_t):
    pts = [p for p in series if left_t <= p.get("time", 0.0) <= right_t]
    if not pts:
        return round((left_t + right_t) / 2.0, 2)
    best = max(pts, key=_boundary_score)
    return round(float(best.get("time", (left_t + right_t) / 2.0)), 2)


def post_process_boundaries(predicted_times, series, min_gap=30, max_gap=300):
    """
    后处理边界：
    - 合并过近边界
    - 大间隔补点
    - 倾向保留静默/翻页附近边界
    """
    if not predicted_times:
        return []
    times = sorted(set(round(float(t), 2) for t in predicted_times))
    if len(times) <= 1:
        return times

    point_by_time = {round(float(p.get("time", 0.0)), 2): p for p in series}

    merged = [times[0]]
    for t in times[1:]:
        if t - merged[-1] >= float(min_gap):
            merged.append(t)
            continue
        prev_pt = point_by_time.get(merged[-1], {"time": merged[-1]})
        cur_pt = point_by_time.get(t, {"time": t})
        if _boundary_score(cur_pt) > _boundary_score(prev_pt):
            merged[-1] = t

    final = [merged[0]]
    for t in merged[1:]:
        while t - final[-1] > float(max_gap):
            split = find_best_split(series, final[-1], t)
            if split - final[-1] < float(min_gap):
                split = round(final[-1] + float(min_gap), 2)
            if t - split < float(min_gap):
                break
            final.append(split)
        final.append(t)

    return sorted(set(round(x, 2) for x in final))


# ============================================================
# 评估
# ============================================================

def evaluate(video_path, annotation_path):
    """评估模型在给定视频上的表现"""
    out_dir, vname = get_output_dir(video_path)
    idx_path = os.path.join(out_dir, "multimodal_index.json")
    if not os.path.exists(idx_path):
        print("multimodal_index.json 不存在，无法评估")
        sys.exit(1)

    mindex = load_json(idx_path)
    ann    = load_json(annotation_path)
    annotations = ann.get("annotations", [])
    series = mindex["time_series"]
    X = extract_features_from_index(mindex)
    b_lbl, i_lbl = label_from_annotation(series, annotations)

    for model_type, true_lbl in [("boundary", b_lbl), ("interference", i_lbl)]:
        try:
            clf, scaler = load_model(model_type)
            from sklearn.metrics import classification_report
            preds  = clf.predict(scaler.transform(X))
            report = classification_report(true_lbl, preds, zero_division=0)
            print(f"\n[{model_type}] 评估结果:")
            print(report)
        except FileNotFoundError:
            print(f"[{model_type}] 模型未找到，跳过")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="训练知识点边界/干扰检测模型")
    parser.add_argument("--video",           default=None,
                        help="单个视频文件路径")
    parser.add_argument("--annotation",      default=None,
                        help="单个视频的标注 JSON 文件路径")
    parser.add_argument("--annotation_dir",  default=None,
                        help="批量标注目录（每个 .json 对应同名视频）")
    parser.add_argument("--eval",            action="store_true",
                        help="只评估，不训练")
    args = parser.parse_args()

    if args.eval:
        if not (args.video and args.annotation):
            print("评估模式需要提供 --video 和 --annotation")
            sys.exit(1)
        evaluate(args.video, args.annotation)
        return

    # 收集训练数据
    pairs = []
    if args.video and args.annotation:
        pairs.append((args.video, args.annotation))
    elif args.annotation_dir:
        for apath in glob.glob(os.path.join(args.annotation_dir, "*.json")):
            vname = Path(apath).stem
            for ext in (".mp4", ".avi", ".mov", ".mkv"):
                # Search in LESSON_DIR (from config) first, then lesson/ sibling of annotation_dir
                candidate_dirs = [
                    LESSON_DIR,
                    os.path.join(os.path.dirname(os.path.abspath(args.annotation_dir)),
                                 "lesson"),
                ]
                for cdir in candidate_dirs:
                    vpath = os.path.join(cdir, vname + ext)
                    if os.path.exists(vpath):
                        pairs.append((vpath, apath))
                        break
                else:
                    continue
                break
    else:
        print("请提供 --video + --annotation 或 --annotation_dir")
        parser.print_help()
        sys.exit(1)

    if not pairs:
        print("未找到任何视频-标注对")
        sys.exit(1)

    all_X, all_b, all_i = [], [], []
    for vpath, apath in pairs:
        print(f"\n处理: {os.path.basename(vpath)}")
        try:
            X, b, i, _ = train_on_video(vpath, apath)
            all_X.append(X)
            all_b.append(b)
            all_i.append(i)
        except Exception as e:
            print(f"  出错（跳过）: {e}")
            traceback.print_exc()

    if not all_X:
        print("没有可用的训练数据")
        sys.exit(1)

    results = aggregate_and_train(all_X, all_b, all_i)
    print("\n训练完成！模型保存在:", MODELS_DIR)
    for mtype, info in results.items():
        print(f"  {mtype}: F1={info['f1']:.4f}  →  {info['path']}")


if __name__ == "__main__":
    main()
