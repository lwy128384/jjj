#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练模块 — 利用人工标注数据提升知识点边界检测准确率

标注文件格式 (annotation.json):
  {
    "video": "lesson1.mp4",
    "annotations": [
      {"start": 0.0,   "end": 180.0, "title": "第一章绪论",  "is_interference": false},
      {"start": 180.0, "end": 210.0, "title": "课间休息",    "is_interference": true},
      {"start": 210.0, "end": 400.0, "title": "第一节基本概念","is_interference": false}
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
import numpy as np
from pathlib import Path

# ============================================================
# 默认参数
# ============================================================
try:
    import config as _cfg
    OUTPUT_DIR       = _cfg.OUTPUT_DIR
    MODELS_DIR       = _cfg.MODELS_DIR
    TRAIN_MODEL_FILE = _cfg.TRAIN_MODEL_FILE
    TRAIN_TEST_SPLIT = _cfg.TRAIN_TEST_SPLIT
    RANDOM_STATE     = _cfg.RANDOM_STATE
    TIME_RESOLUTION  = _cfg.TIME_RESOLUTION
except ImportError:
    OUTPUT_DIR       = r"D:\video\output"
    MODELS_DIR       = r"D:\video\models"
    TRAIN_MODEL_FILE = "boundary_model.pkl"
    TRAIN_TEST_SPLIT = 0.20
    RANDOM_STATE     = 42
    TIME_RESOLUTION  = 1.0


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

    feats  = []
    for i, pt in enumerate(series):
        # 滚动窗口统计
        win = series[max(0, i-W):i+1]
        roll_sil  = sum(1 for p in win if p["is_silence"])   / max(len(win), 1)
        roll_tchr = sum(1 for p in win if p["teacher_present"]) / max(len(win), 1)

        # 前一时刻特征
        prev = series[i-1] if i > 0 else pt
        # 幻灯片跳变
        slide_delta = int(pt["slide_idx"] != prev.get("slide_idx", 0))

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

    for ann in annotations:
        # 边界：起点附近
        for i, t in enumerate(times):
            if abs(t - ann["start"]) <= tolerance_sec:
                bound_label[i] = 1
        # 干扰段内的点
        if ann.get("is_interference", False):
            for i, t in enumerate(times):
                if ann["start"] <= t <= ann["end"]:
                    interf_label[i] = 1

    return bound_label, interf_label


# ============================================================
# 模型训练
# ============================================================

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
    except ImportError:
        raise RuntimeError("请安装 scikit-learn:\n  pip install scikit-learn")

    # 类别不平衡处理：用 class_weight
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
        if y.sum() > 1 else None
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators  = 200,
        max_depth     = 8,
        class_weight  = "balanced",
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
        preds = clf.predict(X_s)
        times = [pt["time"] for pt in mindex["time_series"]]
        return [t for t, p in zip(times, preds) if p == 1]
    except Exception:
        return []


def predict_interference(mindex):
    """使用已训练的模型预测干扰时刻列表"""
    try:
        clf, scaler = load_model("interference")
        X   = extract_features_from_index(mindex)
        X_s = scaler.transform(X)
        preds = clf.predict(X_s)
        times = [pt["time"] for pt in mindex["time_series"]]
        return [t for t, p in zip(times, preds) if p == 1]
    except Exception:
        return []


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
                vpath = os.path.join(
                    os.path.dirname(args.annotation_dir), "lesson", vname + ext)
                if os.path.exists(vpath):
                    pairs.append((vpath, apath))
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
