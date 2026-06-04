#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动调参：基于 Optuna 搜索训练参数。
"""

import os
import json
import argparse
import statistics
import tempfile
import shutil

import optuna

import train
from run_all import run_pipeline


PARAM_KEYS = {
    "BOUNDARY_MODEL_BASE_THRESHOLD": (0.35, 0.75),
    "BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER": (1.0, 10.0),
    "INTERFERENCE_MODEL_THRESHOLD": (0.35, 0.75),
    "INTERFERENCE_POSITIVE_WEIGHT_MULTIPLIER": (1.0, 10.0),
}


def _round_params(params):
    out = {}
    for k, v in params.items():
        if "THRESHOLD" in k:
            out[k] = round(float(v), 2)
        else:
            out[k] = round(float(v), 2)
    return out


def apply_params_to_train(params):
    for k, v in params.items():
        setattr(train, k, float(v))


def score_key(mode, model_type):
    if mode == "f-beta":
        return "f_beta_binary" if model_type == "boundary" else "f_beta_binary"
    return "f1_binary"


def metric_label(mode, model_type):
    if mode == "f-beta":
        return "F1.5" if model_type == "boundary" else "F0.5"
    return "F1"


def metric_json_key(mode, model_type):
    if mode == "f-beta":
        return "f1.5" if model_type == "boundary" else "f0.5"
    return "f1"


def train_once(X, b_lbl, i_lbl, mode):
    b_beta = 1.5 if mode == "f-beta" else 1.0
    i_beta = 0.5 if mode == "f-beta" else 1.0
    clf_b, scaler_b, meta_b = train.train_model(
        X, b_lbl, "boundary", optimize_metric=("f-beta" if mode == "f-beta" else "f1"), beta=b_beta
    )
    clf_i, scaler_i, meta_i = train.train_model(
        X, i_lbl, "interference", optimize_metric=("f-beta" if mode == "f-beta" else "f1"), beta=i_beta
    )
    return (clf_b, scaler_b, meta_b), (clf_i, scaler_i, meta_i)


def evaluate_business(video_path, video_name, clf_b, scaler_b, meta_b, clf_i, scaler_i, meta_i):
    tmp_models = tempfile.mkdtemp(prefix=f"autotune_models_{video_name}_")
    old_models_dir = train.MODELS_DIR
    try:
        train.MODELS_DIR = tmp_models
        meta_b = dict(meta_b)
        meta_i = dict(meta_i)
        train.save_scoped_model(clf_b, scaler_b, meta_b, "boundary", model_scope=video_name)
        train.save_scoped_model(clf_i, scaler_i, meta_i, "interference", model_scope=video_name)
        results = run_pipeline(video_path, start_step=3, end_step=5)
        if 5 not in results or results[5].get("status") != "ok":
            return {"knowledge_count": 0, "avg_duration": 0.0, "removed_count": 9999}, -9999.0
        final_index = results[5]["result"]
        stats = final_index.get("stats", {})
        knowledge_count = int(stats.get("total_output_clips", 0))
        total_dur = float(stats.get("total_output_duration_s", 0.0))
        removed_count = int(stats.get("total_removed_segments", 0))
        avg_duration = total_dur / max(knowledge_count, 1)
        score = (knowledge_count * 1.0) + (avg_duration / 120.0) - (removed_count * 1.5)
        return {
            "knowledge_count": knowledge_count,
            "avg_duration": round(avg_duration, 2),
            "removed_count": removed_count,
        }, float(score)
    finally:
        train.MODELS_DIR = old_models_dir
        shutil.rmtree(tmp_models, ignore_errors=True)


def collect_global_medians(models_dir, current_best):
    vals = {k: [] for k in PARAM_KEYS.keys()}
    for k, v in current_best.items():
        if k in vals:
            vals[k].append(float(v))

    if os.path.isdir(models_dir):
        for scope in sorted(os.listdir(models_dir)):
            scope_dir = os.path.join(models_dir, scope)
            if not os.path.isdir(scope_dir):
                continue
            for mtype in ("boundary", "interference"):
                path = os.path.join(scope_dir, f"{mtype}_model.pkl")
                if not os.path.exists(path):
                    continue
                try:
                    import pickle
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                    tuned = (obj.get("meta") or {}).get("tuned_params", {})
                    for k in vals:
                        if k in tuned:
                            vals[k].append(float(tuned[k]))
                except Exception:
                    continue

    med = {}
    for k, arr in vals.items():
        if arr:
            med[k] = round(float(statistics.median(arr)), 2)
        else:
            med[k] = round(float(current_best[k]), 2)
    return med


def write_config_values(config_path, params):
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    for k, v in params.items():
        import re
        text = re.sub(
            rf"^{k}\s*=.*$",
            f"{k} = {float(v):.2f}",
            text,
            flags=re.MULTILINE,
        )
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(text)


def maybe_save_plot(study, png_path):
    if not png_path:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("提示: 未安装 matplotlib，跳过收敛曲线保存")
        return
    ys = [t.value for t in study.trials if t.value is not None]
    if not ys:
        return
    best_curve = []
    best = -1e18
    for v in ys:
        best = max(best, float(v))
        best_curve.append(best)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(ys) + 1), ys, label="trial")
    plt.plot(range(1, len(best_curve) + 1), best_curve, label="best")
    plt.xlabel("Trial")
    plt.ylabel("Objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()
    print(f"收敛曲线已保存: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="自动调参（Optuna）")
    parser.add_argument("--video", required=True, help="视频路径")
    parser.add_argument("--annotation", required=True, help="标注文件路径")
    parser.add_argument("--trials", type=int, default=50, help="迭代次数")
    parser.add_argument("--mode", choices=["f1", "f-beta", "business"], default="f1", help="优化模式")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument("--apply-config", action="store_true", help="将全局中位数参数写回 config.py")
    parser.add_argument("--plot", default=None, help="可选：收敛曲线 PNG 保存路径")
    args = parser.parse_args()

    X, b_lbl, i_lbl, video_name = train.train_on_video(args.video, args.annotation)

    if b_lbl.sum() < 2 or i_lbl.sum() < 2:
        raise RuntimeError("边界/干扰正样本不足，无法自动调参")

    original_params = {k: float(getattr(train, k)) for k in PARAM_KEYS.keys()}
    trial_records = []

    def objective(trial):
        params = {
            "BOUNDARY_MODEL_BASE_THRESHOLD": trial.suggest_float("BOUNDARY_MODEL_BASE_THRESHOLD", 0.35, 0.75, step=0.01),
            "BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER": trial.suggest_float("BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER", 1.0, 10.0, step=0.1),
            "INTERFERENCE_MODEL_THRESHOLD": trial.suggest_float("INTERFERENCE_MODEL_THRESHOLD", 0.35, 0.75, step=0.01),
            "INTERFERENCE_POSITIVE_WEIGHT_MULTIPLIER": trial.suggest_float("INTERFERENCE_POSITIVE_WEIGHT_MULTIPLIER", 1.0, 10.0, step=0.1),
        }
        apply_params_to_train(params)

        (clf_b, scaler_b, meta_b), (clf_i, scaler_i, meta_i) = train_once(X, b_lbl, i_lbl, args.mode)
        boundary_metric = float(meta_b[score_key(args.mode, "boundary")])
        interference_metric = float(meta_i[score_key(args.mode, "interference")])
        obj = (boundary_metric + interference_metric) / 2.0
        business_metrics = None

        if args.mode == "business":
            business_metrics, business_score = evaluate_business(
                args.video, video_name, clf_b, scaler_b, meta_b, clf_i, scaler_i, meta_i
            )
            obj = float(business_score)

        p = _round_params(params)
        print(f"\nTrial {trial.number + 1}/{args.trials}:")
        print(
            "  参数: "
            f"b_th={p['BOUNDARY_MODEL_BASE_THRESHOLD']}, "
            f"b_w={p['BOUNDARY_POSITIVE_WEIGHT_MULTIPLIER']}, "
            f"i_th={p['INTERFERENCE_MODEL_THRESHOLD']}, "
            f"i_w={p['INTERFERENCE_POSITIVE_WEIGHT_MULTIPLIER']}"
        )
        print(
            f"  边界: P={meta_b['precision']:.2f}, R={meta_b['recall']:.2f}, "
            f"{metric_label(args.mode, 'boundary')}={boundary_metric:.2f}"
        )
        print(
            f"  干扰: P={meta_i['precision']:.2f}, R={meta_i['recall']:.2f}, "
            f"{metric_label(args.mode, 'interference')}={interference_metric:.2f}"
        )
        if business_metrics is not None:
            print(
                "  业务: "
                f"知识点数={business_metrics['knowledge_count']}, "
                f"平均时长={business_metrics['avg_duration']:.2f}s, "
                f"误删片段数={business_metrics['removed_count']}"
            )
        print(f"  目标值: {obj:.4f}")

        trial_records.append({
            "trial": trial.number + 1,
            "params": p,
            "objective": round(float(obj), 4),
            "boundary": {
                "precision": float(meta_b["precision"]),
                "recall": float(meta_b["recall"]),
                "f1": float(meta_b["f1_binary"]),
                "f_beta": float(meta_b["f_beta_binary"]),
                metric_json_key(args.mode, "boundary"): float(boundary_metric),
                "best_threshold": float(meta_b["best_threshold"]),
            },
            "interference": {
                "precision": float(meta_i["precision"]),
                "recall": float(meta_i["recall"]),
                "f1": float(meta_i["f1_binary"]),
                "f_beta": float(meta_i["f_beta_binary"]),
                metric_json_key(args.mode, "interference"): float(interference_metric),
                "best_threshold": float(meta_i["best_threshold"]),
            },
            "business": business_metrics,
        })
        return float(obj)

    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective, n_trials=max(1, int(args.trials)))
    finally:
        apply_params_to_train(original_params)

    best_params = _round_params(study.best_params)
    apply_params_to_train(best_params)
    (clf_b, scaler_b, meta_b), (clf_i, scaler_i, meta_i) = train_once(X, b_lbl, i_lbl, args.mode)
    tuned = dict(best_params)
    meta_b = dict(meta_b)
    meta_i = dict(meta_i)
    meta_b["tuned_params"] = tuned
    meta_i["tuned_params"] = tuned
    train.save_scoped_model(clf_b, scaler_b, meta_b, "boundary", model_scope=video_name)
    train.save_scoped_model(clf_i, scaler_i, meta_i, "interference", model_scope=video_name)

    global_median = collect_global_medians(train.MODELS_DIR, tuned)
    if args.apply_config:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
        write_config_values(cfg_path, global_median)
        print("已写回 config.py（全局中位数参数）")
    else:
        print("提示: 运行 --apply-config 写回 config.py，或手动确认")

    maybe_save_plot(study, args.plot)

    best_record = None
    for rec in trial_records:
        if rec["trial"] == (study.best_trial.number + 1):
            best_record = rec
            break

    result = {
        "video": video_name,
        "video_best_params": best_params,
        "global_median_params": global_median,
        "best_value": round(float(study.best_value), 4),
        "metrics": {
            "boundary": {
                "precision": float(best_record["boundary"]["precision"]) if best_record else float(meta_b["precision"]),
                "recall": float(best_record["boundary"]["recall"]) if best_record else float(meta_b["recall"]),
                "f1": float(best_record["boundary"]["f1"]) if best_record else float(meta_b["f1_binary"]),
                "f_beta": float(best_record["boundary"]["f_beta"]) if best_record else float(meta_b["f_beta_binary"]),
                metric_json_key(args.mode, "boundary"): (
                    float(best_record["boundary"][metric_json_key(args.mode, "boundary")])
                    if best_record else float(meta_b[score_key(args.mode, "boundary")])
                ),
            },
            "interference": {
                "precision": float(best_record["interference"]["precision"]) if best_record else float(meta_i["precision"]),
                "recall": float(best_record["interference"]["recall"]) if best_record else float(meta_i["recall"]),
                "f1": float(best_record["interference"]["f1"]) if best_record else float(meta_i["f1_binary"]),
                "f_beta": float(best_record["interference"]["f_beta"]) if best_record else float(meta_i["f_beta_binary"]),
                metric_json_key(args.mode, "interference"): (
                    float(best_record["interference"][metric_json_key(args.mode, "interference")])
                    if best_record else float(meta_i[score_key(args.mode, "interference")])
                ),
            },
        },
        "trials": int(args.trials),
        "mode": args.mode,
        "trial_history": trial_records,
    }
    if args.mode == "business" and best_record:
        result["business"] = best_record.get("business")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n训练完成！")
    print(f"该视频最优参数: {best_params}")
    print(f"全局中位数参数: {global_median}")
    print(f"结果已输出: {args.output}")


if __name__ == "__main__":
    main()
