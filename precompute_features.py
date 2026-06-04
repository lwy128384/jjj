#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量预计算特征缓存：
遍历 OUTPUT_DIR/*/multimodal_index.json，生成 features.npy。
"""

import os
import glob
import argparse

try:
    import config as _cfg
    OUTPUT_DIR = _cfg.OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = r"D:\video\output"

from train import load_json, cached_extract_features


def main():
    parser = argparse.ArgumentParser(description="预计算训练特征缓存")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录（默认 config.OUTPUT_DIR）")
    parser.add_argument("--force", action="store_true", help="强制重算已有缓存")
    args = parser.parse_args()

    pattern = os.path.join(args.output_dir, "*", "multimodal_index.json")
    index_files = sorted(glob.glob(pattern))
    if not index_files:
        print(f"未找到索引文件: {pattern}")
        return

    total = len(index_files)
    done = 0
    skipped = 0
    failed = 0

    for i, idx_path in enumerate(index_files, 1):
        video_dir = os.path.dirname(idx_path)
        cache_path = os.path.join(video_dir, "features.npy")
        video_name = os.path.basename(video_dir)
        print(f"[{i}/{total}] {video_name}")
        try:
            if os.path.exists(cache_path) and not args.force:
                skipped += 1
                print("  已存在，跳过")
                continue
            if args.force and os.path.exists(cache_path):
                os.remove(cache_path)
            mindex = load_json(idx_path)
            X = cached_extract_features(mindex, cache_path)
            done += 1
            print(f"  已生成: {cache_path}  shape={tuple(X.shape)}")
        except Exception as exc:
            failed += 1
            print(f"  失败: {exc}")

    print("\n预计算完成")
    print(f"  成功: {done}")
    print(f"  跳过: {skipped}")
    print(f"  失败: {failed}")


if __name__ == "__main__":
    main()
