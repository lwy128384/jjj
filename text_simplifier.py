#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一将输出文本转为简体中文。
"""

import json

try:
    from opencc import OpenCC
except ImportError as exc:
    OpenCC = None
    _OPENCC_IMPORT_ERROR = exc
else:
    _OPENCC_IMPORT_ERROR = None

_CONVERTER = OpenCC("t2s") if OpenCC is not None else None


def _require_converter():
    if _CONVERTER is None:
        raise RuntimeError(
            "缺少 opencc-python-reimplemented 依赖，请先运行: pip install -r requirements.txt"
        ) from _OPENCC_IMPORT_ERROR


def simplify_text(value):
    if value is None:
        return ""
    text = str(value)
    if not text:
        return text
    _require_converter()
    return _CONVERTER.convert(text)


def simplify_video_name(name):
    simplified = simplify_text(name).strip()
    return simplified or "未命名视频"


def simplify_recursive(obj, exclude_keys=None):
    exclude = set(exclude_keys or [])

    if isinstance(obj, str):
        return simplify_text(obj)
    if isinstance(obj, list):
        return [simplify_recursive(item, exclude) for item in obj]
    if isinstance(obj, tuple):
        return tuple(simplify_recursive(item, exclude) for item in obj)
    if isinstance(obj, dict):
        normalized = {}
        for key, value in obj.items():
            normalized_key = simplify_text(key) if isinstance(key, str) else key
            if key in exclude:
                normalized[normalized_key] = value
            else:
                normalized[normalized_key] = simplify_recursive(value, exclude)
        return normalized
    return obj


def load_json_simplified(path):
    with open(path, "r", encoding="utf-8") as f:
        return simplify_recursive(json.load(f))


def dump_json_simplified(data, path, exclude_keys=None):
    normalized = simplify_recursive(data, exclude_keys=exclude_keys)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return normalized
