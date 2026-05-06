#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤2: 语音维度分析
  1. 从视频提取音频（ffmpeg）
  2. 语音转录（faster-whisper，纯 CPU）
  3. 说话人区分（MFCC 聚类）
  4. 置信度标记

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
    MIN_SPEAKERS                = _cfg.MIN_SPEAKERS
    MAX_SPEAKERS                = _cfg.MAX_SPEAKERS
    SPEECH_CONFIDENCE_THRESHOLD = _cfg.SPEECH_CONFIDENCE_THRESHOLD
    NO_SPEECH_PROB_THRESHOLD    = _cfg.NO_SPEECH_PROB_THRESHOLD
except ImportError:
    OUTPUT_DIR                  = r"D:\video\output"
    WHISPER_MODEL_SIZE          = "base"
    WHISPER_LANGUAGE            = "zh"
    WHISPER_BEAM_SIZE           = 3
    MIN_SPEAKERS                = 1
    MAX_SPEAKERS                = 4
    SPEECH_CONFIDENCE_THRESHOLD = 0.60
    NO_SPEECH_PROB_THRESHOLD    = 0.50


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


# ============================================================
# 说话人区分
# ============================================================

def diarize_speakers(audio_path, segments):
    """
    基于 MFCC 特征的简单说话人聚类。
    发言时长最多的说话人标为「教师」。
    """
    if len(segments) < 2:
        for s in segments:
            s["speaker"] = "教师"
        return segments, ["教师"]

    print("  说话人区分（MFCC 聚类）…")

    try:
        import librosa
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  警告: 缺少 librosa 或 scikit-learn，所有片段归为「教师」")
        for s in segments:
            s["speaker"] = "教师"
        return segments, ["教师"]

    try:
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        feats, valid_idx = [], []
        for i, seg in enumerate(segments):
            ss = int(seg["start"] * sr)
            es = int(seg["end"]   * sr)
            chunk = audio[ss:es]
            if len(chunk) < sr * 0.4:          # 太短则跳过
                continue
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
            feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            feats.append(feat)
            valid_idx.append(i)

        if len(valid_idx) < 2:
            for s in segments:
                s["speaker"] = "教师"
            return segments, ["教师"]

        X = StandardScaler().fit_transform(np.array(feats))

        # 自动估计说话人数：每 10 段预设 1 人，上下限约束
        n_spk = max(MIN_SPEAKERS,
                    min(MAX_SPEAKERS, len(valid_idx) // 10 + 1))
        n_spk = min(n_spk, len(valid_idx))

        if n_spk == 1:
            labels = np.zeros(len(valid_idx), dtype=int)
        else:
            labels = AgglomerativeClustering(
                n_clusters=n_spk, metric="euclidean", linkage="ward"
            ).fit_predict(X)

        # 统计每类发言时长，时长最长 → 教师
        dur_map = {}
        for vi, lb in zip(valid_idx, labels):
            dur_map[lb] = dur_map.get(lb, 0) + (segments[vi]["end"] - segments[vi]["start"])

        sorted_labels = sorted(dur_map, key=dur_map.get, reverse=True)
        name_map = {lb: ("教师" if i == 0 else f"学生{i}") for i, lb in enumerate(sorted_labels)}

        # idx → speaker name map
        idx2spk = {}
        for vi, lb in zip(valid_idx, labels):
            idx2spk[vi] = name_map[lb]

        for i, seg in enumerate(segments):
            if i in idx2spk:
                seg["speaker"] = idx2spk[i]
            else:
                # 最近有效帧的说话人
                nearest = min(valid_idx, key=lambda j: abs(j - i))
                lb = labels[valid_idx.index(nearest)]
                seg["speaker"] = name_map[lb]

        speakers = [name_map[lb] for lb in sorted_labels]
        print(f"  说话人: {speakers}")
        return segments, speakers

    except Exception as e:
        print(f"  说话人区分出错（{e}），全部标为「教师」")
        for s in segments:
            s["speaker"] = "教师"
        return segments, ["教师"]


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

        valid_segs = [s for s in segments
                      if s["no_speech_prob"] < NO_SPEECH_PROB_THRESHOLD]
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
