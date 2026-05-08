#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
共享配置文件 — 所有步骤通用
修改此文件中的参数来调整各步骤行为
"""

import os

# ============================================================
# 基础路径配置
# ============================================================
BASE_DIR    = r"D:\video"
LESSON_DIR  = os.path.join(BASE_DIR, "lesson")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

# ============================================================
# 步骤1 — 视觉分析
# ============================================================
VISUAL_SAMPLE_FPS = 1          # 每秒采样帧数（用于分析）

# 讲台区域（相对画面宽高，左/上/右/下）
# 默认：中央讲台附近，尽量避开前排学生区域（适合摄像机位置基本居中）
# 若摄像机位置偏左/偏右或远景镜头，请按实际讲台位置重新调整
PODIUM_REGION = (0.38, 0.42, 0.66, 0.94)

# PPT 区域参数（相对坐标，左/上/右/下）
# 建议覆盖投影屏幕主体，尽量避开下方听众区域
PPT_REGION = (0.02, 0.02, 0.98, 0.80)
# 全屏 PPT 识别与 OCR/翻页检测区域
PPT_REGION_FULLSCREEN = (0.00, 0.00, 1.00, 0.90)

# 全屏 PPT 判定阈值
FULLSCREEN_BRIGHT_RATIO = 0.35
FULLSCREEN_LOW_SAT_RATIO = 0.45
FULLSCREEN_EDGE_RATIO = 0.02

# 幻灯片切换 SSIM 阈值（越小越灵敏，0~1）
SLIDE_CHANGE_THRESHOLD = 0.70

# 讲台前景像素占比阈值（低于此值 → 教师不在讲台）
TEACHER_PRESENCE_THRESHOLD = 0.05

# 背景初始化帧数
BG_INIT_FRAMES = 30

# OCR 最低置信度
OCR_CONFIDENCE_THRESHOLD = 0.40

# ============================================================
# 步骤2 — 语音分析
# ============================================================
WHISPER_MODEL_SIZE  = "base"   # tiny / base / small / medium
WHISPER_LANGUAGE    = "zh"     # 语言代码；None = 自动检测
WHISPER_BEAM_SIZE   = 3

MIN_SPEAKERS = 1
MAX_SPEAKERS = 4

SPEECH_CONFIDENCE_THRESHOLD = 0.60   # 低于此值 → 低置信度
NO_SPEECH_PROB_THRESHOLD    = 0.50   # 高于此值 → 视为非语音/静默

# ============================================================
# 步骤3 — 文本语义分析
# ============================================================
SEMANTIC_WINDOW_SIZE    = 3    # 滑动窗口大小（语音段数）
BOUNDARY_THRESHOLD      = 0.35 # 余弦距离阈值（超过 → 潜在边界）
MIN_KNOWLEDGE_DURATION  = 45   # 最短知识点时长（秒）
MAX_KNOWLEDGE_DURATION  = 600  # 最长知识点时长（秒）
TOP_KEYWORDS            = 5    # 每知识点关键词数
MIN_TEXT_LENGTH         = 5    # 最短有效文本长度（字符）

# ============================================================
# 步骤4 — 多模态对齐
# ============================================================
TIME_RESOLUTION = 1.0          # 时间轴分辨率（秒）

# ============================================================
# 步骤5 — 多模态融合与剪辑
# ============================================================
INTERFERENCE_TEACHER_ABSENT_RATIO = 0.70  # 教师缺席比例 > 此值 → 干扰
INTERFERENCE_LOW_SPEECH_RATIO     = 0.80  # 静默/低音比例 > 此值 → 干扰
INTERFERENCE_SILENCE_THRESHOLD    = 15.0  # 连续静默 > 此值（秒） → 干扰
INTERFERENCE_MIN_DURATION         = 5.0   # 最短干扰持续时长（秒）

SEGMENT_MERGE_GAP    = 5.0    # 间隔 < 此值的相邻段合并（秒）
SEGMENT_MIN_DURATION = 20.0   # 最短保留片段（秒）
SEGMENT_PADDING      = 1.0    # 片段首尾缓冲（秒）

# ============================================================
# 训练配置
# ============================================================
TRAIN_MODEL_FILE = "boundary_model.pkl"
TRAIN_TEST_SPLIT = 0.20
RANDOM_STATE     = 42
