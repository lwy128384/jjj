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
PODIUM_REGION = (0.22, 0.45, 0.85, 0.70)

# 全屏 PPT 识别与 OCR/翻页检测区域
PPT_REGION_FULLSCREEN = (0.15, 0.00, 0.87, 0.95)

# 全屏 PPT 判定阈值
FULLSCREEN_BRIGHT_RATIO = 0.35
FULLSCREEN_LOW_SAT_RATIO = 0.45
FULLSCREEN_EDGE_RATIO = 0.005      # 从 0.02 降到 0.005，避免漏检文字少的页面

# 幻灯片切换 SSIM 阈值（越小越灵敏，0~1）
SLIDE_CHANGE_THRESHOLD = 0.70

# 讲台前景像素占比阈值（低于此值 → 教师不在讲台）
TEACHER_PRESENCE_THRESHOLD = 0.05

# 背景初始化帧数
BG_INIT_FRAMES = 30

# OCR 最低置信度
OCR_CONFIDENCE_THRESHOLD = 0.15      # 从 0.40 降到 0.15
OCR_RELAXED_CONFIDENCE_MIN = 0.08  # 从 0.20 降到 0.08
OCR_RELAXED_CONFIDENCE_FACTOR = 0.50
OCR_UPSCALE_MIN_DIM = 600          # 从 900 降到 600，更早触发放大
OCR_UPSCALE_FACTOR = 2.0           # 从 1.5 提高到 2.0
OCR_GAUSSIAN_KERNEL = 1            # 从 3 降到 1，减少模糊
OCR_ADAPTIVE_BLOCK_SIZE = 21       # 从 31 降到 21
OCR_ADAPTIVE_C = 5                 # 从 11 降到 5

# ============================================================
# 步骤2 — 语音分析
# ============================================================
WHISPER_MODEL_SIZE  = "base"   # tiny / base / small / medium
WHISPER_LANGUAGE    = "zh"     # 语言代码；None = 自动检测
WHISPER_BEAM_SIZE   = 3

MIN_SPEAKERS = 2
MAX_SPEAKERS = 2

# 说话人区分固定为二分类：教师 / 学生
DIARIZATION_N_CLUSTERS = 2

# 文本特征（教师话术 / 学生提问话术）融合权重
DIARIZATION_TEXT_WEIGHT = 0.38
DIARIZATION_ACOUSTIC_WEIGHT = 0.62

# 教师口语语气词（用于 teacher cues 与重复词加分）
DIARIZATION_FILLER_CUES = ["嗯", "啊", "呃", "这个", "那个", "就是", "那么"]

# 教师常见课堂表达（命中越多，越偏向教师）
DIARIZATION_TEACHER_BASE_CUES = [
    "我们", "下面", "今天", "讲", "来看", "举个例子", "同学们", "回顾",
    "总结", "总之", "注意", "定义", "公式", "原理", "人工智能", "历史",
    "先", "然后", "接下来", "这个问题",
]
DIARIZATION_TEACHER_EXTRA_CUES = ["大家", "注意看", "来看一下"]
DIARIZATION_TEACHER_CUES = DIARIZATION_TEACHER_BASE_CUES + DIARIZATION_FILLER_CUES + DIARIZATION_TEACHER_EXTRA_CUES

# 学生常见提问表达（命中越多，越偏向学生）
DIARIZATION_STUDENT_CUES = [
    "老师", "请问", "我想问", "是不是", "对吗", "为什么", "怎么",
    "听不清", "没听懂", "可以再说", "啥意思",
]

# 上下文平滑：修正时间上孤立的误判片段
DIARIZATION_SMOOTH_WINDOW = 3
DIARIZATION_SMOOTH_MAX_DURATION = 4.0
DIARIZATION_SMOOTH_MIN_NEIGHBORS = 2

# 片段级教师概率判定与边界复判参数
DIARIZATION_TEACHER_PROB_CENTER_QUANTILE = 0.35
DIARIZATION_TEACHER_PROB_SIGMOID_SCALE = 1.0
DIARIZATION_TEACHER_PROB_BIAS = 0.30
DIARIZATION_TEACHER_PROB_THRESHOLD = 0.45
DIARIZATION_VOICEPRINT_BORDERLINE_LOW = 0.28
DIARIZATION_VOICEPRINT_BORDERLINE_HIGH = 0.60
DIARIZATION_VOICEPRINT_TEACHER_PROTO_QUANTILE = 0.65

# 声纹辅助复判（先用已判教师片段提取教师共同声纹，再回查学生片段）
DIARIZATION_VOICEPRINT_ASSIST_ENABLED = True
DIARIZATION_VOICEPRINT_MIN_TEACHER_SAMPLES = 2
DIARIZATION_VOICEPRINT_MIN_SEGMENT_DURATION = 0.8
DIARIZATION_VOICEPRINT_MAX_STUDENT_DURATION = 40.0
DIARIZATION_VOICEPRINT_SIMILARITY_THRESHOLD = 0.82
DIARIZATION_VOICEPRINT_STUDENT_MARGIN = 0.03

SPEECH_CONFIDENCE_THRESHOLD = 0.60   # 低于此值 → 低置信度
NO_SPEECH_PROB_THRESHOLD    = 0.80   # 高于此值 → 倾向判为非语音/静默 (可被文本规则覆盖)
NO_SPEECH_IGNORE_WITH_TEXT  = True   # 有文本时智能忽略 no_speech_prob
NO_SPEECH_TEXT_SHORT_LEN    = 3      # 高 no_speech_prob + 极短文本（<=）仍按噪音过滤
STEP2_ENABLE_TEXT_CORRECTION = True
STEP2_TEXT_CORRECTION_TERMS = [
    "人工智能", "机器学习", "深度学习", "神经网络", "图灵", "图灵测试",
    "亚里士多德", "算法", "数据集", "低谷", "模型", "训练", "推理",
]
STEP2_TEXT_CORRECTION_MIN_CHARS = 2
# 拼音归一化编辑距离阈值；默认0.22在常见误识别（如“低谱”→“低谷”）和误改风险间折中
STEP2_TEXT_CORRECTION_MAX_PINYIN_NORM_DIST = 0.22
STEP2_TEXT_CORRECTION_MAX_CHAR_DIST = 1
STEP2_TEXT_CORRECTION_MAX_LENGTH_DIFF = 1
STEP2_TEXT_CORRECTION_CHAR_WEIGHT = 0.05

# ============================================================
# 步骤3 — 文本语义分析
# ============================================================
SEMANTIC_WINDOW_SIZE    = 3    # 滑动窗口大小（语音段数）
BOUNDARY_THRESHOLD      = 0.35 # 余弦距离阈值（超过 → 潜在边界）
MIN_KNOWLEDGE_DURATION  = 45   # 最短知识点时长（秒）
MAX_KNOWLEDGE_DURATION  = 600  # 最长知识点时长（秒）
TOP_KEYWORDS            = 5    # 每知识点关键词数
MIN_TEXT_LENGTH         = 5    # 最短有效文本长度（字符）
KEYWORD_TITLE_COUNT     = 2    # 用于知识点标题拼接的关键词数量
KEYWORD_MIN_DOC_FREQ    = 2    # 关键词在有效语音段中的最低文档频次（过滤偶发噪声词）
KEYWORD_BLACKLIST = [
    "这个", "那个", "就是", "然后", "所以", "我们", "你们", "他们",
    "可以", "应该", "需要", "东西", "问题", "内容", "方面", "方法",
]
STEP3_DOMAIN_TERMS = [
    "人工智能", "机器学习", "深度学习", "神经网络", "图灵测试", "亚里士多德",
]
STEP3_ENABLE_TEXT_NORMALIZATION = True
STEP3_TEXT_REPLACE_MAP = {
    "人工质能": "人工智能",
    "运遇": "机遇",
    "亚丽师多德": "亚里士多德",
    "低谱": "低谷",
    "图林": "图灵",
}

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

SEGMENT_MIN_DURATION = 20.0   # 最短保留片段（秒）
SEGMENT_PADDING      = 1.0    # 片段首尾缓冲（秒）

# ============================================================
# 训练配置
# ============================================================
TRAIN_MODEL_FILE = "boundary_model.pkl"
TRAIN_TEST_SPLIT = 0.20
RANDOM_STATE     = 42
