# 课程视频智能剪辑系统 — 详细操作说明

---

## 目录

1. [项目简介](#1-项目简介)
2. [系统要求](#2-系统要求)
3. [安装步骤（手把手）](#3-安装步骤手把手)
4. [目录结构](#4-目录结构)
5. [各步骤说明](#5-各步骤说明)
6. [一键运行全流程](#6-一键运行全流程)
7. [训练模型（可选）](#7-训练模型可选)
8. [常见问题](#8-常见问题)
9. [参数调节说明](#9-参数调节说明)
10. [输出文件说明](#10-输出文件说明)

---

## 1 项目简介

本系统读取 `lesson/` 文件夹中的课程视频，自动完成：

| 步骤 | 文件 | 功能 |
|------|------|------|
| 1 | `step1_visual.py` | 教师讲台检测 + PPT OCR + 翻页检测 |
| 2 | `step2_audio.py` | 语音转录 + 说话人区分 + 置信度标记 |
| 3 | `step3_text.py` | 语义分析 + 知识点边界检测 |
| 4 | `step4_align.py` | 多模态时间对齐，生成统一索引 |
| 5 | `step5_fusion.py` | 干扰片段剔除 + 视频剪辑 + 输出 MP4 |
| — | `run_all.py` | 一句指令运行全部步骤 |
| — | `train.py` | 利用人工标注提升准确率（可选） |

所有处理均在 **CPU** 上完成，完全免费，无需 GPU。

---

## 2 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10 / 11（64 位） |
| Python | 3.10（推荐）或 3.9 |
| 内存 | ≥ 8 GB |
| 硬盘 | ≥ 20 GB 可用空间（用于模型与输出视频） |
| 外部工具 | ffmpeg（免费），见安装步骤 3.2 |

---

## 3 安装步骤（手把手）

### 3.1 安装 Python 3.10

1. 浏览器打开：`https://www.python.org/downloads/release/python-3100/`
2. 下载 **Windows installer (64-bit)**
3. 双击安装，**勾选** `Add Python to PATH`，点击 Install Now
4. 安装完成后，按 `Win + R`，输入 `cmd`，回车打开命令提示符
5. 验证：
   ```
   python --version
   ```
   显示 `Python 3.10.x` 即成功

---

### 3.2 安装 ffmpeg

1. 浏览器打开：`https://www.gyan.dev/ffmpeg/builds/`
2. 下载 `ffmpeg-release-essentials.zip`（约 80 MB）
3. 解压，将解压后的文件夹改名为 `ffmpeg`，移动到 `C:\ffmpeg`
4. 将 `C:\ffmpeg\bin` 添加到系统 PATH：
   - 按 `Win + S` 搜索"环境变量"，打开"编辑系统环境变量"
   - 点击"环境变量" → 选中"Path" → 点击"编辑"
   - 点击"新建"，输入 `C:\ffmpeg\bin`，确定
5. **重新打开**一个新的 cmd 窗口，验证：
   ```
   ffmpeg -version
   ```
   显示版本信息即成功

---

### 3.3 创建项目目录结构

在 cmd 中执行：

```cmd
mkdir D:\video
mkdir D:\video\lesson
mkdir D:\video\output
mkdir D:\video\models
mkdir D:\video\annotations
```

---

### 3.4 复制项目文件

将以下文件全部复制到 `D:\video\`：

```
D:\video\
  config.py
  step1_visual.py
  step2_audio.py
  step3_text.py
  step4_align.py
  step5_fusion.py
  run_all.py
  train.py
  requirements.txt
  README_操作说明.md
```

---

### 3.5 创建虚拟环境（强烈推荐）

> **为什么需要虚拟环境？**
> 本项目依赖版本较多且较为固定（如 numpy 1.26、torch 2.2 等）。
> 使用虚拟环境可以：
> - 避免与系统其他 Python 项目的版本冲突
> - 保持系统 Python 环境干净
> - 随时删除整个 `venv` 文件夹即可完全卸载所有依赖

在 cmd 中执行：

```cmd
cd D:\video

:: 创建虚拟环境（只需执行一次）
python -m venv venv

:: 激活虚拟环境（每次打开新 cmd 窗口都需要执行这一步）
venv\Scripts\activate
```

激活成功后，命令行最左侧会显示 `(venv)` 前缀，例如：

```
(venv) D:\video>
```

> **注意**：之后所有 `python` 和 `pip` 命令都必须在激活虚拟环境后执行。
> 如果关闭了 cmd 窗口，下次重新打开后需要再次运行 `venv\Scripts\activate`。

---

### 3.6 安装 Python 依赖

确认已激活虚拟环境（命令行左侧有 `(venv)` 标记），然后执行：

```cmd
cd D:\video
venv\Scripts\activate
```

**第一步：安装 CPU 版 PyTorch**（必须先安装，约 200 MB）：

```cmd
pip install torch==2.2.2+cpu torchaudio==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
```

**第二步：安装其他依赖**：

```cmd
pip install -r requirements.txt
```

等待安装完成（约 5~10 分钟，视网速而定）。

**验证安装**：

```cmd
python -c "import cv2, faster_whisper, easyocr, jieba, sklearn; print('所有依赖安装成功！')"
```

---

### 3.7 放入视频文件

将课程视频文件（mp4/avi/mov 均可）手动复制到：

```
D:\video\lesson\
```

例如：`D:\video\lesson\高数第一章.mp4`

---

## 4 目录结构

```
D:\video\
├── lesson\                   ← 手动放入原始视频
│   ├── 高数第一章.mp4
│   └── 高数第二章.mp4
├── output\                   ← 自动生成，每个视频一个子文件夹
│   └── 高数第一章\
│       ├── visual_features.json      (步骤1输出)
│       ├── audio_features.json       (步骤2输出)
│       ├── text_features.json        (步骤3输出)
│       ├── multimodal_index.json     (步骤4输出)
│       ├── final_index.json          (步骤5输出)
│       └── segments\
│           ├── 绪论_背景.mp4
│           └── 基本概念_定义.mp4
├── models\                   ← 训练后的模型文件
├── annotations\              ← 人工标注文件（用于训练）
├── config.py
├── step1_visual.py
├── step2_audio.py
├── step3_text.py
├── step4_align.py
├── step5_fusion.py
├── run_all.py
├── train.py
└── requirements.txt
```

---

## 5 各步骤说明

所有步骤均在 `D:\video\` 目录下运行。

### 步骤 1 — 视觉分析

```cmd
cd D:\video
python step1_visual.py --video D:\video\lesson\高数第一章.mp4
```

**功能**：
- 每秒采样一帧
- 背景减除检测教师是否在讲台区域
- 仅在全屏 PPT 时检测幻灯片翻页与识别 PPT 内容（非全屏时不检测）
- 全屏 PPT 连续翻页仅记 1 次，且仅保留该全屏段最后一次 OCR 内容
- 出现全屏 PPT 时，默认教师在讲台区域内

**首次运行**：自动下载 EasyOCR 中文模型（约 50 MB）

**输出**：`output/高数第一章/visual_features.json`

---

### 步骤 2 — 语音分析

```cmd
python step2_audio.py --video D:\video\lesson\高数第一章.mp4
```

**功能**：
- ffmpeg 提取 16kHz 单声道 WAV
- faster-whisper 语音转录（带时间戳）
- 转录后文本纠错：基于专业词典的拼音模糊匹配（可配置）
- 说话人二分类（仅“教师/学生”）：融合文本课堂用语特征 + 声学稳定性特征
- 声纹辅助复判：从已判教师片段提取共同声纹，再回查学生片段并自动纠正
- 上下文平滑修正孤立误判片段
- 标记每段置信度

**首次运行**：自动下载 Whisper base 模型（约 150 MB）

**输出**：`output/高数第一章/audio_features.json`

---

### 步骤 3 — 文本语义分析

```cmd
python step3_text.py --video D:\video\lesson\高数第一章.mp4
```

**功能**：
- 文本归一化（可选 ASR 纠错词典替换）
- jieba 分词 + 自定义停用词/黑名单过滤
- TF-IDF 向量化
- 滑动窗口余弦距离检测语义跳变（知识点边界）
- 结合幻灯片切换辅助验证
- 按文档频次与权重提取关键词，自动命名知识点

**依赖**：需先完成步骤 2（步骤 1 可选但有助于提升准确率）

**输出**：`output/高数第一章/text_features.json`

---

### 步骤 4 — 多模态对齐

```cmd
python step4_align.py --video D:\video\lesson\高数第一章.mp4
```

**功能**：
- 以 1 秒为单位建立统一时间轴
- 每个时刻点包含：教师在场/幻灯片/语音文本/说话人/置信度/知识点标签

**依赖**：需先完成步骤 2 和步骤 3

**输出**：`output/高数第一章/multimodal_index.json`

---

### 步骤 5 — 融合与剪辑

```cmd
python step5_fusion.py --video D:\video\lesson\高数第一章.mp4
```

**功能**：
- 根据规则 + 训练模型辅助检测干扰片段（动态阈值）
- 生成剪辑指令并调用 ffmpeg 剪切
- 输出以知识点命名的独立 MP4
- 不输出干扰片段视频，仅在 `final_index.json` 中统一记录为“干扰片段”

**依赖**：需先完成步骤 4

**输出**：
- `output/高数第一章/segments/*.mp4`
- `output/高数第一章/final_index.json`

---

## 6 一键运行全流程

### 处理 lesson/ 下所有视频

```cmd
cd D:\video
venv\Scripts\activate
python run_all.py
```

### 只处理单个视频

```cmd
cd D:\video
venv\Scripts\activate
python run_all.py --video D:\video\lesson\高数第一章.mp4
```

### 从某步骤开始（例如视频已处理到步骤2，继续步骤3-5）

```cmd
python run_all.py --video D:\video\lesson\高数第一章.mp4 --start 3
```

### 只重新运行某一步

```cmd
python run_all.py --video D:\video\lesson\高数第一章.mp4 --step 5
```

---

## 7 训练模型（可选）

训练可提升知识点边界检测和干扰检测的准确率。

### 7.1 准备标注文件

在 `D:\video\annotations\` 中创建标注文件，文件名与视频同名，例如
`高数第一章_annotation.json`：

```json
{
  "video": "高数第一章.mp4",
  "annotations": [
    {
      "start": 0.0,
      "end": 180.0,
      "title": "课程介绍",
      "is_interference": false
    },
    {
      "start": 205.0,
      "end": 480.0,
      "title": "第一节基本概念",
      "is_interference": false
    },
    {
      "start": 480.0,
      "end": 720.0,
      "title": "第二节极限定义",
      "is_interference": false
    }
  ]
}
```

字段说明：
- `start` / `end`：片段时间戳（秒）
- `title`：知识点名称
- `is_interference`：可选；`true` = 显式干扰片段

> 训练已支持“仅标注知识点片段”：
> - 仅标知识点即可训练边界模型
> - 知识点外时间自动视为干扰候选（用于干扰模型辅助打分）

### 7.2 训练

```cmd
cd D:\video
python train.py --video D:\video\lesson\高数第一章.mp4 ^
                --annotation D:\video\annotations\高数第一章_annotation.json
```

> **注意**：`^` 是 cmd 中的换行符，可以写成一行

### 7.3 批量训练多个视频

```cmd
python train.py --annotation_dir D:\video\annotations\
```

系统会自动在 `lesson/` 目录下寻找同名视频进行配对。

### 7.4 评估模型效果

```cmd
python train.py --eval ^
  --video D:\video\lesson\高数第一章.mp4 ^
  --annotation D:\video\annotations\高数第一章_annotation.json
```

模型文件保存在 `D:\video\models\`：
- `boundary_model.pkl` — 知识点边界检测模型
- `interference_model.pkl` — 干扰片段检测模型

训练完成后，再次运行 `run_all.py` 时将自动使用这些模型。

---

## 8 常见问题

### Q0: 忘记激活虚拟环境，`import` 报错找不到模块
**解决**：每次打开新的 cmd 窗口后，必须先执行：
```cmd
cd D:\video
venv\Scripts\activate
```
命令行左侧出现 `(venv)` 后再运行 `python` 命令。

### Q1: 运行时提示 "ffmpeg 不是内部或外部命令"
**解决**：检查 PATH 设置，参见安装步骤 3.2。确保**重新打开**了 cmd 窗口。

### Q2: EasyOCR / Whisper 模型下载失败
**解决**：
- 检查网络连接
- 可手动下载后放置到对应目录（EasyOCR 模型默认在 `~/.EasyOCR/`，Whisper 模型默认在 `~/.cache/huggingface/`）
- 或使用代理：`set HTTPS_PROXY=http://你的代理地址:端口`

### Q3: OCR 识别中文效果差
**解决**：
- 确保视频中 PPT 文字清晰，分辨率 ≥ 720p
- 调整 `config.py` 中的 `PPT_REGION`，使其精准覆盖 PPT 区域
- 调低 `OCR_CONFIDENCE_THRESHOLD`（如改为 0.3）

### Q4: 教师检测不准确（总说不在讲台）
**解决**：
- 调整 `config.py` 中的 `PODIUM_REGION`，根据实际视频布局修改
- 调低 `TEACHER_PRESENCE_THRESHOLD`（如改为 0.02）
- 确认 `TEACHER_STATIC_COMPENSATION_ENABLED=True`，开启静止补偿人体检测
- 确保视频前几秒有空镜头用于初始化背景

### Q5: 知识点切分太多/太少
**解决**：
- 太多：增大 `BOUNDARY_THRESHOLD`（如 0.45）或 `MIN_KNOWLEDGE_DURATION`（如 90）
- 太少：减小 `BOUNDARY_THRESHOLD`（如 0.25）或 `SEMANTIC_WINDOW_SIZE`（如 2）

### Q6: 内存不足
**解决**：
- 使用更小的 Whisper 模型：将 `config.py` 中 `WHISPER_MODEL_SIZE` 改为 `"tiny"`
- 降低分析帧率：将 `VISUAL_SAMPLE_FPS` 改为 `0.5`（每 2 秒采样一帧）

### Q7: 处理速度太慢
**解决**：
- 将 `WHISPER_MODEL_SIZE` 改为 `"tiny"`（速度提升 ~3x，准确率略降）
- 将 `VISUAL_SAMPLE_FPS` 改为 `0.5`
- 步骤 1 可跳过（直接从步骤 2 开始）：`python run_all.py --start 2`

### Q8: 输出视频文件损坏或无法播放
**解决**：
- 检查原始视频是否完整
- 步骤 5 会自动在 stream copy 失败时切换为重新编码模式
- 手动测试：`ffmpeg -i D:\video\lesson\example.mp4 -t 5 D:\video\test_out.mp4`

---

## 9 参数调节说明

所有参数集中在 `D:\video\config.py` 中，**无需修改其他文件**。

| 参数 | 默认值 | 说明 | 调节建议 |
|------|--------|------|----------|
| `VISUAL_SAMPLE_FPS` | 1 | 每秒采样帧数 | 降低可加快速度 |
| `PODIUM_REGION` | (0.38,0.42,0.66,0.94) | 讲台区域比例 | 先框住讲台与教师活动区，尽量排除前排学生 |
| `PPT_REGION` | (0.02,0.02,0.98,0.8) | PPT 区域比例 | 建议覆盖上方屏幕，避免下方听众 |
| `PPT_REGION_FULLSCREEN` | (0,0,1,0.9) | 全屏 PPT 检测区域 | 建议覆盖投影主画面 |
| `SLIDE_CHANGE_THRESHOLD` | 0.70 | SSIM 翻页阈值 | 越小越灵敏 |
| `FULLSCREEN_BRIGHT_RATIO` | 0.35 | 全屏 PPT 亮部阈值 | 光线偏暗时可下调 |
| `FULLSCREEN_LOW_SAT_RATIO` | 0.45 | 全屏 PPT 低饱和阈值 | 画面偏彩色时可下调 |
| `FULLSCREEN_EDGE_RATIO` | 0.005 | 全屏 PPT 边缘密度阈值 | 文字较少时可下调 |
| `TEACHER_PRESENCE_THRESHOLD` | 0.02 | 运动检测阈值 | 减小可提高静态场景检出率 |
| `TEACHER_STATIC_COMPENSATION_ENABLED` | True | 是否启用静止补偿（人体检测兜底） | 讲台静止授课场景建议开启 |
| `TEACHER_STATIC_DETECT_INTERVAL` | 2 | 静止补偿检测间隔（采样点） | 变大可提速，变小更灵敏 |
| `TEACHER_STATIC_MIN_WEIGHT` | 0.20 | 人体检测最小权重 | 增大可减少误检 |
| `TEACHER_STATIC_MIN_AREA_RATIO` | 0.015 | 人体框最小面积占比 | 增大可减少远处误检 |
| `WHISPER_MODEL_SIZE` | "base" | Whisper 模型 | tiny/base/small/medium |
| `DIARIZATION_N_CLUSTERS` | 2 | 说话人聚类类别数 | 课堂场景建议固定 2（教师/学生） |
| `DIARIZATION_TEXT_WEIGHT` / `DIARIZATION_ACOUSTIC_WEIGHT` | 0.38 / 0.62 | 文本/声学融合权重 | 课堂讲授场景建议以声学为主，文本作辅助 |
| `DIARIZATION_SMOOTH_MAX_DURATION` | 4.0 | 孤立短片段平滑时长上限（秒） | 增大可减少抖动，过大可能过平滑 |
| `DIARIZATION_VOICEPRINT_SIMILARITY_THRESHOLD` | 0.82 | 声纹回标阈值 | 降低可更激进地把学生改判为教师 |
| `DIARIZATION_VOICEPRINT_MIN_TEACHER_SAMPLES` | 2 | 构建教师声纹原型的最少教师片段数 | 样本少时可先用较低值快速启动复判，再按误判情况回调 |
| `STEP2_ENABLE_TEXT_CORRECTION` | True | 是否启用步骤2文本纠错 | ASR 错别词较多时建议开启 |
| `STEP2_TEXT_CORRECTION_TERMS` | 见配置 | 专业术语词典 | 建议按学科持续补充 |
| `STEP2_TEXT_CORRECTION_MAX_PINYIN_NORM_DIST` | 0.22 | 拼音归一化编辑距离阈值 | 越小越保守 |
| `STEP2_TEXT_CORRECTION_MAX_CHAR_DIST` | 1 | 中文字符编辑距离上限 | 越小越保守 |
| `STEP2_TEXT_CORRECTION_MAX_LENGTH_DIFF` | 1 | 术语与候选词长度差上限 | 越小越保守 |
| `STEP2_TEXT_CORRECTION_CHAR_WEIGHT` | 0.05 | 字符编辑距离在综合评分中的权重 | 增大则更重视字形接近 |
| `NO_SPEECH_PROB_THRESHOLD` | 0.80 | no_speech_prob 静默阈值 | 建议课堂场景保持较高值，减少误删有声文本 |
| `NO_SPEECH_IGNORE_WITH_TEXT` | True | 高 no_speech_prob 时是否启用文本兜底 | 建议开启，避免 Whisper 误判导致文本丢失 |
| `NO_SPEECH_TEXT_SHORT_LEN` | 3 | 文本兜底最短长度（字符） | 极短文本仍可过滤噪声，建议 2~4 |
| `BOUNDARY_THRESHOLD` | 0.35 | 语义边界阈值 | 越大切分越少 |
| `MIN_KNOWLEDGE_DURATION` | 45 | 最短知识点（秒）| 增大可避免过度切分 |
| `MAX_KNOWLEDGE_DURATION` | 600 | 最长知识点（秒）| 增大可容纳长讲解 |
| `KEYWORD_TITLE_COUNT` | 2 | 标题拼接关键词数量 | 一般保持 2，过大易冗长 |
| `KEYWORD_MIN_DOC_FREQ` | 2 | 关键词最小文档频次 | 调大可抑制偶发噪声词 |
| `KEYWORD_BLACKLIST` | 见配置 | 关键词黑名单 | 可加入口语废词 |
| `STEP3_DOMAIN_TERMS` | 见配置 | 领域词典 | 加入课程专业术语可提升分词质量 |
| `STEP3_ENABLE_TEXT_NORMALIZATION` | True | 是否启用文本纠错替换 | ASR 错别词较多时建议开启 |
| `STEP3_TEXT_REPLACE_MAP` | 见配置 | ASR 常见误识别替换表 | 按课程场景持续补充 |
| `INTERFERENCE_TEACHER_ABSENT_RATIO` | 0.70 | 教师缺席干扰阈值 | 增大可宽容更多缺席 |
| `INTERFERENCE_SILENCE_THRESHOLD` | 15 | 连续静默判干扰（秒）| 增大忽略短休息 |
| `SEGMENT_MIN_DURATION` | 20 | 最短输出片段（秒）| 减小保留短内容 |

---

## 10 输出文件说明

### `visual_features.json`
```json
{
  "fps": 25.0,
  "duration": 3600.0,
  "teacher_timeline": [
    {"time": 0.0, "in_podium": true, "motion_ratio": 0.12, "static_person_detected": false, "full_screen_ppt": false}
  ],
  "slide_transitions": [
    {"time": 45.0, "ssim": 0.42, "slide_idx": 1}
  ],
  "ppt_content": [
    {"time": 45.0, "slide_idx": 1, "text": "第一章 绪论"}
  ]
}
```

### `audio_features.json`
```json
{
  "language": "zh",
  "segments": [
    {
      "start": 0.0, "end": 5.2,
      "text": "大家好，今天我们来学习...",
      "confidence": 0.92,
      "speaker": "教师",
      "is_low_confidence": false
    }
  ]
}
```

### `text_features.json`
```json
{
  "knowledge_segments": [
    {
      "id": 0, "start": 0.0, "end": 240.0,
      "title": "极限_定义",
      "keywords": ["极限", "定义", "收敛"]
    }
  ],
  "boundaries": [0.0, 240.0, 480.0]
}
```

### `multimodal_index.json`
时间序列中每秒一个点，包含所有模态的对齐特征。

### `final_index.json`
```json
{
  "total_clips": 5,
  "clips": [
    {
      "id": 0,
      "title": "极限_定义",
      "start": 0.0, "end": 238.0,
      "output_file": "D:\\video\\output\\xxx\\segments\\极限_定义.mp4",
      "keywords": ["极限", "定义"]
    }
  ],
  "removed_segments": [
    {"title": "干扰片段", "start": 238.0, "end": 255.0, "reasons": ["静默占比 95%"], "output_policy": "not_exported"}
  ]
}
```

---

## 快速参考命令

```cmd
cd D:\video

:: 每次打开新 cmd 窗口时，先激活虚拟环境
venv\Scripts\activate

:: 一键全流程（推荐）
python run_all.py

:: 只处理单个视频全流程
python run_all.py --video D:\video\lesson\example.mp4

:: 分步运行
python step1_visual.py --video D:\video\lesson\example.mp4
python step2_audio.py  --video D:\video\lesson\example.mp4
python step3_text.py   --video D:\video\lesson\example.mp4
python step4_align.py  --video D:\video\lesson\example.mp4
python step5_fusion.py --video D:\video\lesson\example.mp4

:: 训练
python train.py --video D:\video\lesson\example.mp4 --annotation D:\video\annotations\example_annotation.json
```
