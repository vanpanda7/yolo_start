# YOLO Game Detector

一个干净的 YOLOv8 游戏画面目标检测项目。项目只覆盖离线视频/图片的数据集制作、标注、训练、评估和文件推理，不包含安装器、插件注入、实时屏幕叠加、瞄准辅助或反作弊绕过逻辑。

## 适用范围

- 从本地录像抽帧，生成 YOLO 数据集
- 导入已有截图或图片目录
- Web 标注或 OpenCV 桌面标注
- 训练 YOLOv8 检测模型
- 在图片、目录或视频文件上离线推理
- 对验证集做 mAP 评估并保存报告

默认配置是单类 `player`。建议先把单类检测做稳定，再扩展到更多类别。

## 安装

```bash
cd /home/rick/yolo_start
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果只想先检查脚本，不需要创建虚拟环境也可以直接运行 `python tools/*.py --help`。

## 目录

```text
yolo_start/
├── config.yaml              # 项目配置
├── requirements.txt         # 最小依赖
├── tools/
│   ├── common.py            # 共享路径、配置、标签工具
│   ├── import_images.py     # 导入图片到数据集
│   ├── video_extractor.py   # 本地视频抽帧和可选预标注
│   ├── labeler.py           # OpenCV 静态图片标注
│   ├── server.py            # Web 数据集/标注后台
│   ├── train.py             # 准备 train/val 并训练
│   ├── evaluate.py          # 验证集评估
│   └── infer.py             # 离线图片/视频推理
├── datasets/                # 生成的数据集，git 忽略
├── data/videos/             # 放本地录像，git 忽略
├── data/imports/            # 放待导入图片，git 忽略
├── weights/                 # best.pt / last.pt，git 忽略
├── runs/                    # Ultralytics 训练/评估输出，git 忽略
├── exports/                 # 推理输出，git 忽略
└── reports/                 # 评估报告，git 忽略
```

## 推荐流程

### 1. 准备视频或图片

把本地录像放到：

```bash
mkdir -p data/videos
```

或把截图放到任意目录后导入：

```bash
python tools/import_images.py --source data/imports --source-name screenshots
```

### 2. 从视频抽帧

```bash
python tools/video_extractor.py --input data/videos --interval 2.0
```

默认会用 `yolov8n.pt` 的 COCO `person` 类做预标注，并保留一部分空画面作为负样本。只抽帧、不预标注：

```bash
python tools/video_extractor.py --input data/videos --no-prelabel
```

### 3. 标注和修正

启动 Web 后台：

```bash
python tools/server.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

Web 后台现在按学习路径组织：

- `Dashboard`：看当前数据集数量、训练产物状态、下一步建议和对应命令
- `Videos`：上传本地录像、选择预标注模型、启动抽帧
- `Label`：逐帧修正标注、忽略误检、按来源批量清理
- `Gallery`：浏览已经进入数据集的图片和标注框

所有 Web 操作都落到本地文件：`datasets/game_targets/images/`、`datasets/game_targets/labels/` 和 `datasets/game_targets/sources.json`。

也可以用桌面标注器：

```bash
python tools/labeler.py
```

### 4. 准备数据集并训练

```bash
python tools/train.py --prepare-only
python tools/train.py --epochs 100 --device cpu
```

`train.py` 会优先按 `sources.json` 的视频来源分组切分 train/val，减少连续帧泄漏导致的虚高验证分数。训练完成后会复制：

```text
weights/best.pt
weights/last.pt
```

### 5. 评估

```bash
python tools/evaluate.py --weights weights/best.pt
```

指标会写入：

```text
reports/evaluation.json
```

Ultralytics 的图表会放在：

```text
runs/val/evaluation/
```

### 6. 离线推理

```bash
python tools/infer.py --weights weights/best.pt --source path/to/image_or_video
```

输出默认在：

```text
exports/predict/
```

如果只是检查环境和命令，可以用预训练模型跑一张图：

```bash
python tools/infer.py --weights yolov8n.pt --source path/to/image.jpg
```

## 数据集格式

原始数据放在：

```text
datasets/game_targets/images/
datasets/game_targets/labels/
datasets/game_targets/sources.json
```

每张图片对应一个同名 `.txt` 标签文件，格式是标准 YOLO：

```text
class_id x_center y_center width height
```

坐标都是 0 到 1 的归一化值。空标签文件表示负样本。

训练前会生成：

```text
datasets/game_targets/prepared/
```

这个目录是给 Ultralytics 使用的 train/val 副本，可以随时重建。

## 配置

主要修改 [config.yaml](config.yaml)：

- `classes`：检测类别
- `paths`：数据、权重、输出目录
- `prelabel`：自动预标注模型和阈值
- `video`：抽帧间隔、去重、空样本保存频率
- `train`：训练参数
- `inference`：推理阈值
- `server`：Web 后台端口

## 注意

这个项目用于离线模型训练、分析和验证。不要把它改造成多人游戏实时辅助、自动瞄准、注入插件、反作弊绕过或类似用途。
