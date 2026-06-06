# YOLO Start —— 从零搭建游戏视觉 AI 训练流水线

一个完整的 YOLO 目标检测训练工具链，专为 FPS 游戏画面设计。包含屏幕捕获、AI 辅助标注、数据集管理、模型训练和实时推理。

> ⚠️ **教育目的**：本项目用于学习计算机视觉、目标检测和深度学习。严禁用于任何破坏游戏公平性的行为。

## 功能

| 工具 | 用途 |
|---|---|
| `tools/labeler.py` | OpenCV 桌面标注工具，AI 预标注 + 手动修正 |
| `tools/server.py` | Flask Web 后台，浏览器标注 + 数据集管理 |
| `tools/video_extractor.py` | 视频抽帧，自动 AI 预标注 + 感知哈希去重 |
| `tools/train.py` | YOLOv8 训练脚本，小数据集自动防过拟合 |
| `tools/infer.py` | 实时推理，ESP 风格可视化 + 自动数据收集 |
| `tools/screen_capture.py` | 屏幕捕获模块（BetterCam + mss 双后端） |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt
pip install -r tools/requirements.txt

# 启动 Web 后台
python tools/server.py
# 浏览器打开 http://127.0.0.1:5000
```

## 工作流

```
视频 → video_extractor.py → AI 预标注
         ↓
    Web 标注工作台 → 手动修正 / 批量忽略
         ↓
    train.py → 训练 YOLO 模型
         ↓
    infer.py → 实时推理
         ↓
    继续收集 → 再训练 → 越用越准
```

## 目录结构

```
yolo_start/
├── tools/                  # 核心工具链
│   ├── config.yaml         # 统一配置
│   ├── screen_capture.py   # 屏幕捕获
│   ├── labeler.py          # 桌面标注工具
│   ├── video_extractor.py  # 视频抽帧
│   ├── train.py            # YOLO 训练
│   ├── infer.py            # 实时推理
│   ├── server.py           # Web 后台
│   ├── templates/          # 前端页面
│   └── static/             # 样式文件
├── models/                 # YOLO 模型脚手架（教学用）
├── core/                   # 工具模块
├── data/                   # 资源文件
└── requirements.txt        # Python 依赖
```

## 技术栈

- **Python 3.10+**, Windows
- **YOLOv8** (Ultralytics) — 目标检测
- **OpenCV** — 图像处理
- **Flask** — Web 后台
- **BetterCam / mss** — 屏幕捕获
- **PyTorch** — 深度学习框架

## 许可证

MIT License — 仅供学习使用
