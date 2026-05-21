# Louis Use — ComfyUI Custom Nodes

一组实用的 ComfyUI 自定义节点，涵盖文件夹批量 I/O、图像裁剪、色彩工具、无缝贴图、文字驱动抠图等常用功能。

---

## 安装

**方法一：ComfyUI Manager（推荐）**  
在 ComfyUI Manager 中搜索 `Louis Use` 一键安装。

**方法二：手动克隆**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/comfyui-louis-use Louis_use
# 重启 ComfyUI
```

**依赖安装**
```bash
pip install -r requirements.txt
```

---

## 节点列表

### 📂 文件夹 I/O

#### Folder Image Loader
从文件夹批量加载图片，节点内置缩略图预览，支持原生文件夹选择对话框。

| 参数 | 类型 | 说明 |
|------|------|------|
| `folder_path` | STRING | 图片文件夹路径 |
| `sort_by` | 枚举 | 排序方式：`name` / `date` / `size` |
| `start_index` | INT | 从第几张开始（0-based） |
| `max_images` | INT | 最多加载几张（0 = 全部） |

**输出：** `images` / `masks` / `filenames_json` / `total_count`

支持格式：`.png .jpg .jpeg .webp .bmp .tiff .tif .gif`

---

#### Batch Image Saver
批量保存图片到指定文件夹，每次执行自动创建 `前缀_时间戳` 子文件夹。

| 参数 | 类型 | 说明 |
|------|------|------|
| `images` | IMAGE | 要保存的图片 batch |
| `output_folder` | STRING | 保存根目录 |
| `filename_prefix` | STRING | 文件名前缀 |
| `format` | 枚举 | `png` / `jpg` / `webp` |
| `quality` | INT | JPG/WEBP 质量（1-100） |

---

### ✂️ 图像裁剪

#### Smart Align & Crop
按锚点对齐并裁剪图像到目标分辨率，支持九宫格定位。

#### Divisible Crop
将图像裁剪到可被指定数整除的尺寸，常用于对齐 VAE 要求（8/16/32px 倍数）。

---

### 🎨 色彩工具

#### Color Palette Extractor
提取图片主要颜色，按面积占比生成竖条色谱可视化图，用于配色分析。

| 参数 | 类型 | 说明 |
|------|------|------|
| `num_colors` | INT | 提取颜色数量（2-20） |
| `min_ratio` | FLOAT | 最小占比阈值（过滤极少数颜色） |

**输出：** `palette_image`（色谱图）

---

#### Color Match
将图像色调对齐到参考图，用于修复分块放大后的色漂问题。

| 参数 | 类型 | 说明 |
|------|------|------|
| `method` | 枚举 | `mean_std` / `mvgd` / `wavelet` / `histogram` |
| `strength` | FLOAT | 对齐强度（0=不变，1=完全对齐） |
| `enabled` | BOOLEAN | 一键开关，方便对比 |

推荐方法：`wavelet`（只换色调，保留细节）

---

### 🧩 无缝贴图

#### Seamless Tile Fixer
将任意图像处理为可无缝平铺的贴图（偏移混合法），并生成平铺预览图。

| 参数 | 类型 | 说明 |
|------|------|------|
| `混合强度` | 枚举 | 柔和 / 标准 / 强 |
| `预览格数` | 枚举 | 1×1 / 2×2 / 3×3 / 4×4 |

**输出：** `无缝贴图` / `平铺预览`

---

### ✂️ 文字驱动抠图与深度

#### Text Segmenter
两阶段精细抠图：Grounding DINO 文字定位 + BiRefNet 高质量抠图。

| 参数 | 类型 | 说明 |
|------|------|------|
| `文字提示` | STRING | 英文，多目标用 `. ` 分隔；留空=自动抠前景 |
| `识别严格度` | FLOAT | 0.05~0.95，越高越严格 |
| `推理分辨率` | 枚举 | 512 / 1024 / 1536 / 2048 |
| `羽化半径` | INT | 遮罩边缘羽化像素数 |

**输出：** `抠图` / `遮罩` / `预览`  
首次运行自动下载 Grounding DINO Tiny（~340 MB）和 BiRefNet（~1 GB）。

---

#### Text Segmented Depth
文字驱动目标深度图：Grounding DINO 定位 + Depth-Anything-V2 深度估计 + BiRefNet 前景遮罩融合。

**输出：** `depth_map` / `mask` / `preview`

---

### 🔧 其他工具

#### Reflection Extractor
提取图像高光/反射区域遮罩。

#### Image Flip Horizontal
水平翻转图像（含 batch 支持）。

#### Timer Stop
生成耗时计时器，在节点上显示本次生成用时。

---

## 依赖

| 依赖 | 用途 |
|------|------|
| `torch` / `Pillow` / `numpy` | ComfyUI 自带，核心处理 |
| `transformers` | Text Segmenter / Text Segmented Depth |
| `huggingface_hub` | 模型自动下载 |
| `torchvision` | Text Segmenter 预处理 |
| `aiohttp` | 后端 API 路由（文件夹对话框 / 缩略图） |

---

## 文件结构

```
Louis_use/
├── __init__.py          节点注册 + API 挂载
├── nodes.py             所有节点实现
├── api.py               后端路由（文件夹对话框 / 缩略图预览 / 图片计数）
├── js/
│   └── folder_io.js     前端扩展（缩略图画布、滚轮、文件夹选择按钮）
├── requirements.txt
└── pyproject.toml
```

---

## License

MIT
