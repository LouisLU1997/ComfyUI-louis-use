**English | [中文](README.md)**

# Louis Use — ComfyUI Custom Nodes

A collection of practical ComfyUI custom nodes covering folder batch I/O, image cropping, color tools, seamless tiling, text-driven segmentation and more.

---

## Installation

**Option 1: ComfyUI Manager (Recommended)**  
Search for `Louis Use` in ComfyUI Manager and install with one click.

**Option 2: Manual Clone**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LouisLU1997/ComfyUI-louis-use Louis_use
# Restart ComfyUI
```

**Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## Node List

### 📂 Folder I/O

#### Folder Image Loader
Load all images from a folder in LIST mode — each image is sent individually to downstream nodes. Includes built-in thumbnail preview and native folder picker dialog.

| Parameter | Type | Description |
|-----------|------|-------------|
| `folder_path` | STRING | Path to image folder |

**Outputs:** `image` / `mask` / `filename` (one per image, downstream nodes execute per image automatically)

Supported formats: `.png .jpg .jpeg .webp .bmp .tiff .tif .gif`

---

#### Batch Image Saver
Save images to a specified folder. Can receive filenames from Folder Image Loader to preserve original names.

| Parameter | Type | Description |
|-----------|------|-------------|
| `images` | IMAGE | Image batch to save |
| `output_folder` | STRING | Output directory (absolute or relative to ComfyUI output) |
| `file_format` | Enum | `png` / `jpg` / `webp` |
| `quality` | INT | JPG/WEBP quality (1-100, ignored for PNG) |
| `embed_workflow` | BOOLEAN | Embed workflow metadata into PNG |
| `original_filename` | STRING | Optional — connect Folder Image Loader's filename to preserve original names |

---

### ✂️ Image Cropping

#### Smart Align & Crop
Align and crop images to a target resolution using anchor-based positioning (9-grid layout).

#### Divisible Crop
Crop images to dimensions divisible by a specified number. Useful for VAE alignment (8/16/32px multiples).

---

### 🎨 Color Tools

#### Color Palette Extractor
Extract dominant colors from an image and generate a proportional color swatch visualization for palette analysis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_colors` | INT | Number of colors to extract (2-20) |
| `min_ratio` | FLOAT | Minimum area ratio threshold |

**Output:** `palette_image`

---

#### Color Match
Align the color tone of an image to a reference image. Useful for fixing color drift after tile upscaling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `method` | Enum | `mean_std` / `mvgd` / `wavelet` / `histogram` |
| `strength` | FLOAT | Alignment strength (0=none, 1=full) |
| `enabled` | BOOLEAN | On/off toggle for easy comparison |

Recommended method: `wavelet` (replaces color tone while preserving detail)

---

### 🧩 Seamless Tile

#### Seamless Tile Fixer
Convert any image into a seamlessly tileable texture using the offset-blend method. Generates a tiling preview.

| Parameter | Type | Description |
|-----------|------|-------------|
| `Blend Strength` | Enum | Soft / Normal / Strong |
| `Preview Grid` | Enum | 1×1 / 2×2 / 3×3 / 4×4 |

**Outputs:** `seamless_tile` / `tiling_preview`

---

### ✂️ Text-Driven Segmentation & Depth

#### Text Segmenter
Two-stage precise segmentation: Grounding DINO text localization + BiRefNet high-quality matting.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text_prompt` | STRING | English text; separate multiple targets with `. `; leave empty for auto foreground |
| `threshold` | FLOAT | Detection threshold (0.05~0.95) |
| `resolution` | Enum | 512 / 1024 / 1536 / 2048 |
| `feather` | INT | Mask edge feathering (pixels) |

**Outputs:** `cutout` / `mask` / `preview`  
First run auto-downloads Grounding DINO Tiny (~340 MB) and BiRefNet (~1 GB).

---

#### Text Segmented Depth
Text-driven depth map: Grounding DINO localization + Depth-Anything-V2 depth estimation + BiRefNet foreground mask fusion.

**Outputs:** `depth_map` / `mask` / `preview`

---

### 🔧 Utilities

#### Reflection Extractor
Extract highlight/reflection area masks from images.

#### Image Flip Horizontal
Horizontally flip images (batch supported).

#### Timer Stop
Generation timer that displays elapsed time on the node.

---

## Dependencies

| Dependency | Purpose |
|------------|---------|
| `torch` / `Pillow` / `numpy` | Bundled with ComfyUI, core processing |
| `transformers` | Text Segmenter / Text Segmented Depth |
| `huggingface_hub` | Automatic model download |
| `torchvision` | Text Segmenter preprocessing |
| `aiohttp` | Backend API routes |

---

## License

MIT
