# Louis Use - ComfyUI Custom Nodes

A practical ComfyUI custom node pack for batch folder I/O, image utilities, color tools, seamless texture workflows, prompt conversion, performance tracking, and optional QwenVL-based image/IP screening.

> Package folder name for manual installs: `Louis_use`

## Features

- Folder batch loading and saving with preview UI
- Resolution selector with Chinese aspect-ratio labels
- Smart crop, divisible crop, padding, compositing, flip and invert tools
- Color palette extraction and color matching
- Seamless tile fixer with 1x1, 1x2, 2x1, 2x2, 3x3 and 4x4 preview modes
- Performance Tracker for generation time and VRAM metadata
- Ideogram 4 structured prompt encoder with optional QwenVL semantic conversion
- Optional text-driven segmentation/depth helpers
- Optional QwenVL image copyright/IP guard (`image -> image`)

## Installation

### ComfyUI Manager

Search for `Louis Use` in ComfyUI Manager and install it.

### Manual install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LouisLU1997/ComfyUI-louis-use Louis_use
```

Restart ComfyUI after installation.

### Dependencies

Most core nodes only need packages that ship with ComfyUI, such as `torch`, `Pillow`, `numpy`, and `aiohttp`.

For all optional nodes, install:

```bash
pip install -r requirements.txt
```

Optional heavy features may download additional models on first use:

- Text Segmenter / Text Segmented Depth: Grounding DINO, BiRefNet, Depth-Anything related models
- Ideogram4 Qwen mode / Copyright Guard: requires the separate `ComfyUI-QwenVL` custom node and a QwenVL checkpoint

## Nodes

### Folder I/O

| Node | Description |
| --- | --- |
| Folder Text Loader | Load `.txt` files from a folder as strings. |
| Folder Image Loader | Load image batches from a folder with frontend thumbnail preview. |
| Batch Image Saver | Save image batches, optional `.txt`/`.json` sidecars, overwrite toggle, metadata support. |

### Resolution and Image Utilities

| Node | Description |
| --- | --- |
| Resolution Selector | Convert aspect-ratio labels such as `16:9 横屏` to width/height. |
| Smart Align & Crop | Align and crop to a target resolution. |
| Divisible Crop | Crop to dimensions divisible by 8/16/32 or custom divisor. |
| Image Pad Color | Pad image edges with a chosen color. |
| Image Composite | Composite foreground over background with common blend modes. |
| Image Flip Horizontal | Batch-safe horizontal image flip. |
| Image Invert | RGB color inversion. |
| Reflection Extractor | Extract bright reflection/highlight regions. |

### Color Tools

| Node | Description |
| --- | --- |
| Color Palette Extractor | Extract dominant colors and output a palette visualization. |
| Color Match | Match image color statistics to a reference image. |

### Seamless Texture

| Node | Description |
| --- | --- |
| Seamless Tile Fixer | Offset-blend image into a tileable texture and output a tiled preview. |

Preview modes include `1x1`, `1x2`, `2x1`, `2x2`, `3x3`, and `4x4`.

### Text / AI Helpers

| Node | Description |
| --- | --- |
| Ideogram4 Text Encode | Convert descriptions into Ideogram 4 style structured JSON and CLIP conditioning. Optional QwenVL mode can analyze semantics and reference-image colors. |
| Show Text | Display and persist string output directly on the node. |
| Text Segmenter | Text-driven object segmentation using detection + matting models. |
| Text Segmented Depth | Text-driven target depth-map extraction. |
| Copyright Guard | QwenVL-based `image -> image` screening node. Blocks suspicious images and can cooperate with the frontend auto-requeue helper. |

### Performance

| Node | Description |
| --- | --- |
| Performance Tracker | Image passthrough node that records generation time and peak VRAM, and injects metadata for save nodes. |

## QwenVL Optional Features

`Ideogram4 Text Encode` Qwen mode and `Copyright Guard` require `ComfyUI-QwenVL`.

Install `ComfyUI-QwenVL` separately, then use a VRAM-friendly quantization option such as `4-bit (VRAM-friendly)` if needed. These nodes degrade gracefully when QwenVL is not installed, but the Qwen-powered function itself will not run.

## Publishing / Comfy Registry

This repository includes:

- `pyproject.toml` with `[tool.comfy]` metadata
- `.github/workflows/publish.yml` using `Comfy-Org/publish-node-action`

To publish to the Comfy Registry, add a GitHub secret named `REGISTRY_ACCESS_TOKEN`, then push to `main`/`master` or run the workflow manually.

## File Structure

```text
Louis_use/
├── __init__.py
├── nodes.py
├── ideogram4_text_encode.py
├── ip_copyright_guard.py
├── api.py
├── js/
│   ├── folder_io.js
│   ├── resolution_selector.js
│   ├── show_text.js
│   └── ip_copyright_guard.js
├── requirements.txt
└── pyproject.toml
```

## License

MIT
