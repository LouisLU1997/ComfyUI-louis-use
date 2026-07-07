"""
Ideogram 4 Text Encode — Louis_use

输入普通文本，输出 Ideogram 4 所需的 JSON 格式字符串（及可选的 CONDITIONING）。

JSON 结构：
{
  "aspect_ratio": "16:9",
  "high_level_description": "...",
  "style_description": { "aesthetics": "...", "lighting": "...", ... },
  "compositional_deconstruction": { "background": "...", "elements": [...] }
}

模式：
  Template — 文本直接放入 high_level_description，即开即用，无需额外模型
  Qwen     — 本地 Qwen3-VL-4B-Instruct 语义分析，自动填充所有字段（需安装 ComfyUI-QwenVL）
"""

import json
import re
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# 颜色提取（纯 numpy + PIL，无需额外依赖）
# ──────────────────────────────────────────────────────────────────────────────

def _kmeans_hex(pixels: "np.ndarray", n: int) -> list[str]:
    """对 shape=(N,3) float 像素数组做 k-means，返回按频率排序的 hex 列表。"""
    import numpy as np
    n = min(n, len(pixels))
    rng = np.random.default_rng(42)
    centers = pixels[rng.choice(len(pixels), n, replace=False)].copy()
    labels = np.zeros(len(pixels), dtype=int)
    for _ in range(20):
        dists  = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
        labels = np.argmin(dists, axis=1)
        new_c  = np.array([
            pixels[labels == i].mean(axis=0) if (labels == i).any() else centers[i]
            for i in range(n)
        ])
        if np.allclose(centers, new_c, atol=1):
            break
        centers = new_c
    counts = [(labels == i).sum() for i in range(n)]
    order  = sorted(range(n), key=lambda x: -counts[x])
    return [f"#{int(centers[i][0]):02X}{int(centers[i][1]):02X}{int(centers[i][2]):02X}"
            for i in order]


def _img_to_array(image_tensor) -> "np.ndarray | None":
    """ComfyUI IMAGE tensor → uint8 numpy H×W×3 array（第一帧）。"""
    if image_tensor is None:
        return None
    import numpy as np
    return (image_tensor[0].cpu().float().numpy() * 255).clip(0, 255).astype("uint8")


def _extract_palette(img_arr: "np.ndarray", n: int = 6) -> list[str]:
    """全图提取 n 个主色调。img_arr: H×W×3 uint8。"""
    try:
        import numpy as np
        from PIL import Image
        pil    = Image.fromarray(img_arr).resize((120, 120), Image.LANCZOS)
        pixels = np.array(pil).reshape(-1, 3).astype(float)
        return _kmeans_hex(pixels, n)
    except Exception as e:
        print(f"[Ideogram4TextEncode] 全图颜色提取失败: {e}")
        return []


def _extract_palette_region(img_arr: "np.ndarray", bbox: list, n: int = 3) -> list[str]:
    """按 bbox [y_min, x_min, y_max, x_max]（0-1000 归一化）裁剪区域提取颜色。"""
    try:
        import numpy as np
        H, W = img_arr.shape[:2]
        y0 = max(0, int(bbox[0] / 1000 * H))
        x0 = max(0, int(bbox[1] / 1000 * W))
        y1 = min(H, int(bbox[2] / 1000 * H))
        x1 = min(W, int(bbox[3] / 1000 * W))
        if y1 - y0 < 4 or x1 - x0 < 4:
            return []
        pixels = img_arr[y0:y1, x0:x1].reshape(-1, 3).astype(float)
        return _kmeans_hex(pixels, n)
    except Exception as e:
        print(f"[Ideogram4TextEncode] 区域颜色提取失败 bbox={bbox}: {e}")
        return []


def _detect_subject_bbox(img_arr: "np.ndarray") -> list[int]:
    """
    用边缘密度从参考图检测主要主体的 bbox。
    保持宽高比缩放（不压成正方形），分别按实际 W/H 归一化到 0-1000。
    返回 [y_min, x_min, y_max, x_max]，0-1000 归一化。
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter

        orig_H, orig_W = img_arr.shape[:2]
        # 保持宽高比，长边缩到 96px
        scale  = 96 / max(orig_H, orig_W)
        new_W  = max(1, int(orig_W * scale))
        new_H  = max(1, int(orig_H * scale))

        pil   = Image.fromarray(img_arr).resize((new_W, new_H), Image.LANCZOS)
        gray  = pil.convert("L")
        edges = np.array(gray.filter(ImageFilter.FIND_EDGES)).astype(float)

        # 天空区域（顶部 25%）降权
        sky_end = max(1, new_H // 4)
        edges[:sky_end, :] *= 0.2

        row_sum = edges.sum(axis=1)
        col_sum = edges.sum(axis=0)

        thresh_r = row_sum.max() * 0.35
        thresh_c = col_sum.max() * 0.35

        rows = np.where(row_sum > thresh_r)[0]
        cols = np.where(col_sum > thresh_c)[0]

        if len(rows) < 3 or len(cols) < 3:
            return [250, 150, 800, 850]

        pad = 40
        # 按各自实际维度归一化，避免宽高比不一致导致偏移
        y0 = max(0,    int(rows[0]  / new_H * 1000) - pad)
        y1 = min(1000, int(rows[-1] / new_H * 1000) + pad)
        x0 = max(0,    int(cols[0]  / new_W * 1000) - pad)
        x1 = min(1000, int(cols[-1] / new_W * 1000) + pad)

        # 打印实际参考图像素位置方便对比
        py0 = int(y0 / 1000 * orig_H)
        py1 = int(y1 / 1000 * orig_H)
        px0 = int(x0 / 1000 * orig_W)
        px1 = int(x1 / 1000 * orig_W)
        print(f"[Ideogram4TextEncode] 主体检测 归一化[{y0},{x0},{y1},{x1}]"
              f" | 参考图像素 y={py0}~{py1}/{orig_H}, x={px0}~{px1}/{orig_W}")
        return [y0, x0, y1, x1]
    except Exception as e:
        print(f"[Ideogram4TextEncode] 主体检测失败: {e}")
        return [250, 150, 800, 850]


# 被视为"主体"的 element type（第一个匹配的会被替换 bbox）
_SUBJECT_TYPES = {"car", "vehicle", "person", "human", "figure", "obj", "object", "subject"}


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_tone(r, g, b) -> str:
    """把 RGB 转成简单的色调描述词，辅助 Ideogram 理解颜色意图。"""
    brightness = (r + g + b) / 3
    if brightness < 40:
        return "near-black"
    if brightness > 220:
        return "near-white"
    if r > g + 30 and r > b + 30:
        return "warm red" if brightness < 120 else "warm golden"
    if g > r + 20 and g > b + 20:
        return "cool green"
    if b > r + 30 and b > g + 20:
        return "deep blue"
    if r > b + 20 and g > b + 20:
        return "warm amber" if brightness < 150 else "soft gold"
    if abs(r - g) < 20 and abs(g - b) < 20:
        return "cool gray" if b >= r else "warm gray"
    return "muted tone"


def _apply_image_colors(parsed: dict, img_arr: "np.ndarray") -> dict:
    """
    用参考图的真实颜色覆盖所有 color_palette 字段，
    并将主体 element 的 bbox 替换为图像检测结果。
      - style_description.color_palette  → 全图 6 色
      - style_description.color_tones    → 颜色语言描述（增强 Ideogram 理解）
      - 主体 element 的 bbox             → 边缘检测结果
      - 每个 element.color_palette       → 对应 bbox 区域 3 色
    """
    # 全图主色调 → style_description
    global_pal = _extract_palette(img_arr, n=6)
    if global_pal:
        sd = parsed.setdefault("style_description", {})
        sd["color_palette"] = global_pal
        # 附加色调语言描述，增强 Ideogram 颜色理解
        try:
            tones = [_rgb_to_tone(*_hex_to_rgb(h)) for h in global_pal]
            sd["color_tones"] = ", ".join(dict.fromkeys(tones))  # 去重保序
        except Exception:
            pass
        print(f"[Ideogram4TextEncode] 全图主色: {global_pal}")

    # 检测主体 bbox
    subject_bbox = _detect_subject_bbox(img_arr)

    elements = parsed.get("compositional_deconstruction", {}).get("elements", [])
    subject_replaced = False
    for elem in elements:
        etype = str(elem.get("type", "")).lower()

        # 主体元素替换 bbox
        if not subject_replaced and any(t in etype for t in _SUBJECT_TYPES):
            elem["bbox"] = subject_bbox
            subject_replaced = True
            print(f"[Ideogram4TextEncode] 主体 '{etype}' bbox → {subject_bbox}")

        # 按（更新后的）bbox 提取区域颜色
        bbox = elem.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            region_pal = _extract_palette_region(img_arr, bbox, n=3)
            if region_pal:
                elem["color_palette"] = region_pal
                print(f"[Ideogram4TextEncode]   {etype} {bbox} → {region_pal}")

    return parsed

# ──────────────────────────────────────────────────────────────────────────────
# 宽高比
# ──────────────────────────────────────────────────────────────────────────────

_ASPECT_RATIOS = [
    "不使用",
    # 方形
    "1:1 正方形",
    # 横版
    "16:9 横屏", "16:10 横屏", "4:3 横屏", "3:2 横屏", "5:4 横屏",
    "2:1 超宽",  "21:9 超宽",  "3:1 超宽",
    # 竖版
    "9:16 竖屏", "10:16 竖屏", "3:4 竖屏", "2:3 竖屏", "4:5 竖屏",
    "1:2 竖屏",  "9:21 超高",  "1:3 超高",
]


def _ratio_key(ar: str) -> str:
    """从 '16:9 横屏' 提取纯比例字符串 '16:9'，兼容已是纯比例的输入"""
    import re
    m = re.match(r'(\d+:\d+)', ar.strip())
    return m.group(1) if m else ar.strip()

# ──────────────────────────────────────────────────────────────────────────────
# Qwen 系统提示词
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert at converting image descriptions into structured JSON prompts for Ideogram 4.

Convert the given description into this exact JSON structure:
{
  "high_level_description": "CONDENSED 1-3 sentence summary — DO NOT copy the input, rewrite concisely capturing subject, mood, setting and camera angle",
  "style_description": {
    "aesthetics": "comma-separated style/art movement keywords",
    "lighting": "lighting technique and quality",
    "photo": "resolution and aspect ratio info",
    "medium": "photorealistic / digital art / oil painting / etc.",
    "color_palette": ["#hex1", "#hex2", "#hex3"]
  },
  "compositional_deconstruction": {
    "background": "detailed background environment description",
    "elements": [
      {
        "type": "car / person / obj / plant / etc.",
        "bbox": [y_min, x_min, y_max, x_max],
        "desc": "detailed description of this element",
        "color_palette": ["#hex1", "#hex2"]
      }
    ]
  }
}

Rules:
- Add aspect_ratio as top-level field ONLY if explicitly instructed
- bbox: normalized 0-1000, format [y_min, x_min, y_max, x_max]
- color_palette: 3-5 dominant hex colors inferred from the scene
- elements: 3-5 entries covering ALL major visual layers — always include:
    1. The primary subject (car / person / object)
    2. The ground / floor / terrain / road surface (type="ground") — NEVER skip this
    3. Sky or ceiling if present (type="sky")
    4. Any other significant mid-ground or atmospheric element (mist, water, foliage, etc.)
- For type="ground": desc must capture surface material, texture, atmospheric effects (mist, wet sheen, motion blur, vapor, puddles, dust, etc.)
- photo field: leave it empty string "" — aspect ratio and resolution will be filled in automatically
- Return ONLY valid JSON, no markdown, no extra text"""


# ──────────────────────────────────────────────────────────────────────────────
# Template 转换
# ──────────────────────────────────────────────────────────────────────────────

def _template_convert(text: str, aspect_ratio: str) -> str:
    prompt = {}
    if aspect_ratio and aspect_ratio != "不使用":
        prompt["aspect_ratio"] = _ratio_key(aspect_ratio)
    prompt["high_level_description"] = text.strip()
    prompt["style_description"] = {
        "aesthetics": "", "lighting": "", "photo": "",
        "medium": "photorealistic", "color_palette": [],
    }
    prompt["compositional_deconstruction"] = {"background": "", "elements": []}
    return json.dumps(prompt, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Qwen 转换
# ──────────────────────────────────────────────────────────────────────────────

_qwen_instance = None


def _get_qwen():
    global _qwen_instance
    if _qwen_instance is not None:
        return _qwen_instance
    try:
        from AILab_QwenVL import QwenVLBase
        _qwen_instance = QwenVLBase()
        return _qwen_instance
    except ImportError:
        import folder_paths
        cn_path = Path(folder_paths.base_path) / "custom_nodes" / "ComfyUI-QwenVL"
        if cn_path.exists() and str(cn_path) not in sys.path:
            sys.path.insert(0, str(cn_path))
        try:
            from AILab_QwenVL import QwenVLBase
            _qwen_instance = QwenVLBase()
            return _qwen_instance
        except Exception as e:
            print(f"[Ideogram4TextEncode] ❌ ComfyUI-QwenVL 不可用: {e}")
            return None


def _extract_json(raw: str) -> str:
    raw = re.sub(r'```(?:json)?\s*', '', raw).strip().strip('`').strip()
    s, e = raw.find('{'), raw.rfind('}')
    return raw[s:e + 1] if s != -1 and e > s else raw


_QUANTIZATIONS = ["4-bit (VRAM-friendly)", "8-bit (Balanced)", "None (FP16)"]


def _qwen_convert(text: str, aspect_ratio: str, max_tokens: int,
                  quantization: str = "4-bit (VRAM-friendly)", image=None) -> str | None:
    qwen = _get_qwen()
    if qwen is None:
        print("[Ideogram4TextEncode] ❌ Qwen 实例获取失败，请确认 ComfyUI-QwenVL 已安装")
        return None

    ar_hint = (f'Use aspect_ratio: "{_ratio_key(aspect_ratio)}"\n\n'
               if aspect_ratio and aspect_ratio != "不使用"
               else "Do NOT include aspect_ratio field.\n\n")

    # 有参考图时：预先提取全图主色，注入提示词让 Qwen 知道整体色调
    img_arr = _img_to_array(image)
    color_hint = ""
    if img_arr is not None:
        global_pal = _extract_palette(img_arr, n=6)
        if global_pal:
            pal_str   = ", ".join(global_pal)
            color_hint = (
                f"The reference image's overall dominant colors (pre-extracted): {pal_str}\n"
                "Use these as the base palette. For each element's color_palette, "
                "pick the most relevant subset from this list.\n\n"
            )
            print(f"[Ideogram4TextEncode] 全图主色: {pal_str}")

    prompt = f"{_SYSTEM_PROMPT}\n\n{ar_hint}{color_hint}Description to convert:\n{text.strip()}"

    print(f"[Ideogram4TextEncode] 使用量化模式: {quantization}，{'有参考图' if img_arr is not None else '无参考图'}")
    try:
        result = qwen.run(
            model_name="Qwen3-VL-4B-Instruct",
            quantization=quantization,
            preset_prompt="",
            custom_prompt=prompt,
            image=image, video=None, frame_count=1,
            max_tokens=max_tokens,
            temperature=0.3, top_p=0.9,
            num_beams=1, repetition_penalty=1.1,
            seed=42, keep_model_loaded=False,
            attention_mode="sdpa", use_torch_compile=False, device="auto",
        )
        # Qwen 卸载后主动释放 CUDA 缓存，避免和后续 CLIP 模型抢显存
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("[Ideogram4TextEncode] CUDA 缓存已清理")
        except Exception:
            pass
        raw = result[0] if result else ""
        print(f"[Ideogram4TextEncode] Qwen 输出（前200字）: {raw[:200]}")
        parsed = json.loads(_extract_json(raw))

        # 后处理：按 bbox 区域精确提取各元素颜色，覆盖模型猜测的色值
        if img_arr is not None:
            parsed = _apply_image_colors(parsed, img_arr)

        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Ideogram4TextEncode] ❌ Qwen 转换失败 {type(e).__name__}: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 节点
# ──────────────────────────────────────────────────────────────────────────────

class Louis_use_Ideogram4TextEncode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "",
                    "tooltip": "直接输入提示词，或从外部节点连线输入",
                }),
                "aspect_ratio": (_ASPECT_RATIOS, {
                    "default": "16:9",
                    "tooltip": "宽高比，输出到 Resolution Selector 统一控制分辨率",
                }),
                "mode": (["Template（直接包装）", "Qwen（语义分析）"], {
                    "default": "Qwen（语义分析）",
                    "tooltip": "Template：文本直接放入 JSON，无需模型；Qwen：本地 LLM 分析填充所有字段",
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "接 Ideogram 4 CLIP 编码器，输出 conditioning 用于采样",
                }),
                "ref_image": ("IMAGE", {
                    "tooltip": "颜色参考图（可选）。接入后 Qwen 将从图像中提取真实主色调，覆盖纯文字推断的颜色",
                }),
                "max_tokens": ("INT", {
                    "default": 1200, "min": 400, "max": 2000, "step": 50,
                    "tooltip": "Qwen 最大输出 token 数，建议 1200 以上确保所有字段都能生成完整（Template 模式忽略）",
                }),
                "quantization": (_QUANTIZATIONS, {
                    "default": "4-bit (VRAM-friendly)",
                    "tooltip": "Qwen 模型量化精度。int4 显存最省，FP16 精度最高但 Windows 下容易崩溃",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, aspect_ratio, **kwargs):
        # 兼容旧工作流保存的纯数字比例（如 "16:9"），不报 "Value not in list"
        return True

    RETURN_TYPES  = ("CONDITIONING", "STRING", "STRING", "STRING")
    RETURN_NAMES  = ("conditioning", "json_text", "text", "aspect_ratio")
    FUNCTION      = "encode"
    CATEGORY      = "Louis_use/ideogram4"
    DESCRIPTION   = "将提示词转换为 Ideogram 4 JSON 格式；可选接 CLIP 输出 conditioning"

    def encode(self, text, aspect_ratio, mode, clip=None, ref_image=None,
               max_tokens=1200, quantization="4-bit (VRAM-friendly)"):
        # ── 兼容旧工作流 widget 错位：mode/max_tokens 可能收到整数或非预期字符串 ──
        if not isinstance(mode, str) or not any(k in mode for k in ("Qwen", "Template")):
            mode = "Qwen（语义分析）"
        if not isinstance(max_tokens, int) or max_tokens < 100:
            max_tokens = 1200

        # ── 转换 ──────────────────────────────────────────────────────────────
        if "Qwen" in mode:
            has_img = ref_image is not None
            print(f"[Ideogram4TextEncode] Qwen 模式，文字长度 {len(text)} 字符，{'有参考图' if has_img else '无参考图'}...")
            json_text = _qwen_convert(text, aspect_ratio, max_tokens, quantization, ref_image)
            if json_text:
                print("[Ideogram4TextEncode] ✓ Qwen 转换成功")
            else:
                print("[Ideogram4TextEncode] ⚠️ Qwen 失败，回退 Template")
                json_text = _template_convert(text, aspect_ratio)
        else:
            json_text = _template_convert(text, aspect_ratio)

        # ── 后处理：强制写入 aspect_ratio 和 photo 字段 ─────────────────────
        ar_out = _ratio_key(aspect_ratio) if aspect_ratio != "不使用" else ""
        try:
            parsed = json.loads(json_text)
            # 顶层 aspect_ratio（4-bit 模型经常漏写）
            if ar_out:
                parsed["aspect_ratio"] = ar_out
            elif "aspect_ratio" in parsed:
                del parsed["aspect_ratio"]
            # photo 字段：比例 + 高分辨率描述
            sd = parsed.setdefault("style_description", {})
            photo_parts = ["high resolution"]
            if ar_out:
                photo_parts.append(f"{ar_out} aspect ratio")
            sd["photo"] = ", ".join(photo_parts)
            json_text = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # ── CLIP 编码（可选）────────────────────────────────────────────────
        conditioning = None
        if clip is not None:
            tokens = clip.tokenize(json_text)
            conditioning = clip.encode_from_tokens_scheduled(tokens)

        return (conditioning, json_text, text, ar_out)


# ──────────────────────────────────────────────────────────────────────────────
# 导出
# ──────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "Louis_use_Ideogram4TextEncode": Louis_use_Ideogram4TextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Louis_use_Ideogram4TextEncode": "📝 Ideogram4 Text Encode",
}
