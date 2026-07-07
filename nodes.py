"""
ComfyUI Folder I/O Nodes
- FolderImageLoader: 从文件夹批量加载图片
- FolderImageSaver:  保存图片到带前缀的文件夹
"""

import os
import sys
import re
import json as json_module
import random
import hashlib
import time
import tempfile
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → (1, H, W, 3) float32 tensor，值域 0-1"""
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)          # (1, H, W, 3)


def pil_to_mask(img: Image.Image) -> torch.Tensor:
    """从 RGBA 或灰度图提取 mask → (1, H, W) float32，值域 0-1"""
    img = ImageOps.exif_transpose(img)
    if img.mode == "RGBA":
        arr = np.array(img.split()[-1]).astype(np.float32) / 255.0
    else:
        arr = np.zeros((img.height, img.width), dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)           # (1, H, W)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """单张 (H, W, 3) or (1, H, W, 3) tensor → PIL"""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def collect_images(folder: str, sort_by: str = "name") -> list[str]:
    """递归收集文件夹内所有支持格式的图片路径"""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXT]
    if sort_by == "name":
        paths.sort(key=lambda p: p.name)
    elif sort_by == "date":
        paths.sort(key=lambda p: p.stat().st_mtime)
    elif sort_by == "size":
        paths.sort(key=lambda p: p.stat().st_size)
    return [str(p) for p in paths]


def make_output_folder(base_folder: str, prefix: str) -> Path:
    """创建 base_folder/prefix_YYYYMMDD_HHMMSS/ 并返回路径"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = re.sub(r'[\\/*?:"<>|]', "_", prefix).strip("_") or "output"
    out = Path(base_folder) / f"{safe_prefix}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — FolderTextLoader（批量读取 txt 文件）
# ─────────────────────────────────────────────────────────────────────────────

class ResolutionSelector:
    """
    分辨率选择器（Louis 版）。
    选择宽高比 + 百万像素，输出宽度、高度，以及可直接接入
    Ideogram4 Text Encode 的 aspect_ratio 字符串（如 "16:9"）。
    尺寸自动对齐到 64 的倍数。
    """

    _RATIOS = {
        # 方形
        "1:1 正方形":  (1,  1),
        # 横版
        "16:9 横屏":   (16, 9),
        "16:10 横屏":  (16, 10),
        "4:3 横屏":    (4,  3),
        "3:2 横屏":    (3,  2),
        "5:4 横屏":    (5,  4),
        "2:1 超宽":    (2,  1),
        "21:9 超宽":   (21, 9),
        "3:1 超宽":    (3,  1),
        # 竖版
        "9:16 竖屏":   (9,  16),
        "10:16 竖屏":  (10, 16),
        "3:4 竖屏":    (3,  4),
        "2:3 竖屏":    (2,  3),
        "4:5 竖屏":    (4,  5),
        "1:2 竖屏":    (1,  2),
        "9:21 超高":   (9,  21),
        "1:3 超高":    (1,  3),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "宽高比": (list(cls._RATIOS.keys()), {"default": "16:9"}),
                "百万像素": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 16.0, "step": 0.1,
                    "display": "slider",
                    "tooltip": "输出图像的目标像素总量（百万），宽高由宽高比自动分配",
                }),
            },
            "optional": {
                "aspect_ratio": ("STRING", {
                    "forceInput": True,
                    "tooltip": "接 Ideogram4 Text Encode 的 aspect_ratio 输出，覆盖上方宽高比下拉框",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, 宽高比, **kwargs):
        # 兼容旧工作流保存的纯数字比例（如 "16:9"），不报 "Value not in list"
        return True

    RETURN_TYPES  = ("INT", "INT")
    RETURN_NAMES  = ("宽度", "高度")
    FUNCTION      = "calc"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "根据宽高比和百万像素计算输出分辨率；可接 Ideogram4 Text Encode 的 aspect_ratio 统一控制"

    def calc(self, 宽高比: str, 百万像素: float, aspect_ratio: str = ""):
        import math, re
        ar = aspect_ratio.strip() if aspect_ratio and aspect_ratio.strip() else 宽高比
        # 先精确匹配（中文标签），再正则解析（兼容纯 "16:9" 字符串）
        if ar in self._RATIOS:
            rw, rh = self._RATIOS[ar]
        else:
            m = re.search(r'(\d+):(\d+)', ar)
            if m:
                rw, rh = int(m.group(1)), int(m.group(2))
            else:
                rw, rh = self._RATIOS[宽高比]
        total  = 百万像素 * 1_000_000
        w = max(64, round(math.sqrt(total * rw / rh) / 64) * 64)
        h = max(64, round(total / w / 64) * 64)
        print(f"[ResolutionSelector] {ar} @ {百万像素}MP → {w}×{h}")
        return (w, h)


class FolderTextLoader:
    """
    从指定文件夹批量读取所有 .txt 文件，以 LIST 模式逐条输出文本内容。
    适合：把一批 txt 描述文件送入 Ideogram4TextEncode 等文本处理节点。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "txt 文件所在文件夹，如 D:/prompts",
                }),
            },
        }

    RETURN_TYPES    = ("STRING", "STRING")
    RETURN_NAMES    = ("text", "filename")
    OUTPUT_IS_LIST  = (True, True)
    FUNCTION        = "load_texts"
    CATEGORY        = "Louis_use"
    DESCRIPTION     = "批量读取文件夹内所有 .txt 文件，逐条输出文本内容和文件名"

    def load_texts(self, folder_path: str):
        folder = Path(folder_path.strip())
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder}")

        paths = sorted(folder.rglob("*.txt"), key=lambda p: p.name)
        if not paths:
            raise ValueError(f"文件夹中没有找到 .txt 文件: {folder}")

        texts, filenames = [], []
        for p in paths:
            try:
                content = p.read_text(encoding="utf-8").strip()
                texts.append(content)
                filenames.append(p.name)
                print(f"[FolderTextLoader] 读取: {p.name} ({len(content)} 字符)")
            except Exception as e:
                print(f"[FolderTextLoader] 跳过 {p.name}: {e}")

        if not texts:
            raise RuntimeError("所有 txt 文件均无法读取")

        print(f"[FolderTextLoader] 共 {len(texts)} 个文件")
        return (texts, filenames)

    @classmethod
    def IS_CHANGED(cls, folder_path):
        try:
            folder = Path(folder_path.strip())
            h = hashlib.md5()
            for p in sorted(folder.rglob("*.txt"), key=lambda p: p.name):
                h.update(p.name.encode())
                h.update(str(p.stat().st_mtime).encode())
            return h.hexdigest()
        except Exception:
            return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 节点 1 — FolderImageLoader
# ─────────────────────────────────────────────────────────────────────────────

class FolderImageLoader:
    """
    从指定文件夹读取全部图片，以 LIST 模式逐张输出。
    ComfyUI 自动将每张图单独送入下游节点执行，无需手动循环。
    每张图保留原始分辨率，下游节点各自独立适配。
    节点上方显示所有图片的网格缩略图预览。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES   = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES   = ("image", "mask", "filename", "txt")
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION       = "load_images"
    CATEGORY       = "Louis_use"
    DESCRIPTION    = "读取文件夹内全部图片，逐张送入下游节点；同名 .txt/.json 文件内容从 txt 输出"

    def load_images(self, folder_path: str):
        paths = collect_images(folder_path, "name")
        total = len(paths)

        if total == 0:
            raise ValueError(f"文件夹中没有找到支持格式的图片: {folder_path}")

        images, masks, filenames, texts = [], [], [], []
        for p in paths:
            try:
                img = Image.open(p)
                if hasattr(img, "n_frames") and img.n_frames > 1:
                    img.seek(0)
                images.append(pil_to_tensor(img))
                masks.append(pil_to_mask(img))
                filenames.append(os.path.basename(p))

                # 读取同名 .txt 或 .json（优先 .json）
                stem = Path(p).with_suffix("")
                txt_content = ""
                for ext in (".json", ".txt"):
                    txt_file = Path(str(stem) + ext)
                    if txt_file.exists():
                        txt_content = txt_file.read_text(encoding="utf-8")
                        break
                texts.append(txt_content)

            except Exception as e:
                print(f"[FolderImageLoader] 跳过损坏图片 {p}: {e}")

        if not images:
            raise RuntimeError("所有图片均无法读取，请检查文件完整性")

        print(f"[FolderImageLoader] 共 {total} 张，逐张送入下游节点")
        return (images, masks, filenames, texts)

    @classmethod
    def IS_CHANGED(cls, folder_path):
        try:
            paths = collect_images(folder_path, "name")
            h = hashlib.md5()
            for p in paths:
                h.update(p.encode())
                h.update(str(os.path.getmtime(p)).encode())
            return h.hexdigest()
        except Exception:
            return float("nan")




# ─────────────────────────────────────────────────────────────────────────────
# 节点 3 — SmartAlignCrop（参考对齐 / 自动 1080p 内等比）
# ─────────────────────────────────────────────────────────────────────────────

import math
from typing import Optional
from scipy.ndimage import convolve as _sp_convolve


def _fit_within(w: int, h: int, max_w: int, max_h: int):
    """等比缩小到 max_w × max_h 以内，不放大。"""
    if w <= max_w and h <= max_h:
        return w, h
    scale = min(max_w / w, max_h / h)
    # 用 floor 保证不超出最大边界
    return max(1, math.floor(w * scale)), max(1, math.floor(h * scale))


def _resize_to_cover_and_crop(img: Image.Image, target_w: int, target_h: int,
                               resample: int) -> Image.Image:
    """
    等比缩放使图像完全覆盖 target_w × target_h，再从中心裁切。

    关键：用 math.ceil 确保 scaled 尺寸 >= target，
    避免 round() 向下取整导致 PIL.crop 越界填充黑色/紫色。
    """
    src_w, src_h = img.size

    # 如果已经完全一致，直接返回
    if src_w == target_w and src_h == target_h:
        return img.copy()

    scale    = max(target_w / src_w, target_h / src_h)
    # ceil 保证 scaled >= target，绝不出现负 left/top
    scaled_w = max(target_w, math.ceil(src_w * scale))
    scaled_h = max(target_h, math.ceil(src_h * scale))

    img  = img.resize((scaled_w, scaled_h), resample)

    # 居中裁切：left/top >= 0 已由 max(...) 保证
    left = (scaled_w - target_w) // 2
    top  = (scaled_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


class SmartAlignCrop:
    """
    两种工作模式（自动切换）：

    ① 有参考图（reference_image 连线）
       → 以参考图的宽高为目标，对 images 中每张图等比缩放后居中裁切对齐。
         例：参考 1900×810，输入 1901×820 → 只裁掉几像素，不变形。
         尺寸完全一致时直接透传，不做任何处理。

    ② 无参考图
       → 每张图保持自身比例，等比缩小到 max_width×max_height 以内；
         已在范围内的图原样输出，不做任何操作。
    """

    RESAMPLE_MAP = {
        "Lanczos（推荐）": Image.LANCZOS,
        "Bicubic":          Image.BICUBIC,
        "Bilinear":         Image.BILINEAR,
        "Nearest":          Image.NEAREST,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":   ("IMAGE",),
                "resample": (list(cls.RESAMPLE_MAP.keys()),
                             {"default": "Lanczos（推荐）"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": (
                        "连接参考图后，所有输入图等比缩放并居中裁切到与参考图完全相同的尺寸。"
                        "不连接则每张图自动等比缩小到 max_width×max_height 以内。"
                    ),
                }),
                "max_width":  ("INT", {
                    "default": 1920, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "无参考图时的最大宽度（默认 1920）",
                }),
                "max_height": ("INT", {
                    "default": 1080, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "无参考图时的最大高度（默认 1080）",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "INT", "INT")
    RETURN_NAMES  = ("images", "out_width", "out_height")
    FUNCTION      = "run"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "有参考图→对齐参考尺寸居中裁切；无参考图→等比缩小到1080p内"

    def run(self, images: torch.Tensor, resample: str,
            reference_image: Optional[torch.Tensor] = None,
            max_width: int = 1920, max_height: int = 1080):

        # 统一为 4D: (N, H, W, C)
        if images.ndim == 3:
            images = images.unsqueeze(0)

        N        = images.shape[0]
        src_h    = images.shape[1]
        src_w    = images.shape[2]
        resample_filter = self.RESAMPLE_MAP[resample]

        # ── 模式 ①：有参考图 ──────────────────────────────────────────────────
        if reference_image is not None:
            ref = reference_image
            if ref.ndim == 3:
                ref = ref.unsqueeze(0)
            target_h = int(ref.shape[1])
            target_w = int(ref.shape[2])

            # 尺寸完全一致 → 全批次直接透传（clone 防止下游意外修改）
            if src_w == target_w and src_h == target_h:
                print(f"[SmartAlignCrop] 所有图 {src_w}×{src_h} 与参考一致，直接透传")
                return (images.clone(), target_w, target_h)

            # 尺寸不同 → 逐张处理
            out_tensors = []
            for i in range(N):
                pil_img = tensor_to_pil(images[i])          # (H,W,3) → PIL
                pil_out = _resize_to_cover_and_crop(
                    pil_img, target_w, target_h, resample_filter
                )
                out_tensors.append(pil_to_tensor(pil_out))  # (1,H,W,3)
                print(f"[SmartAlignCrop] 图{i+1}: {src_w}×{src_h} "
                      f"→ {target_w}×{target_h}（参考对齐）")

            result = torch.cat(out_tensors, dim=0)
            return (result, target_w, target_h)

        # ── 模式 ②：无参考图，等比缩小到 max_width×max_height 以内 ────────
        new_w, new_h = _fit_within(src_w, src_h, max_width, max_height)

        if new_w == src_w and new_h == src_h:
            print(f"[SmartAlignCrop] {src_w}×{src_h} 已在 {max_width}×{max_height} 范围内，透传")
            return (images.clone(), src_w, src_h)

        # 需要缩小
        out_tensors = []
        for i in range(N):
            pil_img = tensor_to_pil(images[i])
            pil_out = pil_img.resize((new_w, new_h), resample_filter)
            out_tensors.append(pil_to_tensor(pil_out))
            print(f"[SmartAlignCrop] 图{i+1}: {src_w}×{src_h} → {new_w}×{new_h}")

        result = torch.cat(out_tensors, dim=0)
        return (result, new_w, new_h)


# ─────────────────────────────────────────────────────────────────────────────
# 节点 7 — ReflectionExtractor（镜面反射提取）
# ─────────────────────────────────────────────────────────────────────────────

class ReflectionExtractor:
    """
    镜面高光提取：亮度幂曲线压暗，车身极暗，镜面峰值保留环境色。
    PS 叠加模式：Screen（滤色）
    """

    @classmethod
    def INPUT_TYPES(cls):
        def fs(default, mn, mx, step=0.05):
            return ("FLOAT", {"default": float(default), "min": float(mn),
                              "max": float(mx), "step": step, "display": "slider"})
        return {
            "required": {
                "image":  ("IMAGE",),
                "强度":   fs(1.0, 0.0,  3.0, 0.05),
                "对比度": fs(4.0, 1.0, 10.0, 0.1),
            },
            "optional": {
                "遮罩": ("*",),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("镜面高光通道",)
    OUTPUT_NODE   = True
    FUNCTION      = "extract"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "镜面高光通道提取，黑底输出，适合在PS中以Screen模式叠加增强金属反射质感"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def extract(self, image: torch.Tensor, 强度=1.0, 对比度=4.0, 遮罩=None):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        out_list = []
        for i in range(image.shape[0]):
            img_np  = image[i].numpy().astype(np.float32)
            mask_np = self._load_mask(遮罩, i, img_np.shape[:2])

            lum     = 0.299*img_np[...,0] + 0.587*img_np[...,1] + 0.114*img_np[...,2]
            curve   = np.power(np.clip(lum, 0.0, 1.0), float(对比度))[..., np.newaxis]
            out     = img_np * curve * float(强度)
            mean_ch = out.mean(axis=-1, keepdims=True)
            out     = (mean_ch + (out - mean_ch) * 1.8).clip(0.0, 1.0).astype(np.float32)

            if mask_np is not None:
                out = out * mask_np[..., np.newaxis]

            out_list.append(torch.from_numpy(out).unsqueeze(0))

        out_t = torch.cat(out_list, dim=0)

        # 内嵌预览
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        preview_imgs = []
        for j in range(out_t.shape[0]):
            arr   = (out_t[j].numpy() * 255).clip(0, 255).astype(np.uint8)
            fname = "spec_" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=8)) + ".png"
            Image.fromarray(arr).save(os.path.join(temp_dir, fname), compress_level=1)
            preview_imgs.append({"filename": fname, "subfolder": "", "type": "temp"})

        return {"ui": {"images": preview_imgs}, "result": (out_t,)}

    @staticmethod
    def _load_mask(遮罩, i, hw):
        if 遮罩 is None:
            return None
        if 遮罩.ndim == 4:
            m = 遮罩[min(i, 遮罩.shape[0]-1)].numpy().astype(np.float32)
            mask_np = 0.299*m[...,0] + 0.587*m[...,1] + 0.114*m[...,2]
        elif 遮罩.ndim == 3:
            mask_np = 遮罩[min(i, 遮罩.shape[0]-1)].numpy().astype(np.float32)
        else:
            mask_np = 遮罩.numpy().astype(np.float32)
        H, W = hw
        if mask_np.shape != (H, W):
            pil     = Image.fromarray((mask_np*255).clip(0,255).astype(np.uint8), mode="L")
            mask_np = np.array(pil.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
        if mask_np.max() < 0.01:
            print("[ReflectionExtractor] 遮罩全黑，已自动忽略")
            return None
        return mask_np


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — DivisibleCrop（整数倍尺寸裁切）
# ─────────────────────────────────────────────────────────────────────────────

class DivisibleCrop:
    """
    把图像裁切到宽高都能被指定倍数整除。

    用途：
      扩散模型的 VAE 下采样步长通常是 8 或 16（如 SD1.5/SDXL=8, Flux=16）。
      输入图尺寸不满足这个条件时会报错或黑边。
      本节点一键裁出最接近、但宽高都能整除的尺寸。

    策略：
      居中裁切  — 保留中央内容（默认，适合大多数情况）
      左上裁切  — 保留左上角内容
      右下裁切  — 保留右下角内容
    """

    _MODES = ["居中裁切", "左上裁切", "右下裁切"]

    # 模型 → VAE 下采样步长（即宽高必须能被该数整除）
    _MODEL_MULTIPLES = {
        "SD1.5 (8)":    8,
        "SDXL (8)":     8,
        "SD3 (16)":     16,
        "Flux (16)":    16,
        "HiDream (32)": 32,
        "通用安全 (64)": 64,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "模型": (list(cls._MODEL_MULTIPLES.keys()), {
                    "default": "Flux (16)",
                    "tooltip": (
                        "根据目标扩散模型自动选择整除倍数：\n"
                        "SD1.5 / SDXL → 8\n"
                        "SD3 / Flux → 16\n"
                        "HiDream → 32\n"
                        "通用安全 → 64（兼容几乎所有模型 / 分块采样）"
                    ),
                }),
                "裁切方式": (cls._MODES, {
                    "default": "居中裁切",
                    "tooltip": "保留图像的哪一部分",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE", "INT", "INT")
    RETURN_NAMES  = ("图像", "宽", "高")
    FUNCTION      = "process"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "裁切图像使宽高都能被指定倍数整除（适配扩散模型 VAE 步长）"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def process(self, image: torch.Tensor,
                模型: str = "Flux (16)",
                裁切方式: str = "居中裁切"):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        B, H, W, C = image.shape
        m = int(self._MODEL_MULTIPLES.get(模型, 16))
        new_H = (H // m) * m
        new_W = (W // m) * m

        if new_H <= 0 or new_W <= 0:
            raise ValueError(
                f"图像尺寸 {W}×{H} 太小，无法整除 {m}。请减小倍数或换更大的图"
            )

        if new_H == H and new_W == W:
            return (image, int(W), int(H))

        dy = H - new_H
        dx = W - new_W

        if 裁切方式 == "居中裁切":
            y0 = dy // 2
            x0 = dx // 2
        elif 裁切方式 == "左上裁切":
            y0 = 0
            x0 = 0
        else:  # 右下裁切
            y0 = dy
            x0 = dx

        cropped = image[:, y0:y0 + new_H, x0:x0 + new_W, :].contiguous()
        return (cropped, int(new_W), int(new_H))


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — BatchImageSaver（批量保存，支持同步原文件名）
# ─────────────────────────────────────────────────────────────────────────────

class BatchImageSaver:
    """
    批量保存图片到指定文件夹。

    特点：
      - 自由选择格式：PNG / JPG / WEBP
      - 自由选择目标文件夹（绝对或相对 ComfyUI output 目录）
      - 可选是否嵌入工作流元数据（PNG 才生效）
      - 接入【原文件名】时自动用原文件名命名（仅换扩展名），否则按编号命名
      - 同名文件自动加 _1 _2 后缀防覆盖
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输出文件夹": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "如 D:/output 或相对路径 batch_out",
                }),
                "文件格式": (["png", "jpg", "webp"], {"default": "png"}),
                "质量": ("INT", {
                    "default": 100, "min": 1, "max": 100, "step": 1,
                    "tooltip": "JPG/WEBP 压缩质量（PNG 忽略）",
                }),
                "嵌入工作流": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "True = 把 workflow/prompt 写入 PNG 元数据（可拖回 ComfyUI 还原）\n"
                        "False = 不写入（成品交付 / 防止工作流泄漏）\n"
                        "仅 PNG 生效，JPG/WEBP 一律不写入"
                    ),
                }),
                "覆盖已有文件": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "True = 同名文件直接覆盖（接原文件名时推荐）；False = 自动加 _1 _2 防止覆盖",
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "不接时只保存文本文件",
                }),
                "原文件名": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "接 Folder Image Loader / Folder Text Loader 的 filename 输出，自动用原文件名命名",
                }),
                "text": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "纯文本内容，与图片同名保存为 .txt 文件",
                }),
                "json": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "JSON 内容，与图片同名保存为 .json 文件（接 Ideogram4 Text Encode 的 json_text 输出）",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES  = ()
    OUTPUT_NODE   = True
    FUNCTION      = "save"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "批量保存图片；接入原文件名时自动同步文件名，否则按编号命名"

    def _resolve_dir(self, 输出文件夹: str) -> Path:
        p = Path(输出文件夹.strip())
        if not p.is_absolute():
            p = Path(folder_paths.get_output_directory()) / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(
        self,
        输出文件夹: str,
        文件格式: str = "png",
        质量: int = 100,
        嵌入工作流: bool = True,
        覆盖已有文件: bool = True,
        images=None,
        原文件名: str = "",
        text: str = "",
        json: str = "",
        prompt=None,
        extra_pnginfo=None,
    ):
        if not 输出文件夹.strip():
            raise ValueError("输出文件夹 不能为空")

        out_dir = self._resolve_dir(输出文件夹)

        # ── 原文件名列表 ──────────────────────────────────────────────────────
        src_names: list[str] = []
        if 原文件名:
            src_names = [s.strip() for s in 原文件名.splitlines() if s.strip()]

        # ── 确定循环次数：有图片按图片数，纯 txt 模式按文件名数（至少 1 条）──
        if images is not None:
            if images.ndim == 3:
                images = images.unsqueeze(0)
            N = images.shape[0]
        else:
            N = max(len(src_names), 1)

        ext = 文件格式.lower()
        saved: list[str] = []

        for i in range(N):
            # ── 决定文件名 ──
            if src_names:
                src  = src_names[i] if i < len(src_names) else src_names[-1]
                stem = Path(src).stem
            else:
                stem = f"img_{i + 1:05d}"

            # ── 文件名碰撞处理 ──
            base_path = out_dir / f"{stem}.{ext}"
            if not 覆盖已有文件:
                k = 1
                while (base_path.exists()
                       or (text and base_path.with_suffix(".txt").exists())
                       or (json and base_path.with_suffix(".json").exists())):
                    base_path = out_dir / f"{stem}_{k}.{ext}"
                    k += 1
            fpath = base_path

            # ── 保存图片（有 images 时）────────────────────────────────────
            if images is not None:
                pil = tensor_to_pil(images[i])
                kw: dict = {}
                if ext == "jpg":
                    kw = {"quality": int(质量), "optimize": True}
                    pil = pil.convert("RGB")
                elif ext == "webp":
                    kw = {"quality": int(质量), "method": 6}
                else:  # png
                    kw = {"compress_level": 4}
                    if 嵌入工作流:
                        from PIL.PngImagePlugin import PngInfo
                        meta = PngInfo()
                        if prompt is not None:
                            meta.add_text("prompt", json_module.dumps(prompt))
                        if extra_pnginfo is not None:
                            for k2, v2 in extra_pnginfo.items():
                                meta.add_text(k2, json_module.dumps(v2))
                        kw["pnginfo"] = meta
                pil.save(str(fpath), **kw)
                saved.append(str(fpath))
                print(f"[BatchImageSaver] 已保存: {fpath}")

            # ── 保存 text → .txt ─────────────────────────────────────────────
            if text:
                txt_path = fpath.with_suffix(".txt")
                txt_path.write_text(text, encoding="utf-8")
                print(f"[BatchImageSaver] 已保存: {txt_path}")

            # ── 保存 json → .json ────────────────────────────────────────────
            if json:
                json_path = fpath.with_suffix(".json")
                json_path.write_text(json, encoding="utf-8")
                print(f"[BatchImageSaver] 已保存: {json_path}")

        total = len(saved) if images is not None else 0
        print(f"[BatchImageSaver] 完成 → {out_dir}  (图片 {total} 张)")
        return {}



# ─────────────────────────────────────────────────────────────────────────────
# 节点 — ColorPaletteExtractor
# ─────────────────────────────────────────────────────────────────────────────

class ColorPaletteExtractor:
    """分析图片主要颜色，按占比生成竖条色块图"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":         ("IMAGE",),
                "num_colors":    ("INT",   {"default": 6,    "min": 2,   "max": 20,   "step": 1}),
                "output_width":  ("INT",   {"default": 1000, "min": 64,  "max": 4096, "step": 8}),
                "output_height": ("INT",   {"default": 400,  "min": 64,  "max": 4096, "step": 8}),
                "min_ratio":     ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.5,  "step": 0.005}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("palette_image",)
    FUNCTION      = "extract_palette"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "分析图片主要颜色，按占比生成竖条色块图"

    def extract_palette(self, image, num_colors, output_width, output_height, min_ratio):
        img_pil = tensor_to_pil(image).convert("RGB")

        # 缩小加速量化
        w, h = img_pil.size
        scale = min(400 / w, 400 / h)
        if scale < 1.0:
            img_small = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            img_small = img_pil

        # PIL 量化提取主色
        quantized = img_small.quantize(colors=num_colors)
        palette_raw = quantized.getpalette()          # [R,G,B, R,G,B, ...]
        pixels = np.array(quantized).reshape(-1)
        total  = pixels.size

        # 统计每个调色板索引的像素数
        color_data = []
        for i in range(num_colors):
            ratio = int(np.sum(pixels == i)) / total
            if ratio < min_ratio:
                continue
            r = palette_raw[i * 3]
            g = palette_raw[i * 3 + 1]
            b = palette_raw[i * 3 + 2]
            color_data.append(((r, g, b), ratio))

        if not color_data:
            color_data = [((128, 128, 128), 1.0)]

        # 归一化占比（排除 min_ratio 过滤掉的部分）
        total_ratio = sum(r for _, r in color_data)
        color_data  = [(c, r / total_ratio) for c, r in color_data]

        # 按占比降序排列
        color_data.sort(key=lambda x: x[1], reverse=True)

        # 生成竖条色块图
        result = Image.new("RGB", (output_width, output_height))
        x = 0
        for idx, (color, ratio) in enumerate(color_data):
            # 最后一条填满余量，避免舍入误差留白
            if idx == len(color_data) - 1:
                bar_w = output_width - x
            else:
                bar_w = int(output_width * ratio)
            if bar_w <= 0:
                continue
            result.paste(Image.new("RGB", (bar_w, output_height), color), (x, 0))
            x += bar_w

        return (pil_to_tensor(result),)


class ColorMatch:
    """
    把输入图的色调对齐到参考图，用于修复分块放大后整体色调漂移。

    method:
      mean_std   — 逐通道均值+标准差（Reinhard），快、平滑
      mvgd       — Monge-Kantorovich 协方差匹配（Pitié 2007），保留通道间色彩关系
      wavelet    — 多尺度小波分解，保留高频细节只换低频色调，最适合"放大+细化"场景
      histogram  — 逐通道直方图匹配，最激进，可能产生伪影

    strength: 0=不变，1=完全对齐参考图
    enabled:  关闭时直接透传输入，方便对比

    放置位置：
      • 推荐放在 TileGather 之前（每块 tile 单独匹配对应源 tile，精度最高）
      • 也可以放在 TileImageAssemble 之后（全图匹配，但精度较低）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":     ("IMAGE",),
                "reference": ("IMAGE",),
                "enabled":   ("BOOLEAN", {
                    "default": True,
                    "label_on":  "开 (On)",
                    "label_off": "关 (Off)",
                }),
                "method": (["mean_std", "mvgd", "wavelet", "histogram"], {
                    "default": "wavelet",
                    "tooltip": "wavelet=只换色调保留细节（放大场景推荐）；mvgd=协方差匹配；mean_std=快速；histogram=直方图",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "display": "slider",
                    "tooltip": "对齐强度，0=原样输出，1=完全对齐参考",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "match"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "把输出图色调对齐到参考图（修复分块放大色调漂移），带一键开关"

    @staticmethod
    def _resize_ref(ref: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if ref.shape[1] == target_h and ref.shape[2] == target_w:
            return ref
        return torch.nn.functional.interpolate(
            ref.permute(0, 3, 1, 2),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

    @staticmethod
    def _mean_std(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        img_mean = img.mean(dim=(1, 2), keepdim=True)
        img_std  = img.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        ref_mean = ref.mean(dim=(1, 2), keepdim=True)
        ref_std  = ref.std(dim=(1, 2), keepdim=True)
        return (img - img_mean) / img_std * ref_std + ref_mean

    @staticmethod
    def _mvgd(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Monge-Kantorovich Linear color transfer (Pitié et al. 2007).
        匹配源图与目标图的均值 + 协方差矩阵，
        相比 mean_std 还能保留通道间相关性，色彩迁移更准确、更不易偏色。
        """
        out = torch.empty_like(img)
        eps = 1e-6
        I3  = torch.eye(3, device=img.device, dtype=torch.float64)

        for n in range(img.shape[0]):
            ref_n = ref[min(n, ref.shape[0] - 1)]

            src = img[n].reshape(-1, 3).to(torch.float64)
            tgt = ref_n.reshape(-1, 3).to(torch.float64)

            src_mean = src.mean(dim=0)
            tgt_mean = tgt.mean(dim=0)

            src_cov = torch.cov(src.T) + eps * I3
            tgt_cov = torch.cov(tgt.T) + eps * I3

            # 对称矩阵开方：A^(1/2) = V·diag(sqrt(λ))·V^T
            es, Vs = torch.linalg.eigh(src_cov)
            es = es.clamp(min=eps)
            sqrt_src     = Vs @ torch.diag(es.sqrt()) @ Vs.T
            inv_sqrt_src = Vs @ torch.diag(1.0 / es.sqrt()) @ Vs.T

            M = sqrt_src @ tgt_cov @ sqrt_src
            eM, VM = torch.linalg.eigh(M)
            eM = eM.clamp(min=eps)
            sqrt_M = VM @ torch.diag(eM.sqrt()) @ VM.T

            T = inv_sqrt_src @ sqrt_M @ inv_sqrt_src

            result = (src - src_mean) @ T.T + tgt_mean
            out[n] = result.reshape(img.shape[1:]).to(img.dtype)

        return out

    @staticmethod
    def _wavelet_blur(image: torch.Tensor, radius: int) -> torch.Tensor:
        """单层小波（3×3 高斯近似）模糊。image 形状为 [N, 3, H, W]。"""
        kernel = torch.tensor([
            [0.0625, 0.125, 0.0625],
            [0.125,  0.25,  0.125 ],
            [0.0625, 0.125, 0.0625],
        ], dtype=image.dtype, device=image.device)[None, None].repeat(3, 1, 1, 1)
        padded = torch.nn.functional.pad(image, (radius,) * 4, mode="replicate")
        return torch.nn.functional.conv2d(padded, kernel, groups=3, dilation=radius)

    @classmethod
    def _wavelet_decompose(cls, image: torch.Tensor, levels: int = 5):
        """多尺度小波分解，返回 (高频累加, 最低频)。"""
        high_freq = torch.zeros_like(image)
        low_freq  = image
        for i in range(levels):
            radius = 2 ** i
            low_freq  = cls._wavelet_blur(image, radius)
            high_freq = high_freq + (image - low_freq)
            image     = low_freq
        return high_freq, low_freq

    @classmethod
    def _wavelet(cls, img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        小波重构色彩匹配：
        把 img 分解成高频(细节) + 低频(色调)；
        把 ref 分解成同样两层；
        用 img 的高频 + ref 的低频组合 → 保留细节只换色调。
        """
        img_cf = img.permute(0, 3, 1, 2).contiguous()
        ref_cf = ref.permute(0, 3, 1, 2).contiguous()

        img_high, _      = cls._wavelet_decompose(img_cf)
        _,        ref_lo = cls._wavelet_decompose(ref_cf)

        result = img_high + ref_lo
        return result.permute(0, 2, 3, 1)

    @staticmethod
    def _histogram(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(img)
        for n in range(img.shape[0]):
            ref_n = ref[min(n, ref.shape[0] - 1)]
            for c in range(3):
                src_flat = img[n, :, :, c].cpu().numpy().flatten()
                tgt_flat = ref_n[:, :, c].cpu().numpy().flatten()
                src_sorted_idx = np.argsort(src_flat)
                tgt_sorted     = np.sort(tgt_flat)
                indices = np.linspace(0, len(tgt_sorted) - 1, len(src_flat)).astype(np.int64)
                mapped  = tgt_sorted[indices]
                result  = np.empty_like(src_flat)
                result[src_sorted_idx] = mapped
                out[n, :, :, c] = torch.from_numpy(
                    result.reshape(img.shape[1], img.shape[2])
                ).to(img.device, img.dtype)
        return out

    def match(self, image: torch.Tensor, reference: torch.Tensor,
              enabled: bool, method: str, strength: float):
        if not enabled or strength <= 0.0:
            return (image,)

        ref = self._resize_ref(reference.to(image.device, image.dtype),
                               image.shape[1], image.shape[2])

        if method == "histogram":
            matched = self._histogram(image, ref)
        elif method == "mvgd":
            matched = self._mvgd(image, ref)
        elif method == "wavelet":
            matched = self._wavelet(image, ref)
        else:
            matched = self._mean_std(image, ref)

        if strength < 1.0:
            matched = image * (1.0 - strength) + matched * strength

        return (matched.clamp(0.0, 1.0),)


# ─────────────────────────────────────────────────────────────────────────────
# 无缝贴图工具集
# ─────────────────────────────────────────────────────────────────────────────

class SeamlessTileFixer:
    """
    将图像处理为可无缝平铺的贴图。

    方法：
      偏移混合法（保持尺寸）—— 将图像在 X/Y 方向各偏移 50%，
        使原本在边缘的接缝移到中央，再通过渐变交叉混合消除中央接缝。
        输出尺寸与输入相同，四边可完美平铺。

    混合强度：柔和 / 标准 / 强 → 控制中央接缝混合宽度（占图像尺寸的比例）。
    预览格数：将生成的无缝贴图平铺成 N×N 的预览图像。
    """

    _METHODS       = ["偏移混合法（保持尺寸）"]
    _GRID_SIZES    = ["1×1", "1×2", "2×1", "2×2", "3×3", "4×4"]
    _BLEND_LEVELS  = ["柔和", "标准", "强", "极强"]
    _BLEND_RATIOS  = {"柔和": 0.10, "标准": 0.18, "强": 0.30, "极强": 0.45}
    # 预览格数 → (行, 列)
    _GRID_COUNTS   = {
        "1×1": (1, 1), "1×2": (1, 2), "2×1": (2, 1),
        "2×2": (2, 2), "3×3": (3, 3), "4×4": (4, 4),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":    ("IMAGE",),
                "方法":     (cls._METHODS,      {"default": "偏移混合法（保持尺寸）"}),
                "预览格数": (cls._GRID_SIZES,   {"default": "3×3"}),
                "混合强度": (cls._BLEND_LEVELS, {"default": "标准"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE")
    RETURN_NAMES  = ("无缝贴图", "平铺预览")
    FUNCTION      = "fix"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "偏移50%+交叉混合，生成四边无缝平铺贴图，附 行×列 平铺预览"

    def fix(self, image: torch.Tensor, 方法: str, 预览格数: str, 混合强度: str):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        blend_ratio  = self._BLEND_RATIOS.get(混合强度, 0.10)
        rows, cols   = self._GRID_COUNTS.get(预览格数, (3, 3))

        results, previews = [], []
        for i in range(image.shape[0]):
            arr = image[i].numpy().astype(np.float32)

            seamless = self._offset_blend(arr, blend_ratio)
            results.append(torch.from_numpy(seamless).unsqueeze(0))
            previews.append(torch.from_numpy(
                self._tiled_preview(seamless, rows, cols)
            ).unsqueeze(0))

        return (torch.cat(results), torch.cat(previews))

    @staticmethod
    def _offset_blend(arr: np.ndarray, blend_ratio: float) -> np.ndarray:
        """
        1. 将图像在 X/Y 各偏移 50%（接缝移到中央）
        2. 对中央接缝两侧对称像素对做羽化交叉混合：
           接缝正中 w=0.5（两侧取平均 → 完全连续，无硬线），
           向外按余弦平滑羽化到 w=0（不改动），过渡自然柔和。
        3. 结果四边无缝，中央过渡平滑无硬缝
        """
        H, W = arr.shape[:2]

        rolled = np.roll(arr, H // 2, axis=0)
        rolled = np.roll(rolled, W // 2, axis=1)

        bh = max(2, int(H * blend_ratio))
        bw = max(2, int(W * blend_ratio))
        cy, cx = H // 2, W // 2

        result = rolled.copy()

        # 修复水平接缝（以 cy 为轴，对称向外各 bh 行）
        for i in range(bh):
            row_a = (cy - i - 1) % H   # 接缝上方行
            row_b = (cy + i)     % H   # 接缝下方行
            t = (i + 0.5) / bh                      # 0（接缝）→1（边界）
            w = 0.25 * (1.0 + math.cos(math.pi * t))  # 接缝处 w=0.5，边界处 w=0
            a = result[row_a].copy()
            b = result[row_b].copy()
            result[row_a] = (1.0 - w) * a + w * b
            result[row_b] = (1.0 - w) * b + w * a

        # 修复垂直接缝
        for i in range(bw):
            col_a = (cx - i - 1) % W
            col_b = (cx + i)     % W
            t = (i + 0.5) / bw
            w = 0.25 * (1.0 + math.cos(math.pi * t))
            a = result[:, col_a].copy()
            b = result[:, col_b].copy()
            result[:, col_a] = (1.0 - w) * a + w * b
            result[:, col_b] = (1.0 - w) * b + w * a

        return result.clip(0.0, 1.0)

    @staticmethod
    def _tiled_preview(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """将无缝贴图平铺成 rows×cols（行×列），缩放到不超过 1024px。"""
        tiled = np.tile(arr, (rows, cols, 1))
        H, W = tiled.shape[:2]
        max_px = 1024
        if H > max_px or W > max_px:
            scale = min(max_px / H, max_px / W)
            nh, nw = max(1, int(H * scale)), max(1, int(W * scale))
            pil = Image.fromarray((tiled * 255).clip(0, 255).astype(np.uint8))
            tiled = np.array(pil.resize((nw, nh), Image.LANCZOS)).astype(np.float32) / 255.0
        return tiled.clip(0.0, 1.0)





# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace 下载公共工具（TextSegmenter / AutoMatter 共用）
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_pkg(import_name: str, pip_spec: str):
    """
    确保某依赖可用，不可用则用当前解释器 pip 自动安装/升级。
    import_name: 用于检测的模块路径，如 'timm.layers'
    pip_spec:    安装规格，如 'timm>=1.0.0'
    """
    import importlib, importlib.util, subprocess, sys
    try:
        if importlib.util.find_spec(import_name) is not None:
            return
    except (ImportError, ModuleNotFoundError, ValueError):
        pass  # 父包版本太旧时 find_spec 可能抛错，继续安装

    print(f"[Louis_use] 缺少依赖 {import_name}，正在自动安装 {pip_spec} …")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-U", pip_spec,
    ])
    importlib.invalidate_caches()
    print(f"[Louis_use] {pip_spec} 安装完成 ✓")


def _make_hf_forced_tqdm():
    """返回一个始终显示进度条的 tqdm 子类（Windows CMD 非 TTY 友好）。"""
    import sys, inspect
    from tqdm import tqdm as _Base
    _ok = set(inspect.signature(_Base.__init__).parameters)

    class _ForcedTqdm(_Base):
        def __init__(self, *args, **kw):
            kw["disable"]       = False
            kw["file"]          = sys.stderr
            kw["dynamic_ncols"] = True
            kw = {k: v for k, v in kw.items() if k in _ok}  # 过滤 HF 私有参数
            super().__init__(*args, **kw)
    return _ForcedTqdm


def _hf_snapshot(model_id: str, **extra_kw) -> str:
    """
    获取 HuggingFace 模型本地路径：
    - 优先用 local_files_only=True 直接返回缓存路径，完全不联网
    - 本地无缓存时才真正下载，并显示进度条
    - 不做 weight_ok 二次检查（huggingface_hub 内部已做 sha256 校验）
    """
    import logging, warnings
    from huggingface_hub import snapshot_download

    for _log in ("httpx", "huggingface_hub",
                 "huggingface_hub.file_download", "huggingface_hub.repocard"):
        logging.getLogger(_log).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)

    # ── 1. 尝试纯本地加载（不联网，毫秒级返回）──────────────────────────────
    try:
        return snapshot_download(repo_id=model_id, local_files_only=True, **extra_kw)
    except Exception:
        pass

    # ── 2. 本地无完整缓存，正式下载 ─────────────────────────────────────────
    _Tqdm = _make_hf_forced_tqdm()
    try:
        return snapshot_download(repo_id=model_id, tqdm_class=_Tqdm, **extra_kw)
    except TypeError:
        return snapshot_download(repo_id=model_id, **extra_kw)


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — TextSegmenter（文字驱动抠图）
# ─────────────────────────────────────────────────────────────────────────────

class TextSegmenter:
    """
    文字驱动精细抠图（两阶段）：
      Stage 1 — Grounding DINO 根据文字输出精准 bounding box
      Stage 2 — BiRefNet 在 box 区域内高质量抠图

    • 文字建议英文，多目标用英文句号分隔：person. cat. red car.
    • 文字留空 → 跳过 Grounding DINO，BiRefNet 直接处理整张图（自动抠显著前景）
    • 首次运行自动下载 Grounding DINO Tiny（~340 MB）和 BiRefNet（~1 GB）
    """

    _GDINO_ID    = "IDEA-Research/grounding-dino-tiny"
    _BIREFNET_ID = "ZhengPeng7/BiRefNet"
    _cache: dict = {}

    _RESOLUTIONS = ["512", "1024", "1536", "2048"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "文字提示": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "多目标用句号分隔，如: person. cat. red car.",
                    "tooltip": "英文效果最好；多目标用 '. ' 分隔；留空=自动抠前景",
                }),
                "识别严格度": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05,
                    "display": "slider",
                    "tooltip": "数值越高越严格（只框非常明显的目标）；越低越宽松（更容易找到目标，但可能误识别）",
                }),
                "推理分辨率": (cls._RESOLUTIONS, {
                    "default": "1024",
                    "tooltip": "BiRefNet 内部推理分辨率；越高越精细",
                }),
                "羽化半径": ("INT", {
                    "default": 2, "min": 0, "max": 30, "step": 1,
                    "tooltip": "遮罩边缘羽化（像素），0=硬边",
                }),
                "反转遮罩": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "反转（保留背景）",
                    "label_off": "正常（保留前景）",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES  = ("抠图", "遮罩", "预览")
    FUNCTION      = "segment"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = (
        "Grounding DINO 文字定位 + BiRefNet 精细抠图。"
        "文字留空则自动抠前景。首次运行下载两个模型（共 ~1.3 GB）。"
    )

    # ── Grounding DINO 延迟加载 ─────────────────────────────────────────────────
    @classmethod
    def _get_gdino(cls):
        key = "gdino"
        if key not in cls._cache:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            from huggingface_hub import snapshot_download
            # 判断是首次下载还是直接从本地缓存加载
            try:
                snapshot_download(repo_id=cls._GDINO_ID, local_files_only=True)
                print(f"[TextSegmenter] 从本地缓存加载 Grounding DINO Tiny…")
            except Exception:
                print(f"[TextSegmenter] 首次下载 Grounding DINO Tiny（约 340 MB）…")
            local_dir = _hf_snapshot(cls._GDINO_ID)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            proc  = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
            model = (AutoModelForZeroShotObjectDetection
                     .from_pretrained(local_dir, local_files_only=True)
                     .to(device).eval())
            cls._cache[key] = (proc, model, device)
            print(f"[TextSegmenter] Grounding DINO 就绪，设备: {device}")
        return cls._cache[key]

    # ── BiRefNet 延迟加载 ───────────────────────────────────────────────────────
    @classmethod
    def _get_birefnet(cls):
        key = "birefnet"
        if key not in cls._cache:
            # BiRefNet 远程代码依赖 timm>=0.9（需 timm.layers），旧版会报
            # ModuleNotFoundError: No module named 'timm.layers'，此处自动修复
            _ensure_pkg("timm.layers", "timm>=1.0.0")
            from transformers import AutoModelForImageSegmentation
            from huggingface_hub import snapshot_download
            # 判断是首次下载还是直接从本地缓存加载
            try:
                snapshot_download(repo_id=cls._BIREFNET_ID, local_files_only=True)
                print(f"[TextSegmenter] 从本地缓存加载 BiRefNet…")
            except Exception:
                print(f"[TextSegmenter] 首次下载 BiRefNet（约 1 GB）…")
            local_dir = _hf_snapshot(cls._BIREFNET_ID)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = (AutoModelForImageSegmentation
                      .from_pretrained(local_dir, trust_remote_code=True,
                                       local_files_only=True)
                      .to(device).eval())
            torch.set_float32_matmul_precision("high")
            cls._cache[key] = (model, device)
            print(f"[TextSegmenter] BiRefNet 就绪，设备: {device}")
        return cls._cache[key]

    # ── 主函数 ─────────────────────────────────────────────────────────────────
    def segment(self, image: torch.Tensor,
                文字提示: str, 识别严格度: float,
                推理分辨率: str, 羽化半径: int, 反转遮罩: bool):
        from torchvision import transforms
        from PIL import ImageFilter

        if image.ndim == 3:
            image = image.unsqueeze(0)

        text    = 文字提示.strip()
        use_text = text != ""
        res     = int(推理分辨率)

        birefnet_pre = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        cutouts, masks_out, previews = [], [], []

        for b in range(image.shape[0]):
            arr     = image[b].numpy().astype(np.float32)
            H, W, _ = arr.shape
            pil_img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))

            # ── Stage 1: Grounding DINO 定位 ───────────────────────────────────
            crop_boxes = []   # 可能检测到多个目标
            if use_text:
                proc, gdino, gd_dev = self._get_gdino()
                # Grounding DINO 要求文字以 '. ' 结尾
                gd_text = text if text.endswith(".") else text + "."
                inputs  = proc(images=pil_img, text=gd_text,
                               return_tensors="pt").to(gd_dev)
                with torch.no_grad():
                    outputs = gdino(**inputs)
                raw = proc.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    target_sizes=[(H, W)],
                )[0]
                keep  = raw["scores"].cpu() >= float(识别严格度)
                boxes = raw["boxes"].cpu()[keep]   # xyxy 绝对坐标
                if len(boxes) == 0:
                    print(f"[TextSegmenter] 未检测到 '{text}'（识别严格度过高？），改为全图抠图")
                else:
                    pad = 0.08   # bounding box 外扩 8%
                    for box in boxes:
                        x1, y1, x2, y2 = box.tolist()
                        pw = (x2 - x1) * pad
                        ph = (y2 - y1) * pad
                        x1 = max(0, int(x1 - pw))
                        y1 = max(0, int(y1 - ph))
                        x2 = min(W, int(x2 + pw))
                        y2 = min(H, int(y2 + ph))
                        crop_boxes.append((x1, y1, x2, y2))

            # ── Stage 2: BiRefNet 精细抠图 ────────────────────────────────────
            br_model, br_dev = self._get_birefnet()
            mask_full = np.zeros((H, W), dtype=np.float32)

            br_dtype  = next(br_model.parameters()).dtype   # 自动匹配模型精度
            run_boxes = crop_boxes if crop_boxes else [(0, 0, W, H)]
            for (x1, y1, x2, y2) in run_boxes:
                pil_crop = pil_img.crop((x1, y1, x2, y2))
                cW, cH   = pil_crop.size
                inp = birefnet_pre(pil_crop).unsqueeze(0).to(device=br_dev, dtype=br_dtype)
                with torch.no_grad():
                    preds = br_model(inp)
                pred = preds[-1].sigmoid().squeeze().cpu().numpy()
                pil_pred = Image.fromarray(
                    (pred * 255).clip(0, 255).astype(np.uint8), mode="L"
                ).resize((cW, cH), Image.LANCZOS)
                # 多目标取并集
                mask_full[y1:y2, x1:x2] = np.maximum(
                    mask_full[y1:y2, x1:x2],
                    np.array(pil_pred) / 255.0
                )

            if 反转遮罩:
                mask_full = 1.0 - mask_full

            if 羽化半径 > 0:
                pil_mf = Image.fromarray(
                    (mask_full * 255).astype(np.uint8), mode="L"
                ).filter(ImageFilter.GaussianBlur(radius=羽化半径))
                mask_full = np.array(pil_mf) / 255.0

            # ── 输出 ──────────────────────────────────────────────────────────
            cutout  = arr * mask_full[..., np.newaxis]
            tint    = np.array([0.10, 0.90, 0.40], dtype=np.float32)
            alpha_v = (mask_full * 0.5)[..., np.newaxis]
            preview = arr * (1.0 - alpha_v) + tint * alpha_v

            cutouts.append(torch.from_numpy(cutout.clip(0, 1)).unsqueeze(0))
            masks_out.append(torch.from_numpy(mask_full).unsqueeze(0))
            previews.append(torch.from_numpy(preview.clip(0, 1)).unsqueeze(0))

        return (torch.cat(cutouts),
                torch.cat(masks_out),
                torch.cat(previews))



# ─────────────────────────────────────────────────────────────────────────────
# 节点 — TextSegmentedDepth（文字驱动目标深度图）
# ─────────────────────────────────────────────────────────────────────────────

class TextSegmentedDepth:
    """
    一步生成「仅含目标」的深度图：
      Stage 1 — Depth Anything V2（约 95 MB）对全图做深度估计
      Stage 2 — Grounding DINO 文字定位目标（留空则跳过）
      Stage 3 — BiRefNet 精细分割生成遮罩
      Stage 4 — 深度图 × 遮罩，背景清零

    • 可选接入「外部深度图」（如 controlnet_aux 已生成），跳过内部深度估计
    • 文字留空 → BiRefNet 直接抠显著前景，无需 GDINO
    """

    _DEPTH_ID    = "depth-anything/Depth-Anything-V2-Small-hf"
    _GDINO_ID    = "IDEA-Research/grounding-dino-tiny"
    _BIREFNET_ID = "ZhengPeng7/BiRefNet"
    _cache: dict = {}

    _RESOLUTIONS = ["512", "1024", "1536", "2048"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "文字提示": ("STRING", {
                    "default": "car",
                    "multiline": False,
                    "placeholder": "英文，多目标用句号分隔: person. car.",
                    "tooltip": "留空 = BiRefNet 自动抠显著前景",
                }),
                "识别严格度": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05,
                    "display": "slider",
                    "tooltip": "数值越高越严格（只框非常明显的目标）；越低越宽松（更容易找到目标，但可能误识别）",
                }),
                "分割分辨率": (cls._RESOLUTIONS, {
                    "default": "1024",
                    "tooltip": "BiRefNet 推理分辨率，越高越精细",
                }),
                "羽化半径": ("INT", {
                    "default": 2, "min": 0, "max": 30, "step": 1,
                    "tooltip": "遮罩边缘羽化像素，0=硬边",
                }),
                "反转遮罩": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "反转（保留背景深度）",
                    "label_off": "正常（保留前景深度）",
                }),
            },
            "optional": {
                "外部深度图": ("IMAGE", {
                    "tooltip": "提供则跳过内部 Depth-Anything-V2，直接使用此深度图",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES  = ("遮罩深度图", "深度图", "遮罩", "预览")
    FUNCTION      = "extract"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = (
        "Depth-Anything-V2 + GDINO + BiRefNet 一步生成仅含目标的深度图。"
        "可选传入外部深度图跳过内部深度估计步骤。"
    )

    # ── Depth Anything V2 ──────────────────────────────────────────────────────
    @classmethod
    def _get_depth_pipe(cls):
        key = "depth_v2"
        if key not in cls._cache:
            from transformers import pipeline as hf_pipeline
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id=cls._DEPTH_ID, local_files_only=True)
                print("[TextSegmentedDepth] 从本地缓存加载 Depth-Anything-V2-Small…")
            except Exception:
                print("[TextSegmentedDepth] 首次下载 Depth-Anything-V2-Small（约 95 MB）…")
            local_dir  = _hf_snapshot(cls._DEPTH_ID)
            device_id  = 0 if torch.cuda.is_available() else -1
            pipe = hf_pipeline(
                "depth-estimation",
                model=local_dir,
                device=device_id,
                local_files_only=True,
            )
            cls._cache[key] = pipe
            print(f"[TextSegmentedDepth] Depth-Anything-V2 就绪，device={device_id}")
        return cls._cache[key]

    # ── Grounding DINO ─────────────────────────────────────────────────────────
    @classmethod
    def _get_gdino(cls):
        key = "gdino"
        if key not in cls._cache:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id=cls._GDINO_ID, local_files_only=True)
                print("[TextSegmentedDepth] 从本地缓存加载 Grounding DINO Tiny…")
            except Exception:
                print("[TextSegmentedDepth] 首次下载 Grounding DINO Tiny（约 340 MB）…")
            local_dir = _hf_snapshot(cls._GDINO_ID)
            device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            proc  = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
            model = (AutoModelForZeroShotObjectDetection
                     .from_pretrained(local_dir, local_files_only=True)
                     .to(device).eval())
            cls._cache[key] = (proc, model, device)
            print(f"[TextSegmentedDepth] Grounding DINO 就绪，设备: {device}")
        return cls._cache[key]

    # ── BiRefNet ───────────────────────────────────────────────────────────────
    @classmethod
    def _get_birefnet(cls):
        key = "birefnet"
        if key not in cls._cache:
            # BiRefNet 远程代码依赖 timm>=0.9（需 timm.layers），旧版会报
            # ModuleNotFoundError: No module named 'timm.layers'，此处自动修复
            _ensure_pkg("timm.layers", "timm>=1.0.0")
            from transformers import AutoModelForImageSegmentation
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id=cls._BIREFNET_ID, local_files_only=True)
                print("[TextSegmentedDepth] 从本地缓存加载 BiRefNet…")
            except Exception:
                print("[TextSegmentedDepth] 首次下载 BiRefNet（约 1 GB）…")
            local_dir = _hf_snapshot(cls._BIREFNET_ID)
            device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = (AutoModelForImageSegmentation
                     .from_pretrained(local_dir, trust_remote_code=True,
                                      local_files_only=True)
                     .to(device).eval())
            torch.set_float32_matmul_precision("high")
            cls._cache[key] = (model, device)
            print(f"[TextSegmentedDepth] BiRefNet 就绪，设备: {device}")
        return cls._cache[key]

    # ── 主函数 ─────────────────────────────────────────────────────────────────
    def extract(self, image: torch.Tensor,
                文字提示: str, 识别严格度: float,
                分割分辨率: str, 羽化半径: int, 反转遮罩: bool,
                外部深度图=None):
        from torchvision import transforms
        from PIL import ImageFilter

        if image.ndim == 3:
            image = image.unsqueeze(0)

        text     = 文字提示.strip()
        use_text = text != ""
        seg_res  = int(分割分辨率)

        birefnet_pre = transforms.Compose([
            transforms.Resize((seg_res, seg_res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        masked_depth_out, depth_out, mask_out, preview_out = [], [], [], []

        for b in range(image.shape[0]):
            arr     = image[b].numpy().astype(np.float32)
            H, W, _ = arr.shape
            pil_img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))

            # ── Step 1: 深度估计 ──────────────────────────────────────────────
            if 外部深度图 is not None:
                ext = 外部深度图
                if ext.ndim == 3:
                    ext = ext.unsqueeze(0)
                idx       = min(b, ext.shape[0] - 1)
                d_arr     = ext[idx].numpy().astype(np.float32)       # (H', W', C)
                # 统一转 3 通道灰度
                if d_arr.shape[-1] == 1:
                    d_arr = np.repeat(d_arr, 3, axis=-1)
                elif d_arr.shape[-1] != 3:
                    d_arr = d_arr[..., :3]
                pil_d     = Image.fromarray(
                    (d_arr * 255).clip(0, 255).astype(np.uint8)
                ).resize((W, H), Image.LANCZOS)
                depth_arr = np.array(pil_d).astype(np.float32) / 255.0
            else:
                pipe   = self._get_depth_pipe()
                result = pipe(pil_img)
                # predicted_depth: torch Tensor shape [H', W']
                pred   = result["predicted_depth"]
                d_min, d_max = float(pred.min()), float(pred.max())
                if d_max > d_min:
                    pred_norm = (pred - d_min) / (d_max - d_min)
                else:
                    pred_norm = torch.zeros_like(pred)
                pred_np = pred_norm.cpu().numpy().astype(np.float32)  # [H', W']
                pil_d   = Image.fromarray(
                    (pred_np * 255).clip(0, 255).astype(np.uint8), mode="L"
                ).resize((W, H), Image.LANCZOS).convert("RGB")
                depth_arr = np.array(pil_d).astype(np.float32) / 255.0  # [H, W, 3]

            # ── Step 2: Grounding DINO 定位 ───────────────────────────────────
            crop_boxes = []
            if use_text:
                proc, gdino, gd_dev = self._get_gdino()
                gd_text = text if text.endswith(".") else text + "."
                inputs  = proc(images=pil_img, text=gd_text,
                               return_tensors="pt").to(gd_dev)
                with torch.no_grad():
                    outputs = gdino(**inputs)
                raw  = proc.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, target_sizes=[(H, W)]
                )[0]
                keep  = raw["scores"].cpu() >= float(识别严格度)
                boxes = raw["boxes"].cpu()[keep]
                if len(boxes) == 0:
                    print(f"[TextSegmentedDepth] 未检测到 '{text}'，改为全图分割")
                else:
                    pad = 0.08
                    for box in boxes:
                        x1, y1, x2, y2 = box.tolist()
                        pw = (x2 - x1) * pad
                        ph = (y2 - y1) * pad
                        crop_boxes.append((
                            max(0, int(x1 - pw)), max(0, int(y1 - ph)),
                            min(W, int(x2 + pw)), min(H, int(y2 + ph)),
                        ))

            # ── Step 3: BiRefNet 精细分割 ─────────────────────────────────────
            br_model, br_dev = self._get_birefnet()
            mask_full        = np.zeros((H, W), dtype=np.float32)
            br_dtype         = next(br_model.parameters()).dtype
            run_boxes        = crop_boxes if crop_boxes else [(0, 0, W, H)]

            for (x1, y1, x2, y2) in run_boxes:
                pil_crop = pil_img.crop((x1, y1, x2, y2))
                cW, cH   = pil_crop.size
                inp = birefnet_pre(pil_crop).unsqueeze(0).to(device=br_dev, dtype=br_dtype)
                with torch.no_grad():
                    preds = br_model(inp)
                pred = preds[-1].sigmoid().squeeze().cpu().numpy()
                pil_pred = Image.fromarray(
                    (pred * 255).clip(0, 255).astype(np.uint8), mode="L"
                ).resize((cW, cH), Image.LANCZOS)
                mask_full[y1:y2, x1:x2] = np.maximum(
                    mask_full[y1:y2, x1:x2],
                    np.array(pil_pred) / 255.0
                )

            if 反转遮罩:
                mask_full = 1.0 - mask_full

            if 羽化半径 > 0:
                pil_mf = Image.fromarray(
                    (mask_full * 255).astype(np.uint8), mode="L"
                ).filter(ImageFilter.GaussianBlur(radius=羽化半径))
                mask_full = np.array(pil_mf) / 255.0

            # ── Step 4: 深度图 × 遮罩，背景清零 ──────────────────────────────
            masked = depth_arr * mask_full[..., np.newaxis]

            # 预览：绿色半透明叠层标注分割区域
            tint    = np.array([0.10, 0.90, 0.40], dtype=np.float32)
            alpha_v = (mask_full * 0.5)[..., np.newaxis]
            preview = depth_arr * (1.0 - alpha_v) + tint * alpha_v

            masked_depth_out.append(torch.from_numpy(masked.clip(0, 1)).unsqueeze(0))
            depth_out.append(torch.from_numpy(depth_arr.clip(0, 1)).unsqueeze(0))
            mask_out.append(torch.from_numpy(mask_full).unsqueeze(0))
            preview_out.append(torch.from_numpy(preview.clip(0, 1)).unsqueeze(0))

        return (
            torch.cat(masked_depth_out),
            torch.cat(depth_out),
            torch.cat(mask_out),
            torch.cat(preview_out),
        )



# ─────────────────────────────────────────────────────────────────────────────
# 节点 — ImagePadColor（向图像四周扩展纯色边框）
# ─────────────────────────────────────────────────────────────────────────────

class ImagePadColor:
    """
    在原图四周扩展纯色边框（默认黑色），用于补足画布、加边、letterbox 等场景。
    • 上/下/左/右 像素数可独立设置，互不影响
    • color：前端色轮选色，默认 #000000 黑色
    • 输入 RGB → 输出 RGB；输入 RGBA → 输出 RGBA（新增区域 alpha=1，完全不透明）
    • 同时输出新画布的宽高，方便接后续节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":  ("IMAGE",),
                "top":    ("INT", {"default": 200, "min": 0, "max": 8192, "step": 1,
                                   "tooltip": "顶部扩展像素数"}),
                "bottom": ("INT", {"default": 200, "min": 0, "max": 8192, "step": 1,
                                   "tooltip": "底部扩展像素数"}),
                "left":   ("INT", {"default": 200, "min": 0, "max": 8192, "step": 1,
                                   "tooltip": "左侧扩展像素数"}),
                "right":  ("INT", {"default": 200, "min": 0, "max": 8192, "step": 1,
                                   "tooltip": "右侧扩展像素数"}),
                "color":  ("COLOR", {"default": "#000000",
                                     "tooltip": "扩展区域填充色，默认黑色"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION     = "pad"
    CATEGORY     = "Louis_use"
    DESCRIPTION  = "在原图四周扩展指定像素的纯色边框（颜色可选，默认黑）"

    @staticmethod
    def _parse_color(hex_str: str):
        """解析 #rgb 或 #rrggbb → (r, g, b) float 0-1"""
        h = hex_str.strip().lstrip("#")
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return r, g, b

    def pad(self, image: torch.Tensor, top: int, bottom: int,
            left: int, right: int, color: str):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        B, H, W, C = image.shape
        new_H = H + top + bottom
        new_W = W + left + right

        if top == bottom == left == right == 0:
            return (image, int(W), int(H))

        r, g, b = self._parse_color(color)
        # 与输入通道数对齐（3 或 4）；alpha 通道扩展区填 1.0（不透明）
        fill = [r, g, b] + ([1.0] if C == 4 else [])
        fill_t = torch.tensor(fill, dtype=image.dtype, device=image.device)

        canvas = fill_t.view(1, 1, 1, C).expand(B, new_H, new_W, C).clone()
        canvas[:, top:top + H, left:left + W, :] = image

        return (canvas, int(new_W), int(new_H))


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — ImageComposite（图像合成 / 混合模式叠加）
# ─────────────────────────────────────────────────────────────────────────────

class ImageComposite:
    """
    将前景图像叠加到背景图像上。
    • 两路 IMAGE 输入均支持 RGB（3通道）和 RGBA（4通道）透明图
    • foreground_mask（可选）：用遮罩覆盖前景 alpha 通道；有遮罩时前景图的
      原始 alpha 被替换，无遮罩时沿用前景图自带的 alpha（无则视为全不透明）
    • 14 种 Photoshop 标准混合模式
    • opacity 参数控制前景整体不透明度
    • 若背景带 alpha 通道，输出也带 alpha；否则输出 RGB
    • 前景尺寸与背景不一致时自动双线性缩放对齐
    """

    BLEND_MODES = [
        "正常 (Normal)",
        "正片叠底 (Multiply)",
        "滤色 (Screen)",
        "叠加 (Overlay)",
        "柔光 (Soft Light)",
        "强光 (Hard Light)",
        "颜色减淡 (Color Dodge)",
        "颜色加深 (Color Burn)",
        "线性减淡/加 (Linear Dodge)",
        "线性加深 (Linear Burn)",
        "差值 (Difference)",
        "排除 (Exclusion)",
        "变亮 (Lighten)",
        "变暗 (Darken)",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background":  ("IMAGE",),
                "foreground":  ("IMAGE",),
                "blend_mode":  (cls.BLEND_MODES, {"default": "正常 (Normal)"}),
                "opacity":     ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0,
                    "step": 0.01, "display": "slider",
                }),
            },
            "optional": {
                "foreground_mask": ("MASK",),   # 覆盖前景 alpha；(B,H,W) 或 (H,W)，0-1
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "composite"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "将前景叠加到背景，支持遮罩 alpha、RGBA 透明、14 种混合模式及不透明度"

    # ── 混合模式 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _blend(bg: torch.Tensor, fg: torch.Tensor, mode: str) -> torch.Tensor:
        """bg / fg: (B, H, W, 3) float32 0-1；返回混合颜色（不含 alpha）"""
        eps = 1e-7
        if "正常" in mode:
            return fg
        if "正片叠底" in mode:
            return bg * fg
        if "滤色" in mode:
            return 1.0 - (1.0 - bg) * (1.0 - fg)
        if "叠加" in mode:
            return torch.where(bg < 0.5,
                               2.0 * bg * fg,
                               1.0 - 2.0 * (1.0 - bg) * (1.0 - fg))
        if "柔光" in mode:
            # W3C Composite Level 1 — Soft Light
            d = torch.where(bg <= 0.25,
                            ((16.0 * bg - 12.0) * bg + 4.0) * bg,
                            bg.clamp(min=0.0).sqrt())
            return torch.where(fg <= 0.5,
                               bg - (1.0 - 2.0 * fg) * bg * (1.0 - bg),
                               bg + (2.0 * fg - 1.0) * (d - bg))
        if "强光" in mode:
            return torch.where(fg < 0.5,
                               2.0 * bg * fg,
                               1.0 - 2.0 * (1.0 - bg) * (1.0 - fg))
        if "颜色减淡" in mode:
            return (bg / (1.0 - fg + eps)).clamp(0.0, 1.0)
        if "颜色加深" in mode:
            return (1.0 - (1.0 - bg) / (fg + eps)).clamp(0.0, 1.0)
        if "线性减淡" in mode:
            return (bg + fg).clamp(0.0, 1.0)
        if "线性加深" in mode:
            return (bg + fg - 1.0).clamp(0.0, 1.0)
        if "差值" in mode:
            return (bg - fg).abs()
        if "排除" in mode:
            return bg + fg - 2.0 * bg * fg
        if "变亮" in mode:
            return torch.max(bg, fg)
        if "变暗" in mode:
            return torch.min(bg, fg)
        return fg  # 兜底 fallback

    def composite(
        self,
        background:       torch.Tensor,
        foreground:        torch.Tensor,
        blend_mode:        str,
        opacity:           float,
        foreground_mask:   torch.Tensor | None = None,
    ) -> tuple:
        import torch.nn.functional as F

        device = background.device
        foreground = foreground.to(device=device, dtype=background.dtype)

        # ── 批次广播对齐 ─────────────────────────────────────────────────────
        B = max(background.shape[0], foreground.shape[0])
        if background.shape[0] == 1 and B > 1:
            background = background.expand(B, -1, -1, -1)
        if foreground.shape[0] == 1 and B > 1:
            foreground = foreground.expand(B, -1, -1, -1)

        # ── 拆分 RGB / Alpha ─────────────────────────────────────────────────
        has_bg_alpha = background.shape[-1] == 4
        bg_rgb   = background[..., :3]                            # (B,H,W,3)
        bg_alpha = (background[..., 3:4] if has_bg_alpha
                    else torch.ones(B, background.shape[1], background.shape[2], 1,
                                    device=device, dtype=background.dtype))

        fg_rgb   = foreground[..., :3]

        # ── 前景 alpha 来源：外部遮罩 > 图像自带 alpha > 全1 ────────────────
        if foreground_mask is not None:
            # MASK: (B,H,W) 或 (H,W) → 统一到 (B,H,W,1)
            m = foreground_mask.to(device=device, dtype=background.dtype)
            if m.dim() == 2:
                m = m.unsqueeze(0)                    # (1,H,W)
            if m.shape[0] == 1 and B > 1:
                m = m.expand(B, -1, -1)
            fg_alpha = m.unsqueeze(-1)                # (B,H,W,1)
        elif foreground.shape[-1] == 4:
            fg_alpha = foreground[..., 3:4]
        else:
            fg_alpha = torch.ones(B, foreground.shape[1], foreground.shape[2], 1,
                                  device=device, dtype=foreground.dtype)

        # ── 前景尺寸对齐到背景 ───────────────────────────────────────────────
        H, W = background.shape[1], background.shape[2]

        def _resize_hwc(t: torch.Tensor) -> torch.Tensor:
            """(B,H,W,C) → bilinear resize → (B,H,W,C)"""
            return F.interpolate(
                t.permute(0, 3, 1, 2),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        if fg_rgb.shape[1] != H or fg_rgb.shape[2] != W:
            fg_rgb = _resize_hwc(fg_rgb)
        if fg_alpha.shape[1] != H or fg_alpha.shape[2] != W:
            fg_alpha = _resize_hwc(fg_alpha)

        # ── 混合模式 + Alpha 合成（Porter-Duff "over"）───────────────────────
        blended      = self._blend(bg_rgb, fg_rgb, blend_mode).clamp(0.0, 1.0)
        fg_alpha_eff = (fg_alpha * opacity).clamp(0.0, 1.0)

        out_rgb   = (blended * fg_alpha_eff + bg_rgb * (1.0 - fg_alpha_eff)).clamp(0.0, 1.0)
        out_alpha = (fg_alpha_eff + bg_alpha * (1.0 - fg_alpha_eff)).clamp(0.0, 1.0)

        out = torch.cat([out_rgb, out_alpha], dim=-1) if has_bg_alpha else out_rgb
        return (out,)


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — ImageFlipHorizontal
# ─────────────────────────────────────────────────────────────────────────────

class ImageFlipHorizontal:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION     = "flip"
    CATEGORY     = "Louis_use"
    DESCRIPTION  = "水平翻转图像"

    def flip(self, image: torch.Tensor) -> tuple:
        # image shape: (B, H, W, C)
        return (torch.flip(image, dims=[2]),)


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — ImageToGrayscale（黑白转换）
# ─────────────────────────────────────────────────────────────────────────────

class ImageInvert:
    """图像颜色反转（黑变白、白变黑）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "invert"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "图像颜色反转，黑变白、白变黑"

    def invert(self, image: torch.Tensor) -> tuple:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        out = image.clone()
        out[..., :3] = 1.0 - image[..., :3]   # RGB 反转，alpha 不动
        return (out,)


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — TimerStop
# ─────────────────────────────────────────────────────────────────────────────

from collections import deque

# 全局队列（兜底）+ per-node 字典（精确归因）
_bar_queue: deque = deque()
_node_bar_timings: dict = {}           # unique_id (str) → duration (float)
_currently_executing_node_id: str | None = None


def _reset_bar_queue():
    _bar_queue.clear()
    _node_bar_timings.clear()


def _patch_executor():
    """prompt 开始时清空队列；per-node execute 时记录当前 node_id。"""
    try:
        import execution

        # ─── 外层：per-prompt 重置 ───
        _orig_outer = execution.PromptExecutor.execute

        def _hooked_outer(self, prompt, prompt_id, *args, **kwargs):
            _reset_bar_queue()
            return _orig_outer(self, prompt, prompt_id, *args, **kwargs)

        execution.PromptExecutor.execute = _hooked_outer

        # ─── 内层：async def execute(server, dynprompt, caches, current_item, ...) ───
        _orig_inner = execution.execute

        async def _hooked_inner(server, dynprompt, caches, current_item, *args, **kwargs):
            global _currently_executing_node_id
            _currently_executing_node_id = str(current_item) if current_item is not None else None
            try:
                return await _orig_inner(server, dynprompt, caches, current_item, *args, **kwargs)
            finally:
                _currently_executing_node_id = None

        execution.execute = _hooked_inner
        print("[Louis_use] TimerStop: executor 钩子注入成功 ✓")
    except Exception as e:
        print(f"[Louis_use] TimerStop: executor 钩子注入失败（{e}）")


_MIN_STEPS = 4  # 步数低于此的进度条（模型加载等）不计入


def _patch_progress_bar():
    """ProgressBar 完成时将耗时存入 per-node 字典和兜底队列。"""
    try:
        import comfy.utils
        _orig_init   = comfy.utils.ProgressBar.__init__
        _orig_update = comfy.utils.ProgressBar.update_absolute

        def _hooked_init(self, total):
            self._louis_start = time.perf_counter()
            return _orig_init(self, total)

        def _hooked_update(self, value, total=None, preview=None):
            result = _orig_update(self, value, total, preview)
            start  = getattr(self, "_louis_start", None)
            if start is not None and self.total and self.current >= self.total:
                if self.total >= _MIN_STEPS:
                    duration = time.perf_counter() - start
                    _bar_queue.append(duration)
                    nid = _currently_executing_node_id
                    if nid:
                        _node_bar_timings[nid] = duration
                self._louis_start = None
            return result

        comfy.utils.ProgressBar.__init__       = _hooked_init
        comfy.utils.ProgressBar.update_absolute = _hooked_update
        print("[Louis_use] TimerStop: ProgressBar 钩子注入成功 ✓")
    except Exception as e:
        print(f"[Louis_use] TimerStop: ProgressBar 钩子注入失败（{e}）")


_patch_executor()
_patch_progress_bar()


def _find_upstream_timing(unique_id, prompt):
    """BFS 向上游遍历 prompt 图，找到最近的有计时数据的节点，返回其耗时（秒）。"""
    if not unique_id or not prompt:
        return None
    visited: set = set()
    queue = [str(unique_id)]
    first = True
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        if not first and node_id in _node_bar_timings:
            return _node_bar_timings[node_id]
        first = False
        node_data = prompt.get(node_id, {})
        for val in node_data.get("inputs", {}).values():
            if isinstance(val, list) and len(val) >= 2 and isinstance(val[0], (str, int)):
                uid = str(val[0])
                if uid not in visited:
                    queue.append(uid)
    return None


class TimerStop:
    """
    放在 VAEDecode 之后，输出本次 prompt 从开始执行到此节点的耗时。
    elapsed_sec / elapsed_ms 可直接接数学节点做比较或显示。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prefix": ("STRING", {"default": "生成耗时: ", "multiline": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt":    "PROMPT",
            },
        }

    RETURN_TYPES  = ("STRING", "FLOAT")
    RETURN_NAMES  = ("time_str", "elapsed_sec")
    FUNCTION      = "stop"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "输出本次生成的耗时；time_str 供显示，elapsed_sec（秒）供 BatchImageSaver 嵌入元数据"

    def stop(self, image, prefix: str, unique_id=None, prompt=None):
        duration = _find_upstream_timing(unique_id, prompt)
        if duration is None:
            duration = _bar_queue.popleft() if _bar_queue else None
        if duration is None:
            time_str    = f"{prefix}--:--"
            elapsed_sec = 0.0
        else:
            m           = int(duration) // 60
            s           = int(duration) % 60
            time_str    = f"{prefix}{m:02d}:{s:02d}"
            elapsed_sec = round(duration, 2)

        print(f"[TimerStop] {time_str}  ({elapsed_sec}s)")
        return (time_str, elapsed_sec)

    @classmethod
    def IS_CHANGED(cls, image, prefix, unique_id=None, prompt=None):
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — VRAMMonitor
# ─────────────────────────────────────────────────────────────────────────────

class VRAMMonitor:
    """
    接在 VAEDecode 后面，读取本次生成的峰值显存占用。
    vram_gb  → 接 BatchImageSaver 的 vram_gb 输入，自动嵌入 PNG 元数据。
    vram_str → 供文字显示节点使用。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES  = ("FLOAT", "STRING")
    RETURN_NAMES  = ("vram_gb", "vram_str")
    FUNCTION      = "measure"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "测量本次生成的峰值显存占用（GB），接在 VAEDecode 后"

    def measure(self, image):
        try:
            import torch
            if torch.cuda.is_available():
                vram_bytes = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                vram_gb = round(vram_bytes / (1024 ** 3), 2)
            else:
                vram_gb = 0.0
        except Exception as e:
            print(f"[VRAMMonitor] 读取失败: {e}")
            vram_gb = 0.0

        vram_str = f"显存占用: {vram_gb:.2f} GB"
        print(f"[VRAMMonitor] {vram_str}")
        return (vram_gb, vram_str)

    @classmethod
    def IS_CHANGED(cls, image):
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — TimerVRAM（合并计时 + 显存，图片直通，自动写入任意保存节点）
# ─────────────────────────────────────────────────────────────────────────────

class TimerVRAM:
    """
    计时 + 显存二合一节点。
    · 图片直通（image in → image out），放在 VAEDecode 和任意保存节点之间即可。
    · 通过 extra_pnginfo 隐藏机制，自动把 louis_gen_time / louis_vram_gb
      注入到同一 prompt 里所有保存节点（官方 Save Image、BatchImageSaver 等均生效）。
    · 输出 time_str / vram_str 可接文字显示节点，elapsed_sec / vram_gb 可选接或不接。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "prompt":        "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id":     "UNIQUE_ID",
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES  = ("image", "time_str", "vram_str", "elapsed_sec", "vram_gb")
    FUNCTION      = "run"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = (
        "计时 + 显存二合一。图片直通，数据自动嵌入任意保存节点的 PNG 元数据。"
        "time_str / vram_str 可选接文字显示，elapsed_sec / vram_gb 可不接。"
    )

    def run(self, image, prompt=None, extra_pnginfo=None, unique_id=None):
        # ── 显存 ──
        try:
            import torch
            if torch.cuda.is_available():
                vram_bytes = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                vram_gb = round(vram_bytes / (1024 ** 3), 2)
            else:
                vram_gb = 0.0
        except Exception:
            vram_gb = 0.0

        # ── 耗时：优先从 per-node 字典精确取，兜底用队列 ──
        duration = _find_upstream_timing(unique_id, prompt)
        if duration is None:
            duration = _bar_queue.popleft() if _bar_queue else None
        elapsed_sec = round(duration, 2) if duration else 0.0

        # ── 格式化字符串 ──
        if elapsed_sec > 0:
            m, s     = int(elapsed_sec) // 60, int(elapsed_sec) % 60
            time_str = f"生成耗时: {m:02d}:{s:02d}"
        else:
            time_str = "生成耗时: --:--"
        vram_str = f"显存占用: {vram_gb:.2f} GB"

        # ── 注入 extra_pnginfo（任意保存节点都会读取） ──
        if isinstance(extra_pnginfo, dict):
            if elapsed_sec > 0:
                extra_pnginfo["louis_gen_time"] = elapsed_sec
            if vram_gb > 0:
                extra_pnginfo["louis_vram_gb"]  = vram_gb

        print(f"[TimerVRAM] {time_str}  |  {vram_str}")
        return (image, time_str, vram_str, elapsed_sec, vram_gb)

    @classmethod
    def IS_CHANGED(cls, image, prompt=None, extra_pnginfo=None, unique_id=None):
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 节点 — ShowText（在节点上显示文本，方便复制）
# ─────────────────────────────────────────────────────────────────────────────

class ShowText:
    """
    在节点上直接显示输入的字符串，便于查看 / 选中复制（例如 QwenVL 生成的 prompt）。
    · 输入 text（STRING，forceInput），接上游字符串输出
    · 输出 text（STRING）原样透传，可继续接 CLIPTextEncode 等下游节点
    · 节点本身是 OUTPUT_NODE，运行后文本回填到节点显示，并写入工作流
      widgets_values，刷新/重开工作流仍然可见。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id":     "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("text",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE   = True
    FUNCTION      = "show"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "在节点上显示输入文本，方便选中复制（搭配 QwenVL 等文本生成节点）"

    def show(self, text, unique_id=None, extra_pnginfo=None):
        # text 是 list（INPUT_IS_LIST=True）
        texts = list(text) if isinstance(text, list) else [text]
        texts = [str(t) for t in texts]

        # 把文本注入工作流节点的 widgets_values，使下次打开仍可见
        try:
            if unique_id and extra_pnginfo and isinstance(extra_pnginfo, list):
                meta = extra_pnginfo[0]
                if isinstance(meta, dict) and "workflow" in meta:
                    uid = unique_id[0] if isinstance(unique_id, list) else unique_id
                    for n in meta["workflow"].get("nodes", []):
                        if str(n.get("id")) == str(uid):
                            n["widgets_values"] = texts
                            break
        except Exception as e:
            print(f"[Louis_use_ShowText] 写入 widgets_values 失败: {e}")

        return {"ui": {"text": texts}, "result": (texts,)}


from .ideogram4_text_encode import (
    NODE_CLASS_MAPPINGS        as _IDGTE_CLASS,
    NODE_DISPLAY_NAME_MAPPINGS as _IDGTE_DISPLAY,
)

from .ip_copyright_guard import (
    NODE_CLASS_MAPPINGS        as _IPG_CLASS,
    NODE_DISPLAY_NAME_MAPPINGS as _IPG_DISPLAY,
)


NODE_CLASS_MAPPINGS = {
    "Louis_use_ResolutionSelector":    ResolutionSelector,
    "Louis_use_FolderTextLoader":      FolderTextLoader,
    "Louis_use_FolderImageLoader":     FolderImageLoader,
    "Louis_use_SmartAlignCrop":        SmartAlignCrop,
    "Louis_use_ReflectionExtractor":   ReflectionExtractor,
    "Louis_use_DivisibleCrop":         DivisibleCrop,
    "Louis_use_BatchImageSaver":       BatchImageSaver,
    "Louis_use_ColorPaletteExtractor": ColorPaletteExtractor,
    "Louis_use_ColorMatch":            ColorMatch,
    # 无缝贴图工具集
    "Louis_use_SeamlessTileFixer":     SeamlessTileFixer,
    # 文字驱动抠图（CLIPSeg 定位 + BiRefNet 精细抠图）
    "Louis_use_TextSegmenter":         TextSegmenter,
    # 文字驱动目标深度图（Depth-Anything-V2 + GDINO + BiRefNet）
    "Louis_use_TextSegmentedDepth":    TextSegmentedDepth,
    # 图像四周扩展纯色边框
    "Louis_use_ImagePadColor":         ImagePadColor,
    # 图像合成（混合模式叠加）
    "Louis_use_ImageComposite":        ImageComposite,
    # 图像翻转
    "Louis_use_ImageFlipHorizontal":   ImageFlipHorizontal,
    # 黑白转换
    "Louis_use_ImageInvert":            ImageInvert,
    # 性能追踪（计时 + 显存二合一）
    "Louis_use_TimerVRAM":             TimerVRAM,
    # 显示文本（搭配 QwenVL 等文本生成节点）
    "Louis_use_ShowText":              ShowText,
    **_IDGTE_CLASS,
    **_IPG_CLASS,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Louis_use_ResolutionSelector":    "📐 Resolution Selector",
    "Louis_use_FolderTextLoader":      "📄 Folder Text Loader",
    "Louis_use_FolderImageLoader":     "📂 Folder Image Loader",
    "Louis_use_SmartAlignCrop":        "✂️ Smart Align & Crop",
    "Louis_use_ReflectionExtractor":   "✨ Reflection Extractor",
    "Louis_use_DivisibleCrop":         "📐 Divisible Crop",
    "Louis_use_BatchImageSaver":       "🗃️ Batch Image Saver",
    "Louis_use_ColorPaletteExtractor": "🎨 Color Palette Extractor",
    "Louis_use_ColorMatch":            "🎨 Color Match",
    "Louis_use_SeamlessTileFixer":     "🧩 Seamless Tile Fixer",
    "Louis_use_TextSegmenter":         "✂️ Text Segmenter",
    "Louis_use_TextSegmentedDepth":    "🚗 汽车深度图",
    "Louis_use_ImagePadColor":         "🖼️ Image Pad Color",
    "Louis_use_ImageComposite":        "🖼️ Image Composite",
    "Louis_use_ImageFlipHorizontal":   "↔️ Image Flip Horizontal",
    "Louis_use_ImageInvert":            "🔄 Image Invert",
    "Louis_use_TimerVRAM":             "📊 Performance Tracker",
    "Louis_use_ShowText":              "📝 Show Text",
    **_IDGTE_DISPLAY,
    **_IPG_DISPLAY,
}

