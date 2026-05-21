"""
ComfyUI Folder I/O Nodes
- FolderImageLoader: 从文件夹批量加载图片
- FolderImageSaver:  保存图片到带前缀的文件夹
"""

import os
import re
import json
import random
import hashlib
import time
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

    RETURN_TYPES   = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES   = ("image", "mask", "filename")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION       = "load_images"
    CATEGORY       = "Louis_use"
    DESCRIPTION    = "读取文件夹内全部图片，逐张送入下游节点；图片总数显示在节点状态栏"

    def load_images(self, folder_path: str):
        paths = collect_images(folder_path, "name")
        total = len(paths)

        if total == 0:
            raise ValueError(f"文件夹中没有找到支持格式的图片: {folder_path}")

        images, masks, filenames = [], [], []
        for p in paths:
            try:
                img = Image.open(p)
                if hasattr(img, "n_frames") and img.n_frames > 1:
                    img.seek(0)
                images.append(pil_to_tensor(img))
                masks.append(pil_to_mask(img))
                filenames.append(os.path.basename(p))
            except Exception as e:
                print(f"[FolderImageLoader] 跳过损坏图片 {p}: {e}")

        if not images:
            raise RuntimeError("所有图片均无法读取，请检查文件完整性")

        print(f"[FolderImageLoader] 共 {total} 张，逐张送入下游节点")
        return (images, masks, filenames)

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
        "SD1.5 (8)":   8,
        "SDXL (8)":    8,
        "SD3 (16)":    16,
        "Flux (16)":   16,
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
                "images": ("IMAGE",),
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
            },
            "optional": {
                "原文件名": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "接 Folder Image Loader 的 filename 输出，自动用原文件名命名",
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
        images: torch.Tensor,
        输出文件夹: str,
        文件格式: str = "png",
        质量: int = 100,
        嵌入工作流: bool = True,
        原文件名: str = "",
        prompt=None,
        extra_pnginfo=None,
    ):
        if not 输出文件夹.strip():
            raise ValueError("输出文件夹 不能为空")

        out_dir = self._resolve_dir(输出文件夹)

        if images.ndim == 3:
            images = images.unsqueeze(0)
        N = images.shape[0]

        # 原文件名可能是单个字符串或换行分隔多个
        src_names: list[str] = []
        if 原文件名:
            src_names = [s.strip() for s in 原文件名.splitlines() if s.strip()]

        ext = 文件格式.lower()
        saved: list[str] = []

        for i in range(N):
            pil = tensor_to_pil(images[i])

            # ── 决定文件名：有原文件名则同步，否则按编号 ──
            if src_names:
                src  = src_names[i] if i < len(src_names) else src_names[-1]
                stem = Path(src).stem
            else:
                stem = f"img_{i + 1:05d}"

            # ── 同名自动加 _1 _2 后缀防覆盖 ──
            fpath = out_dir / f"{stem}.{ext}"
            k = 1
            while fpath.exists():
                fpath = out_dir / f"{stem}_{k}.{ext}"
                k += 1

            # ── 保存参数 ──
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
                        meta.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for k2, v2 in extra_pnginfo.items():
                            meta.add_text(k2, json.dumps(v2))
                    kw["pnginfo"] = meta

            pil.save(str(fpath), **kw)
            saved.append(str(fpath))
            print(f"[BatchImageSaver] 已保存: {fpath}")

        print(f"[BatchImageSaver] 共保存 {len(saved)} 张 → {out_dir}")
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
    _GRID_SIZES    = ["1×1", "2×2", "3×3", "4×4"]
    _BLEND_LEVELS  = ["柔和", "标准", "强"]
    _BLEND_RATIOS  = {"柔和": 0.05, "标准": 0.10, "强": 0.20}
    _GRID_COUNTS   = {"1×1": 1, "2×2": 2, "3×3": 3, "4×4": 4}

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
    DESCRIPTION   = "偏移50%+交叉混合，生成四边无缝平铺贴图，附 N×N 平铺预览"

    def fix(self, image: torch.Tensor, 方法: str, 预览格数: str, 混合强度: str):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        blend_ratio = self._BLEND_RATIOS.get(混合强度, 0.10)
        grid_n      = self._GRID_COUNTS.get(预览格数, 3)

        results, previews = [], []
        for i in range(image.shape[0]):
            arr = image[i].numpy().astype(np.float32)

            seamless = self._offset_blend(arr, blend_ratio)
            results.append(torch.from_numpy(seamless).unsqueeze(0))
            previews.append(torch.from_numpy(
                self._tiled_preview(seamless, grid_n)
            ).unsqueeze(0))

        return (torch.cat(results), torch.cat(previews))

    @staticmethod
    def _offset_blend(arr: np.ndarray, blend_ratio: float) -> np.ndarray:
        """
        1. 将图像在 X/Y 各偏移 50%（接缝移到中央）
        2. 以余弦权重对中央接缝两侧对称像素对做交叉混合
        3. 结果四边无缝，中央过渡平滑
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
            # 越靠近接缝权重越大（mix 更多对方内容）
            t = (i + 0.5) / bh          # 0→1，从接缝处到混合边界
            w = 0.5 - 0.5 * math.cos(math.pi * (1.0 - t))   # 接缝处 w≈1，边界处 w≈0
            a = result[row_a].copy()
            b = result[row_b].copy()
            result[row_a] = (1.0 - w) * a + w * b
            result[row_b] = (1.0 - w) * b + w * a

        # 修复垂直接缝
        for i in range(bw):
            col_a = (cx - i - 1) % W
            col_b = (cx + i)     % W
            t = (i + 0.5) / bw
            w = 0.5 - 0.5 * math.cos(math.pi * (1.0 - t))
            a = result[:, col_a].copy()
            b = result[:, col_b].copy()
            result[:, col_a] = (1.0 - w) * a + w * b
            result[:, col_b] = (1.0 - w) * b + w * a

        return result.clip(0.0, 1.0)

    @staticmethod
    def _tiled_preview(arr: np.ndarray, n: int) -> np.ndarray:
        """将无缝贴图平铺成 n×n，缩放到不超过 1024px。"""
        tiled = np.tile(arr, (n, n, 1))
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
                        pw = (x2 - x1) * pad;  ph = (y2 - y1) * pad
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
# 导出
# ─────────────────────────────────────────────────────────────────────────────



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
# 节点 — TimerStop
# ─────────────────────────────────────────────────────────────────────────────

# 每个完成的 ProgressBar 耗时按顺序入队，TimerStop 依次出队
from collections import deque
_bar_queue: deque = deque()


def _reset_bar_queue():
    _bar_queue.clear()


def _patch_executor():
    """prompt 开始时清空队列。"""
    try:
        import execution
        _orig = execution.PromptExecutor.execute

        def _hooked(self, prompt, prompt_id, *args, **kwargs):
            _reset_bar_queue()
            return _orig(self, prompt, prompt_id, *args, **kwargs)

        execution.PromptExecutor.execute = _hooked
        print("[Louis_use] TimerStop: executor 钩子注入成功 ✓")
    except Exception as e:
        print(f"[Louis_use] TimerStop: executor 钩子注入失败（{e}）")


_MIN_STEPS = 4  # 步数低于此的进度条（模型加载等）不计入队列


def _patch_progress_bar():
    """每个 ProgressBar 完成时将耗时入队，开始时间存在实例上避免全局冲突。"""
    try:
        import comfy.utils
        _orig_init = comfy.utils.ProgressBar.__init__
        _orig_update = comfy.utils.ProgressBar.update_absolute

        def _hooked_init(self, total):
            self._louis_start = time.perf_counter()
            return _orig_init(self, total)

        def _hooked_update(self, value, total=None, preview=None):
            result = _orig_update(self, value, total, preview)
            start = getattr(self, "_louis_start", None)
            if start is not None and self.total and self.current >= self.total:
                if self.total >= _MIN_STEPS:
                    _bar_queue.append(time.perf_counter() - start)
                self._louis_start = None
            return result

        comfy.utils.ProgressBar.__init__ = _hooked_init
        comfy.utils.ProgressBar.update_absolute = _hooked_update
        print("[Louis_use] TimerStop: ProgressBar 钩子注入成功 ✓")
    except Exception as e:
        print(f"[Louis_use] TimerStop: ProgressBar 钩子注入失败（{e}）")


_patch_executor()
_patch_progress_bar()


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
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("time_str",)
    FUNCTION      = "stop"
    CATEGORY      = "Louis_use"
    DESCRIPTION   = "输出本次生成的耗时（秒/毫秒），接在 VAEDecode 后面使用"

    def stop(self, image, prefix: str):
        duration = _bar_queue.popleft() if _bar_queue else None
        if duration is None:
            time_str = f"{prefix}--:--"
        else:
            m = int(duration) // 60
            s = int(duration) % 60
            time_str = f"{prefix}{m:02d}:{s:02d}"

        print(f"[TimerStop] {time_str}")
        return (time_str,)

    @classmethod
    def IS_CHANGED(cls, image, prefix):
        return float("nan")  # 每次都重新执行


NODE_CLASS_MAPPINGS = {
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
    # 图像翻转
    "Louis_use_ImageFlipHorizontal":   ImageFlipHorizontal,
    # 生成耗时计时
    "Louis_use_TimerStop":             TimerStop,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Louis_use_FolderImageLoader":     "📂 Folder Image Loader",
    "Louis_use_SmartAlignCrop":        "✂️ Smart Align & Crop",
    "Louis_use_ReflectionExtractor":   "✨ Reflection Extractor",
    "Louis_use_DivisibleCrop":         "📐 Divisible Crop",
    "Louis_use_BatchImageSaver":       "🗃️ Batch Image Saver",
    "Louis_use_ColorPaletteExtractor": "🎨 Color Palette Extractor",
    "Louis_use_ColorMatch":            "🎨 Color Match",
    "Louis_use_SeamlessTileFixer":     "🧩 Seamless Tile Fixer",
    "Louis_use_TextSegmenter":         "✂️ Text Segmenter",
    "Louis_use_TextSegmentedDepth":    "🚗 Text Segmented Depth",
    "Louis_use_ImageFlipHorizontal":   "↔️ Image Flip Horizontal",
    "Louis_use_TimerStop":             "⏱️ Timer Stop",
}
