# -*- coding: utf-8 -*-
"""
Louis_use — 版权筛选器（IP Copyright Guard）

纯 image → image 的版权检测闸门：
· 输入一张图，用本地 Qwen3-VL 检测是否含黑名单 IP（龙猫、米奇等）。
· 干净   → 原样放行 image，下游正常保存。
· 命中IP → 阻断输出（坏图不往下走、不保存），并发 UI 信号让前端
            自动换种子重新排队，直到生成出干净的图（或达最大重试）。

ComfyUI 工作流无法用连线做循环，循环由配套 JS（js/ip_copyright_guard.js）
监听本节点的执行结果后调用 app.queuePrompt() 完成。
"""

import json
import re
import sys
from pathlib import Path

import torch

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except Exception:
    ExecutionBlocker = None


# ──────────────────────────────────────────────────────────────────────────────
# QwenVL 复用
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
            print(f"[IPGuard] ❌ ComfyUI-QwenVL 不可用: {e}")
            return None


def _extract_json(raw: str) -> str:
    raw = re.sub(r'```(?:json)?\s*', '', raw).strip().strip('`').strip()
    s, e = raw.find('{'), raw.rfind('}')
    return raw[s:e + 1] if s != -1 and e > s else raw


_QUANTIZATIONS = ["4-bit (VRAM-friendly)", "8-bit (Balanced)", "None (FP16)"]


def _detect_ip(image, blacklist, also_general, quantization, max_tokens=256):
    """返回 (found: bool, names: list[str], raw: str)。"""
    qwen = _get_qwen()
    if qwen is None:
        return (False, [], "QwenVL 不可用，跳过检测")

    bl_str = "、".join(blacklist) if blacklist else "（无指定）"
    general_rule = (
        "Also flag ANY other obviously famous copyrighted/trademarked character "
        "(anime, game, film, cartoon mascots, brand logos), even if not in the list.\n"
        if also_general else
        "Only flag characters from the list above; ignore generic/original characters.\n"
    )
    prompt = (
        "You are a strict copyright / intellectual-property detector.\n"
        "Carefully look at the image.\n"
        f"Decide whether it contains any of these copyrighted characters/IP: {bl_str}.\n"
        "Count close look-alikes and re-colored / re-posed versions of the same character too.\n"
        f"{general_rule}"
        "Respond with ONLY compact JSON, no extra words:\n"
        '{"found": true or false, "names": ["detected name", ...]}'
    )

    try:
        result = qwen.run(
            model_name="Qwen3-VL-4B-Instruct",
            quantization=quantization,
            preset_prompt="",
            custom_prompt=prompt,
            image=image, video=None, frame_count=1,
            max_tokens=max_tokens,
            temperature=0.1, top_p=0.9,
            num_beams=1, repetition_penalty=1.1,
            seed=42, keep_model_loaded=False,
            attention_mode="sdpa", use_torch_compile=False, device="auto",
        )
        raw = result[0] if result else ""
    except Exception as e:
        print(f"[IPGuard] ❌ 检测调用失败 {type(e).__name__}: {e}")
        return (False, [], f"检测失败: {e}")

    try:
        data  = json.loads(_extract_json(raw))
        found = bool(data.get("found", False))
        names = data.get("names", []) or []
        if isinstance(names, str):
            names = [names]
        names = [str(n).strip() for n in names if str(n).strip()]
        if names and not found:
            found = True
        return (found, names, raw)
    except Exception:
        low = raw.lower()
        if '"found": true' in low or '"found":true' in low:
            return (True, [], raw)
        return (False, [], raw)


def _cleanup_qwen():
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 节点
# ──────────────────────────────────────────────────────────────────────────────

class IPCopyrightGuard:
    """
    版权筛选器（image → image）。
    放在 VAEDecode 和保存节点之间：
      · 无版权 IP → 原样输出图片
      · 检测到 IP → 阻断输出（不保存），并通知前端自动换种子重生成
    重试循环由配套 JS 完成，需把生成种子的 control_after_generate 设为 randomize。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "版权黑名单": ("STRING", {
                    "multiline": True,
                    "default": "龙猫\n米奇\n皮卡丘\n海绵宝宝\nHello Kitty",
                    "tooltip": "每行一个版权角色/IP 名称；检测到则阻断并重生成",
                }),
                "也拦截其他知名IP": ("BOOLEAN", {"default": True}),
                "检测精度": (_QUANTIZATIONS, {"default": "4-bit (VRAM-friendly)"}),
                "最大重试": ("INT", {
                    "default": 5, "min": 0, "max": 50,
                    "tooltip": "前端最多自动重排队次数，达到后即使有 IP 也放行",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "检测报告")
    FUNCTION     = "guard"
    OUTPUT_NODE  = True
    CATEGORY     = "Louis_use"
    DESCRIPTION  = ("版权筛选器（image→image）：QwenVL 检测画面是否含版权 IP，"
                    "干净放行，命中则阻断输出并通知前端换种子重生成。")

    def guard(self, image, 版权黑名单, 也拦截其他知名IP, 检测精度, 最大重试, unique_id=None):
        blacklist = [ln.strip() for ln in 版权黑名单.splitlines() if ln.strip()]

        found, names, raw = _detect_ip(image, blacklist, 也拦截其他知名IP, 检测精度)
        _cleanup_qwen()

        if found:
            report = f"❌ 命中版权 IP: {', '.join(names) if names else '是'}"
        else:
            report = "✅ 通过：无版权 IP"
        print(f"[IPGuard] {report}")

        # UI 信号：前端据此决定是否重排队
        ui = {
            "ipguard": [json.dumps({
                "found":   found,
                "names":   names,
                "report":  report,
                "max_retry": int(最大重试),
            }, ensure_ascii=False)]
        }

        # 命中 → 阻断 image 输出（下游保存节点被跳过），坏图不落盘
        if found and ExecutionBlocker is not None:
            return {"ui": ui, "result": (ExecutionBlocker(None), report)}

        return {"ui": ui, "result": (image, report)}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "Louis_use_IPCopyrightGuard": IPCopyrightGuard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Louis_use_IPCopyrightGuard": "🛡️ 版权筛选器",
}
