"""
Ideogram 4 CFG Guider for Louis_use.

官方采样预设（来自 ideogram-oss/ideogram4/src/ideogram4/sampler_configs.py）：
  V4_DEFAULT_20 : steps=20, main_cfg=7.0, polish_cfg=3.0, polish_steps=2
  V4_QUALITY_48 : steps=48, main_cfg=7.0, polish_cfg=3.0, polish_steps=3
  V4_TURBO_12   : steps=12, main_cfg=7.0, polish_cfg=3.0, polish_steps=1

实现要点：
  - 正向传递：条件模型 + 文本 → 走 _run_conditional
  - 负向传递：同一模型，强制 c_crossattn=None → 走 _run_image_only（真正无条件）
  - ConditioningZeroOut 方案会让负向仍走有条件分支（零特征），
    配合高 CFG 产生异常中间状态，从而触发安全过滤器。
"""

import comfy.samplers


class _NullContextWrapper:
    """
    包装 BaseModel，强制将 apply_model 的 c_crossattn 置为 None。
    这会触发 Ideogram 4 的 _run_image_only（真正的无条件）分支。
    """
    def __init__(self, model):
        self._model = model

    def apply_model(self, x, t, c_concat=None, c_crossattn=None,
                    control=None, transformer_options={}, **extra_conds):
        return self._model.apply_model(
            x, t,
            c_concat=c_concat,
            c_crossattn=None,           # 强制无条件
            control=control,
            transformer_options=transformer_options,
            **extra_conds,
        )

    def __getattr__(self, name):
        return getattr(self._model, name)


class _Ideogram4CFGGuider(comfy.samplers.CFGGuider):
    """
    Ideogram 4 专用 CFG 引导器。

    正向：条件模型 + 文本，走 _run_conditional。
    负向：同一模型，c_crossattn=None，走 _run_image_only（真正无条件）。

    CFG 公式：denoised = uncond + cfg × (cond − uncond)
    CFG 值根据 sigma 位置动态切换：主阶段用较高值，最后 polish_steps 步用较低值。
    """

    def set_conds(self, positive):
        # 只需要正向条件，负向由内部无条件传递实现
        self.inner_set_conds({"positive": positive})

    def set_schedule(self, main_cfg: float, polish_cfg: float, polish_steps: int):
        self._main_cfg     = main_cfg
        self._polish_cfg   = polish_cfg
        self._polish_steps = polish_steps

    def _resolve_cfg(self, timestep, model_options: dict) -> float:
        sigmas = model_options.get("transformer_options", {}).get("sample_sigmas")
        if sigmas is None or len(sigmas) <= 1:
            return self._main_cfg

        num_steps = len(sigmas) - 1
        if self._polish_steps <= 0 or num_steps <= self._polish_steps:
            return self._main_cfg

        threshold_sigma = float(sigmas[num_steps - self._polish_steps])
        current_sigma   = float(timestep.max())
        return self._polish_cfg if current_sigma <= threshold_sigma else self._main_cfg

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        cfg = self._resolve_cfg(timestep, model_options)

        # ── 正向：条件模型 + 文本（_run_conditional）─────────────────────
        cond_pred = comfy.samplers.sampling_function(
            self.inner_model, x, timestep,
            None,
            self.conds.get("positive"),
            1.0,                        # cfg=1 取原始条件预测
            model_options=model_options,
            seed=seed,
        )

        # ── 负向：同一模型，context=None（_run_image_only）───────────────
        uncond_wrapper = _NullContextWrapper(self.inner_model)
        uncond_pred = comfy.samplers.sampling_function(
            uncond_wrapper, x, timestep,
            None,
            self.conds.get("positive"),  # positive cond 用于维持 batch_size，会被剥离
            1.0,
            model_options=model_options,
            seed=seed,
        )

        # ── CFG 合成 ──────────────────────────────────────────────────
        return uncond_pred + cfg * (cond_pred - uncond_pred)


class Louis_use_Ideogram4Guider:
    """
    Ideogram 4 CFG 引导器

    ✓ 负向传递自动使用模型内置的无条件分支（_run_image_only），
      不再需要 ConditioningZeroOut，避免安全过滤器误触发。

    ✓ 逐步 CFG 调度：前几步高 CFG 快速收敛，最后 polish_steps 步
      切换到低 CFG 细化细节，与官方推理代码行为一致。

    接线方式：
      model    ← UNETLoader（ideogram4_fp8_scaled.safetensors）
      positive ← CLIPTextEncode
      guider   → SamplerCustomAdvanced
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("MODEL",),
                "positive": ("CONDITIONING",),
                "main_cfg": ("FLOAT", {
                    "default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "主阶段 CFG，官方默认 7.0",
                }),
                "polish_cfg": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Polish 阶段 CFG，官方默认 3.0",
                }),
                "polish_steps": ("INT", {
                    "default": 2, "min": 0, "max": 20,
                    "tooltip": "末尾 polish 步数：20步→2，48步→3，12步→1",
                }),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION     = "get_guider"
    CATEGORY     = "Louis_use/ideogram4"

    def get_guider(self, model, positive, main_cfg, polish_cfg, polish_steps, **kwargs):
        # negative 参数由旧工作流传入时自动忽略（内部使用无条件分支替代）
        guider = _Ideogram4CFGGuider(model)
        guider.set_conds(positive)
        guider.set_schedule(main_cfg, polish_cfg, polish_steps)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "Louis_use_Ideogram4Guider": Louis_use_Ideogram4Guider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Louis_use_Ideogram4Guider": "🎯 Ideogram4 CFG Guider",
}
