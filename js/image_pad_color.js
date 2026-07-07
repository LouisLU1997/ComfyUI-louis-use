/**
 * Louis_use — ImagePadColor 实时预览（DOM Canvas 版）
 *
 * 给 Louis_use_ImagePadColor 节点加一块 220px 高的 canvas，调参时即时
 * 显示「彩色边框 + 原图」的合成效果，无需点运行。
 *
 * 限制：节点连上输入图后需要至少运行过一次工作流，浏览器才能从 /temp/
 * 拿到图像缩略图；此后所有滑块/颜色调整都是即时绘制。
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "Louis_use_ImagePadColor";
const PREVIEW_HEIGHT = 220;

/** 从上游节点最近一次执行的输出里取第一张图，返回 Promise<HTMLImageElement|null> */
function fetchUpstreamImage(node) {
  try {
    const linkId = node.inputs?.[0]?.link;
    if (linkId == null) return Promise.resolve(null);
    const link = app.graph.links[linkId];
    if (!link) return Promise.resolve(null);
    const out = app.nodeOutputs?.[link.origin_id];
    const meta = out?.images?.[0];
    if (!meta) return Promise.resolve(null);

    const url = api.apiURL(
      "/view?filename=" + encodeURIComponent(meta.filename) +
      "&subfolder=" + encodeURIComponent(meta.subfolder || "") +
      "&type=" + (meta.type || "temp") +
      "&t=" + Date.now()
    );
    return new Promise((resolve) => {
      const img = new Image();
      img.onload  = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = url;
    });
  } catch (e) {
    return Promise.resolve(null);
  }
}

function getW(node, name, fallback = 0) {
  const w = node.widgets?.find((w) => w.name === name);
  return w ? w.value : fallback;
}

app.registerExtension({
  name: "Louis_use.ImagePadColor.LivePreview",
  async nodeCreated(node) {
    if (node.comfyClass !== NODE_TYPE) return;

    // ── DOM 容器 ──────────────────────────────────────────────────────────
    const wrap = document.createElement("div");
    Object.assign(wrap.style, {
      position:     "relative",
      width:        "100%",
      height:       PREVIEW_HEIGHT + "px",
      background:   "#1a1a1a",
      borderRadius: "4px",
      overflow:     "hidden",
      boxSizing:    "border-box",
    });

    const cv = document.createElement("canvas");
    Object.assign(cv.style, {
      position: "absolute",
      inset:    "0",
      width:    "100%",
      height:   "100%",
      display:  "block",
    });
    wrap.appendChild(cv);

    const label = document.createElement("div");
    Object.assign(label.style, {
      position:      "absolute",
      bottom:        "4px",
      left:          "0",
      width:         "100%",
      textAlign:     "center",
      color:         "#cfcfcf",
      font:          "11px Arial",
      pointerEvents: "none",
      textShadow:    "0 0 3px #000, 0 0 3px #000",
    });
    label.textContent = "连接输入图并运行一次后，调参实时预览";
    wrap.appendChild(label);

    // ── 注册为 DOM widget（高度由 DOM 撑开，节点自动留位置）────────────
    const widget = node.addDOMWidget("preview", "image_pad_preview", wrap, {
      serialize:   false,
      hideOnZoom:  false,
    });
    widget.computeSize = () => [0, PREVIEW_HEIGHT];

    node._padImg = null;

    // ── 绘制函数 ──────────────────────────────────────────────────────────
    function redraw() {
      const ctx = cv.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const W = Math.max(1, wrap.clientWidth  || node.size?.[0] || 200);
      const H = Math.max(1, wrap.clientHeight || PREVIEW_HEIGHT);

      if (cv.width !== W * dpr || cv.height !== H * dpr) {
        cv.width  = W * dpr;
        cv.height = H * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const img = node._padImg;
      if (!img || !img.naturalWidth) {
        label.textContent = "连接输入图并运行一次后，调参实时预览";
        return;
      }

      const imgW  = img.naturalWidth;
      const imgH  = img.naturalHeight;
      const top   = Number(getW(node, "top",    0)) || 0;
      const bot   = Number(getW(node, "bottom", 0)) || 0;
      const left  = Number(getW(node, "left",   0)) || 0;
      const right = Number(getW(node, "right",  0)) || 0;
      const color = String(getW(node, "color", "#000000"));

      const newW = imgW + left + right;
      const newH = imgH + top + bot;

      const padPx  = 8;
      const labelH = 18;
      const availW = W - padPx * 2;
      const availH = H - padPx * 2 - labelH;
      const scale  = Math.min(availW / newW, availH / newH);
      const drawW  = newW * scale;
      const drawH  = newH * scale;
      const x0     = (W - drawW) / 2;
      const y0     = padPx + (availH - drawH) / 2;

      // 棋盘格底（视觉对照，便于看清纯黑边框范围）
      drawChecker(ctx, x0, y0, drawW, drawH);

      // 纯色边框
      ctx.fillStyle = color;
      ctx.fillRect(x0, y0, drawW, drawH);

      // 原图
      ctx.drawImage(
        img,
        x0 + left * scale,
        y0 + top  * scale,
        imgW * scale,
        imgH * scale
      );

      // 描边
      ctx.strokeStyle = "#555";
      ctx.lineWidth   = 1;
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, drawW - 1, drawH - 1);

      label.textContent = `${newW} × ${newH}  (原图 ${imgW} × ${imgH})`;
    }

    function drawChecker(ctx, x, y, w, h) {
      const s = 8;
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      for (let i = 0; i < Math.ceil(w / s); i++) {
        for (let j = 0; j < Math.ceil(h / s); j++) {
          ctx.fillStyle = (i + j) % 2 ? "#2a2a2a" : "#222";
          ctx.fillRect(x + i * s, y + j * s, s, s);
        }
      }
      ctx.restore();
    }

    // ── 监听滑块/颜色 widget 变化 → 即时重绘 ─────────────────────────────
    const LIVE = new Set(["top", "bottom", "left", "right", "color"]);
    for (const w of node.widgets) {
      if (!LIVE.has(w.name)) continue;
      const orig = w.callback;
      w.callback = function (v) {
        if (orig) orig.call(this, v);
        redraw();
      };
    }

    // ── 拉取上游图：每次工作流执行完都刷新一次 ─────────────────────────
    async function refresh() {
      const img = await fetchUpstreamImage(node);
      if (img) {
        node._padImg = img;
        redraw();
      }
    }
    const onExec = () => refresh();
    api.addEventListener("executed", onExec);
    api.addEventListener("execution_success", onExec);

    // ── 输入连线变更：清缓存 ───────────────────────────────────────────
    const origConn = node.onConnectionsChange;
    node.onConnectionsChange = function (type, slot, isConnected, link, ioSlot) {
      if (origConn) origConn.call(this, type, slot, isConnected, link, ioSlot);
      if (type === 1 && slot === 0) {
        node._padImg = null;
        redraw();
      }
    };

    // 节点宽度变化时自动重绘（DPR / 容器尺寸响应）
    const ro = new ResizeObserver(() => redraw());
    ro.observe(wrap);

    // 初次尝试拉取（页面刷新后 app.nodeOutputs 可能已有缓存）
    setTimeout(() => { refresh(); redraw(); }, 200);

    // 清理
    const origRemove = node.onRemoved;
    node.onRemoved = function () {
      api.removeEventListener("executed", onExec);
      api.removeEventListener("execution_success", onExec);
      try { ro.disconnect(); } catch (e) {}
      if (origRemove) origRemove.call(this);
    };
  },
});
