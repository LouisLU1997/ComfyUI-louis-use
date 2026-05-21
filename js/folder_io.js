/**
 * ComfyUI Folder I/O — 前端扩展
 * 缩略图画在 canvas 上，用 capture 阶段 wheel 监听实现滚动（不与 ComfyUI 缩放冲突）
 */
import { app } from "../../scripts/app.js";

const THEME = {
  loader: { color: "#0d3347", bgcolor: "#071e2e" },
  saver:  { color: "#1e0d33", bgcolor: "#120720" },
};

function getWidget(node, name) {
  return node.widgets?.find(w => w.name === name);
}

async function pickFolder(title, initialDir = "") {
  const resp = await fetch("/folder_io/open_dialog", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, initial_dir: initialDir }),
  });
  const data = await resp.json();
  return data.path || null;
}

// ── 布局常量 ──────────────────────────────────────────────────────────────────
const CELL_H      = 63;
const THUMB_GAP   = 4;
const THUMB_PAD   = 6;
const THUMB_ROWS  = 2;
const THUMB_COLS  = 3;
const THUMB_COUNT = 30;

const THUMB_AREA_H =
  THUMB_PAD + THUMB_ROWS * CELL_H + (THUMB_ROWS - 1) * THUMB_GAP + THUMB_PAD; // 142

const NODE_MIN_W =
  THUMB_PAD + THUMB_COLS * 84 + (THUMB_COLS - 1) * THUMB_GAP + THUMB_PAD + 20; // ≈ 290

// ── widget.last_y → 最后一个 widget 的底部 Y（内容区坐标） ───────────────────
function widgetBottom(node) {
  const wh = window.LiteGraph?.NODE_WIDGET_HEIGHT ?? 20;
  let max = -1;
  for (const w of node.widgets ?? []) {
    if (typeof w.last_y === "number") max = Math.max(max, w.last_y + wh);
  }
  return max >= 0 ? max : null;
}

// ── 屏幕坐标 → canvas 世界坐标 ───────────────────────────────────────────────
function screenToWorld(clientX, clientY) {
  const el   = app.canvas.canvas;
  const rect = el.getBoundingClientRect();
  const ds   = app.canvas.ds;
  return [
    (clientX - rect.left) / ds.scale - ds.offset[0],
    (clientY - rect.top)  / ds.scale - ds.offset[1],
  ];
}

// ── canvas 绘制 ───────────────────────────────────────────────────────────────
function setupCanvasThumbs(nodeType) {
  const origFg = nodeType.prototype.onDrawForeground;

  nodeType.prototype.onDrawForeground = function (ctx) {
    origFg?.apply(this, arguments);

    if (!this._thumbAreaH) return;

    const wb     = widgetBottom(this);
    const startY = wb != null ? wb + 4 : this.size[1] - this._thumbAreaH;

    // 高度校正（首帧后 last_y 才可用，此处直接修正，下帧生效）
    if (wb != null) {
      const targetH = startY + this._thumbAreaH;
      if (Math.abs(this.size[1] - targetH) > 2) {
        this.size[1] = targetH;
        app.graph.setDirtyCanvas(true, true);
      }
    }

    // 根据节点宽度自适应格子宽度
    const availW = this.size[0] - THUMB_PAD * 2 - (THUMB_COLS - 1) * THUMB_GAP;
    const cellW  = Math.max(20, Math.floor(availW / THUMB_COLS));

    // 裁剪到缩略图区域
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, startY, this.size[0], this._thumbAreaH);
    ctx.clip();

    // 背景
    ctx.fillStyle = "#0b1825";
    ctx.fillRect(THUMB_PAD, startY, this.size[0] - THUMB_PAD * 2, this._thumbAreaH);

    const cx = this.size[0] / 2;
    const cy = startY + this._thumbAreaH / 2;

    if (this._thumbLoading) {
      ctx.fillStyle = "#4a6a80";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("加载缩略图…", cx, cy);
      ctx.restore();
      return;
    }

    const imgs = this._thumbImages ?? [];
    if (!imgs.length) {
      ctx.fillStyle = "#4a6a80";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("📭 没有找到图片", cx, cy);
      ctx.restore();
      return;
    }

    const scrollY = this._thumbScrollY ?? 0;
    for (let i = 0; i < imgs.length; i++) {
      const col = i % THUMB_COLS;
      const row = Math.floor(i / THUMB_COLS);
      const x   = THUMB_PAD + col * (cellW + THUMB_GAP);
      const y   = startY + THUMB_PAD + row * (CELL_H + THUMB_GAP) - scrollY;
      if (y + CELL_H < startY || y > startY + this._thumbAreaH) continue;

      ctx.fillStyle = "#1a2a3a";
      ctx.fillRect(x, y, cellW, CELL_H);
      const img = imgs[i];
      if (img.complete && img.naturalWidth > 0) {
        try { ctx.drawImage(img, x, y, cellW, CELL_H); } catch (_) {}
      }
    }

    // 滚动条
    const totalRows = Math.ceil(imgs.length / THUMB_COLS);
    if (totalRows > THUMB_ROWS) {
      const totalH    = THUMB_PAD + totalRows * (CELL_H + THUMB_GAP) + THUMB_PAD;
      const maxScroll = totalH - this._thumbAreaH;
      const trackH    = this._thumbAreaH - THUMB_PAD * 2;
      const sbH       = Math.max(14, (this._thumbAreaH / totalH) * trackH);
      const sbY       = startY + THUMB_PAD + (scrollY / maxScroll) * (trackH - sbH);
      ctx.fillStyle = "rgba(140,180,220,0.45)";
      ctx.fillRect(this.size[0] - 7, sbY, 4, sbH);
    }

    ctx.restore();
  };
}

// ── 为每个节点实例绑定 wheel 监听（capture 阶段，优先于 ComfyUI 缩放）─────────
function bindWheelScroll(node) {
  const canvasEl = app.canvas.canvas;

  const handler = (e) => {
    if (!node._thumbAreaH || !node._thumbImages?.length) return;

    // 鼠标世界坐标
    const [wx, wy] = screenToWorld(e.clientX, e.clientY);

    // 是否在节点范围内
    // node.pos[1] 可能是 title 顶或 content 顶，给 ±50 容错
    if (wx < node.pos[0] || wx > node.pos[0] + node.size[0]) return;
    if (wy < node.pos[1] - 60 || wy > node.pos[1] + node.size[1] + 60) return;

    // 换算到内容区 Y（假设 pos[1] 是内容区顶；若含 title 则会偏大，容错在下方）
    const nodeY = wy - node.pos[1];

    // 求缩略图起始 Y（内容区坐标）
    const wb         = widgetBottom(node);
    const thumbStart = wb != null ? wb + 4 : node.size[1] - node._thumbAreaH;

    // 宽松判断：nodeY 超过 thumbStart 的 80%（兼容 title 高度偏移）
    if (nodeY < thumbStart * 0.6) return;
    if (nodeY > node.size[1] + 60) return;

    // 在缩略图区域 → 拦截并滚动
    e.preventDefault();
    e.stopImmediatePropagation();

    const imgs      = node._thumbImages;
    const totalRows = Math.ceil(imgs.length / THUMB_COLS);
    const totalH    = THUMB_PAD + totalRows * (CELL_H + THUMB_GAP) + THUMB_PAD;
    const maxScroll = Math.max(0, totalH - node._thumbAreaH);
    if (!maxScroll) return;

    node._thumbScrollY = Math.max(
      0,
      Math.min(maxScroll, (node._thumbScrollY ?? 0) + (e.deltaY ?? 0) * 0.4)
    );
    app.graph.setDirtyCanvas(true, true);
  };

  // capture:true → 在 ComfyUI 缩放逻辑之前触发
  canvasEl.addEventListener("wheel", handler, { passive: false, capture: true });

  // 节点删除时解绑
  const origRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    canvasEl.removeEventListener("wheel", handler, { capture: true });
    origRemoved?.apply(this, args);
  };
}

// ── 节点尺寸 ──────────────────────────────────────────────────────────────────
function applyNodeSize(node, showThumbs) {
  node._thumbAreaH   = showThumbs ? THUMB_AREA_H : 0;
  node._thumbScrollY = 0;
  const [, baseH] = node.computeSize();
  node.setSize([Math.max(node.size[0], NODE_MIN_W), baseH + node._thumbAreaH]);
  app.graph.setDirtyCanvas(true, true);
}

// ── 加载缩略图 ─────────────────────────────────────────────────────────────────
async function renderThumbs(node, folderPath) {
  node._thumbImages  = [];
  node._thumbLoading = false;
  node._thumbScrollY = 0;

  if (!folderPath?.trim()) {
    applyNodeSize(node, false);
    return;
  }

  node._thumbLoading = true;
  applyNodeSize(node, true);

  try {
    const r = await fetch(
      `/folder_io/preview?path=${encodeURIComponent(folderPath)}&count=${THUMB_COUNT}`
    );
    const d = await r.json();
    node._thumbImages = (d.thumbs ?? []).map((src) => {
      const img  = new Image();
      img.onload = () => app.graph.setDirtyCanvas(true, true);
      img.src    = src;
      return img;
    });
  } catch (err) {
    console.warn("[FolderIO] thumbnail failed:", err);
  } finally {
    node._thumbLoading = false;
    app.graph.setDirtyCanvas(true, true);
  }
}

// ── 状态行 ─────────────────────────────────────────────────────────────────────
// 注意：用 sw.name 而非 sw.value。
// text widget 的 value 在 disabled 状态下颜色极浅，几乎不可见；
// name 由 canvas 直接绘制，始终清晰显示。
async function refreshStatus(node, folderPath) {
  const sw = node._swStatus;          // 直接用存好的引用，不靠名字搜索
  if (!sw) return;
  if (!folderPath?.trim()) {
    sw.name = "📊 未选择文件夹";
    app.graph.setDirtyCanvas(true);
    return;
  }
  sw.name = "⏳ 扫描中…";
  app.graph.setDirtyCanvas(true);
  try {
    const r = await fetch("/folder_io/count", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path: folderPath }),
    });
    const d = await r.json();
    sw.name = d.error
      ? `⚠ ${d.error}`
      : `📷 共 ${d.count} 张  ${d.extensions?.join(" ") ?? ""}`;
  } catch {
    sw.name = "⚠ 无法连接后端";
  }
  app.graph.setDirtyCanvas(true);
}

async function applyFolder(node, folder) {
  const fw = getWidget(node, "folder_path");
  if (fw) fw.value = folder;
  await Promise.all([refreshStatus(node, folder), renderThumbs(node, folder)]);
}

// ══════════════════════════════════════════════════════════════════════════════
app.registerExtension({
  name: "FolderIO.Nodes",

  async beforeRegisterNodeDef(nodeType, nodeData) {

    // ── FolderImageLoader ──────────────────────────────────────────────────
    if (nodeData.name === "Louis_use_FolderImageLoader") {
      Object.assign(nodeType.prototype, THEME.loader);
      setupCanvasThumbs(nodeType);

      const _onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        _onCreated?.apply(this, arguments);

        if (this.size[0] < NODE_MIN_W) this.size[0] = NODE_MIN_W;

        const folderWidget = getWidget(this, "folder_path");

        const btn = this.addWidget("button", "📂 选择文件夹", null, async () => {
          if (btn._busy) return;
          btn._busy = true;
          btn.name  = "⏳ 等待对话框…";
          app.graph.setDirtyCanvas(true);
          try {
            const selected = await pickFolder("选择图片文件夹", folderWidget?.value ?? "");
            if (selected) await applyFolder(this, selected);
          } finally {
            btn._busy = false;
            btn.name  = "📂 选择文件夹";
            app.graph.setDirtyCanvas(true);
          }
        });
        btn.serialize = false;

        const sw = this.addWidget("text", "📊 未选择文件夹", "", () => {});
        sw.disabled  = true;
        sw.serialize = false;
        this._swStatus = sw;   // 存引用，refreshStatus 直接用，不靠名字搜索

        // 绑定滚轮（每个节点实例独立绑定）
        bindWheelScroll(this);

        if (folderWidget) {
          const _orig = folderWidget.callback;
          folderWidget.callback = async (v) => {
            _orig?.call(folderWidget, v);
            await refreshStatus(this, v);
            await renderThumbs(this, v);
          };
          if (folderWidget.value) {
            setTimeout(() => applyFolder(this, folderWidget.value), 300);
          }
        }
      };

      const _onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        _onExecuted?.apply(this, arguments);
        const fw = getWidget(this, "folder_path");
        if (fw?.value) refreshStatus(this, fw.value);
      };
    }

    // ── FolderImageSaver ───────────────────────────────────────────────────
    if (nodeData.name === "Louis_use_FolderImageSaver") {
      Object.assign(nodeType.prototype, THEME.saver);

      const _onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        _onCreated?.apply(this, arguments);

        const folderWidget = getWidget(this, "output_folder");

        const btn = this.addWidget("button", "📂 选择输出文件夹", null, async () => {
          const selected = await pickFolder("选择输出文件夹", folderWidget?.value ?? "");
          if (selected && folderWidget) {
            folderWidget.value = selected;
            app.graph.setDirtyCanvas(true);
          }
        });
        btn.serialize = false;

        const sw = this.addWidget("text", "💾 上次保存", "尚未执行", () => {});
        sw.disabled  = true;
        sw.serialize = false;
      };

      const _onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (msg) {
        _onExecuted?.apply(this, arguments);
        const saved = msg?.output?.saved_folder;
        const sw    = getWidget(this, "💾 上次保存");
        if (sw && saved) {
          const f  = Array.isArray(saved) ? saved[0] : saved;
          sw.value = f.length > 48 ? "…" + f.slice(-46) : f;
          app.graph.setDirtyCanvas(true);
        }
      };
    }
  },
});
