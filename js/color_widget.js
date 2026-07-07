/**
 * Louis_use — COLOR 色轮选色器 Widget
 *
 * 为所有使用 ("COLOR", {"default": "#rrggbb"}) 类型的节点
 * 注册一个可点击的色块 Widget，点击后弹出浏览器原生色轮。
 */
import { app } from "../../scripts/app.js";

const COLOR_TYPE = "COLOR";

/** 根据 hex 颜色亮度决定前景字色 */
function contrastColor(hex) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return (r * 299 + g * 587 + b * 114) / 1000 > 128 ? "#000000" : "#ffffff";
}

/** 构建 COLOR widget 对象 */
function makeColorWidget(name, defaultVal) {
  const widget = {
    name,
    type: COLOR_TYPE,
    value: defaultVal || "#ffffff",
    y: 0,
    last_y: 0,
    last_h: 32,

    draw(ctx, node, widgetWidth, widgetY, height) {
      // 如果前面已有同名 COLOR widget，说明自己是副本，跳过绘制
      const ws = node.widgets || [];
      for (let _i = 0; _i < ws.length; _i++) {
        if (ws[_i] === this) break;                            // 自己是最早的，正常画
        if (ws[_i].type === COLOR_TYPE && ws[_i].name === this.name) return; // 是副本，跳过
      }

      this.last_y = widgetY;
      this.last_h = height;

      const b = 3; // border
      // 黑色边框
      ctx.fillStyle = "#000";
      ctx.fillRect(0, widgetY, widgetWidth, height);
      // 选中颜色填充
      ctx.fillStyle = this.value;
      ctx.fillRect(b, widgetY + b, widgetWidth - b * 2, height - b * 2);
      // 标签 + hex 值
      ctx.fillStyle = contrastColor(this.value);
      ctx.font = "bold 12px Arial";
      ctx.textAlign = "center";
      ctx.fillText(
        `${this.name}  ${this.value.toUpperCase()}`,
        widgetWidth * 0.5,
        widgetY + height / 2 + 4
      );
    },

    mouse(e, pos, node) {
      if (
        e.type !== "pointerdown" &&
        e.type !== "mousedown" &&
        e.type !== "click"
      )
        return;

      // 只响应在本 widget 纵向范围内的点击
      const y0 = this.last_y;
      const y1 = y0 + this.last_h;
      if (pos[1] < y0 || pos[1] > y1) return;

      // 创建隐藏的 <input type="color"> 并触发
      const picker = document.createElement("input");
      picker.type = "color";
      picker.value = this.value;
      Object.assign(picker.style, {
        position: "fixed",
        left: `${e.clientX}px`,
        top: `${e.clientY}px`,
        opacity: "0",
        width: "1px",
        height: "1px",
        pointerEvents: "none",
        zIndex: "9999",
      });
      picker.setAttribute("aria-hidden", "true");
      document.body.appendChild(picker);

      const self = this;
      // 实时预览
      picker.addEventListener("input", () => {
        self.value = picker.value;
        node.setDirtyCanvas(true, true);
      });
      // 确认选色
      picker.addEventListener("change", () => {
        self.value = picker.value;
        if (node.graph) node.graph._version++;
        node.setDirtyCanvas(true, true);
        picker.remove();
      });
      // 取消
      picker.addEventListener("blur", () => picker.remove());

      e.stopPropagation?.();
      e.preventDefault?.();

      if (typeof picker.showPicker === "function") {
        try {
          picker.showPicker();
        } catch {
          picker.click();
        }
      } else {
        picker.click();
      }
    },

    computeSize(width) {
      return [width, 32];
    },
  };

  return widget;
}

// ══════════════════════════════════════════════════════════════════════════════
// 注册扩展
// ══════════════════════════════════════════════════════════════════════════════

app.registerExtension({
  name: "Louis_use.ColorWidget",

  getCustomWidgets() {
    return {
      /**
       * COLOR 工厂：
       * layerstyle 的 ensureColorWidgets 在 onNodeCreated 里也会添加 COLOR widget，
       * 如果它先跑（onNodeCreated 先于工厂），则 node.widgets 里已有该 widget，
       * 直接返回已有的，避免重复。如果工厂先跑则正常创建。
       */
      [COLOR_TYPE](node, inputName, inputData) {
        const existing = (node.widgets || []).find(
          (w) => w.name === inputName && w.type === COLOR_TYPE
        );
        if (existing) return { widget: existing, minWidth: 150, minHeight: 32 };

        const defaultVal = inputData?.[1]?.default || "#ffffff";
        const widget = makeColorWidget(inputName, defaultVal);
        return {
          widget: node.addCustomWidget(widget),
          minWidth: 150,
          minHeight: 32,
        };
      },
    };
  },

  /**
   * beforeRegisterNodeDef：对含 COLOR 输入的节点，三层去重防止重复色块：
   *   1. 实例级 addCustomWidget 拦截（从源头阻止）
   *   2. origOnCreated 结束后立即同步 splice
   *   3. setTimeout(0) 异步再 splice 一次
   */
  beforeRegisterNodeDef(nodeType, nodeData) {
    const required = nodeData?.input?.required || {};
    const hasColor = Object.values(required).some((v) => v?.[0] === COLOR_TYPE);
    if (!hasColor) return;

    const origOnCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      // ── 实例级 addCustomWidget 拦截 ──────────────────────────────────────
      const origAdd = this.addCustomWidget.bind(this);
      this.addCustomWidget = function (widget) {
        if (widget?.type === COLOR_TYPE) {
          const exists = (this.widgets || []).some(
            (w) => w.type === COLOR_TYPE && w.name === widget.name
          );
          if (exists) return widget;
        }
        return origAdd(widget);
      };

      const result = origOnCreated?.apply(this, arguments);

      // ── 同步去重 ──────────────────────────────────────────────────────────
      if (this.widgets) {
        const seen = new Set();
        for (let i = 0; i < this.widgets.length; i++) {
          const w = this.widgets[i];
          if (w.type === COLOR_TYPE) {
            if (seen.has(w.name)) {
              this.widgets.splice(i--, 1);
            } else {
              seen.add(w.name);
            }
          }
        }
      }

      // ── 异步去重 + 尺寸修正 ──────────────────────────────────────────────
      const self = this;
      setTimeout(() => {
        if (self.widgets) {
          const seen2 = new Set();
          for (let i = 0; i < self.widgets.length; i++) {
            const w = self.widgets[i];
            if (w.type === COLOR_TYPE) {
              if (seen2.has(w.name)) {
                self.widgets.splice(i--, 1);
              } else {
                seen2.add(w.name);
              }
            }
          }
        }
        self.setSize?.(self.computeSize());
        app.graph?.setDirtyCanvas(true, true);
      }, 0);

      return result;
    };
  },
});
