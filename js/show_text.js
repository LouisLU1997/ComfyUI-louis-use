/**
 * Louis_use — ShowText
 *
 * 把上游字符串显示在节点上：
 *   - 用 ComfyUI 原生 STRING multiline widget（主题一致、可选中、可复制）
 *   - 只读 + 半透明，视觉上和「输入框」区分
 *   - 支持 list 多段：N 段文本就生成 N 个 widget
 *   - 文本由后端写入 widgets_values 持久化，重开工作流照样显示
 *   - 节点自动 resize 适应文本长度
 */
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_TYPE = "Louis_use_ShowText";

app.registerExtension({
  name: "Louis_use.ShowText",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    /** 清掉之前生成的显示 widget（保留 forceInput 转换出的隐藏 widget，若有） */
    function clearDynamicWidgets(node) {
      if (!node.widgets) return 0;
      const baseCount = node.inputs?.[0]?.widget ? 1 : 0;
      for (let i = baseCount; i < node.widgets.length; i++) {
        node.widgets[i].onRemove?.();
      }
      node.widgets.length = baseCount;
      return baseCount;
    }

    /** 用一组字符串重建显示 widget */
    function render(node, items) {
      const baseCount = clearDynamicWidgets(node);

      // forceInput 在某些前端版本下 widgets_values 首位会是占位，需要跳过
      const list = Array.isArray(items) ? [...items] : [items];
      if (baseCount && list.length > 1 && !list[0]) list.shift();

      const flat = [];
      for (const entry of list) {
        if (Array.isArray(entry)) flat.push(...entry);
        else flat.push(entry);
      }

      for (const txt of flat) {
        const name = "text_" + (node.widgets?.length ?? 0);
        const created = ComfyWidgets.STRING(
          node, name, ["STRING", { multiline: true }], app
        );
        const w = created.widget;
        w.value = txt == null ? "" : String(txt);
        // 只读 + 半透明（视觉上和编辑框区分）
        if (w.inputEl) {
          w.inputEl.readOnly = true;
          w.inputEl.style.opacity = "0.7";
          w.inputEl.style.cursor = "text";
        }
      }

      // 下一帧再算尺寸，让 widget 先布局好
      requestAnimationFrame(() => {
        const sz = node.computeSize();
        sz[0] = Math.max(sz[0], node.size[0]);
        sz[1] = Math.max(sz[1], node.size[1]);
        node.onResize?.(sz);
        app.graph.setDirtyCanvas(true, false);
      });
    }

    // ── 执行完成时收到 ui.text ─────────────────────────────────────────
    const origExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      origExecuted?.apply(this, arguments);
      if (message?.text != null) render(this, message.text);
    };

    // ── 工作流恢复：保存初始 widgets_values，恢复完再渲染 ──────────────
    const SAVED = Symbol("Louis_use_ShowText.savedValues");

    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      // 新版前端会在 configure 里清空 widgets_values，所以先抢一份
      this[SAVED] = info?.widgets_values;
      return origConfigure?.apply(this, arguments);
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      origOnConfigure?.apply(this, arguments);
      const saved = this[SAVED];
      if (!saved || !saved.length) return;
      // forceInput 占位的处理：转换 widget 存在时跳过首项
      const skip = saved.length > 1 && this.inputs?.[0]?.widget ? 1 : 0;
      requestAnimationFrame(() => render(this, saved.slice(skip)));
    };
  },
});
