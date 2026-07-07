/**
 * Louis_use — ResolutionSelector
 *
 * 当 aspect_ratio 输入槽被连线时，自动隐藏「宽高比」下拉框（连线优先）；
 * 断线后恢复显示。
 */
import { app } from "../../scripts/app.js";

const NODE_TYPE = "Louis_use_ResolutionSelector";
const WIDGET_NAME = "宽高比";

function setHidden(widget, hidden) {
    if (!widget) return;
    if (hidden) {
        widget.__origType        = widget.__origType        ?? widget.type;
        widget.__origComputeSize = widget.__origComputeSize ?? widget.computeSize;
        widget.type        = "hidden";
        widget.computeSize = () => [0, -4];
    } else {
        if (widget.__origType != null) {
            widget.type = widget.__origType;
            delete widget.__origType;
        }
        if (widget.__origComputeSize != null) {
            widget.computeSize = widget.__origComputeSize;
            delete widget.__origComputeSize;
        }
    }
}

function syncWidgetVisibility(node) {
    const slot = node.inputs?.find(i => i.name === "aspect_ratio");
    const w    = node.widgets?.find(w => w.name === WIDGET_NAME);
    const connected = slot?.link != null;
    setHidden(w, connected);
    requestAnimationFrame(() => {
        node.setSize(node.computeSize());
        app.graph.setDirtyCanvas(true, false);
    });
}

app.registerExtension({
    name: "Louis_use.ResolutionSelector",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        // 连线 / 断线时触发
        const origConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info, inputInfo) {
            origConnectionsChange?.apply(this, arguments);
            if (type === 1 /* INPUT */ && inputInfo?.name === "aspect_ratio") {
                syncWidgetVisibility(this);
            }
        };

        // 节点刚创建时初始化（新建节点）
        const origNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origNodeCreated?.apply(this, arguments);
            requestAnimationFrame(() => syncWidgetVisibility(this));
        };

        // 工作流加载后恢复状态
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            origOnConfigure?.apply(this, arguments);
            requestAnimationFrame(() => syncWidgetVisibility(this));
        };
    },
});
