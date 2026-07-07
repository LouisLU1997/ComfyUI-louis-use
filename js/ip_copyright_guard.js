/**
 * Louis_use — 版权筛选器（IP Copyright Guard）
 *
 * 监听「版权筛选器」节点的执行结果：
 *   · 干净 → 什么都不做，工作流正常结束（图片已保存）
 *   · 命中 IP → 自动调用 app.queuePrompt() 重新排队（换种子重生成），
 *               直到干净或达到「最大重试」次数。
 *
 * 依赖：生成种子的 control_after_generate 设为 randomize，
 *       这样每次重排队都会换种子，否则会一直生成同一张图。
 */
import { app } from "../../scripts/app.js";

const NODE_TYPE = "Louis_use_IPCopyrightGuard";

// 按节点 id 记录已重试次数（一次"干净"或新的手动运行后清零）
const retryCount = {};

app.registerExtension({
    name: "Louis_use.IPCopyrightGuard",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            origExecuted?.apply(this, arguments);

            const raw = message?.ipguard?.[0];
            if (!raw) return;

            let data;
            try { data = JSON.parse(raw); } catch (e) { return; }

            const nid = this.id;

            if (!data.found) {
                // 干净：清零计数
                retryCount[nid] = 0;
                console.log(`[IPGuard] 节点#${nid} ${data.report}`);
                return;
            }

            // 命中 IP
            const used = retryCount[nid] || 0;
            const max  = Number(data.max_retry) || 0;
            console.warn(`[IPGuard] 节点#${nid} ${data.report}（已重试 ${used}/${max}）`);

            if (used >= max) {
                retryCount[nid] = 0;
                console.warn(`[IPGuard] 节点#${nid} 达到最大重试 ${max}，停止。`);
                alert(`版权筛选器：重试 ${max} 次仍检测到版权 IP（${(data.names || []).join(", ")}），已停止。请调整提示词。`);
                return;
            }

            retryCount[nid] = used + 1;
            console.log(`[IPGuard] 节点#${nid} 换种子重新生成（第 ${retryCount[nid]} 次）…`);
            // 重新排队：control_after_generate=randomize 时会自动换种子
            app.queuePrompt(0, 1);
        };
    },
});
