print("[Louis_use] 开始加载...")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    from . import api
    print("[Louis_use] API 路由注册成功 ✓")
except Exception as e:
    print(f"[Louis_use] API 注册失败: {e}")
    import traceback
    traceback.print_exc()

WEB_DIRECTORY = "./js"

print(f"[Louis_use] 加载完成，共 {len(NODE_CLASS_MAPPINGS)} 个节点: {list(NODE_CLASS_MAPPINGS.keys())}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
