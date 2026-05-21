"""
ComfyUI Folder I/O — 后端 API
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from aiohttp import web

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


# ── 系统原生文件夹对话框 ──────────────────────────────────────────────────────

def _pick_windows(title, initial_dir):
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
        f"$d.Description='{title}';"
        "$d.ShowNewFolderButton=$true;"
    )
    if initial_dir and Path(initial_dir).exists():
        ps += f"$d.SelectedPath='{initial_dir}';"
    ps += "if($d.ShowDialog() -eq 'OK'){Write-Output $d.SelectedPath}"
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, timeout=120
        )
        raw = r.stdout or b""
        for enc in ("gbk", "gb2312", "utf-8-sig", "utf-8"):
            try:
                path = raw.decode(enc).strip()
                if path:
                    return path
            except (UnicodeDecodeError, ValueError):
                continue
        return None
    except Exception as e:
        print(f"[Louis_use] PowerShell 对话框失败: {e}")
        return None


def _pick_macos(title, initial_dir):
    script = f'POSIX path of (choose folder with prompt "{title}"'
    if initial_dir and Path(initial_dir).exists():
        script += f' default location POSIX file "{initial_dir}"'
    script += ")"
    r = subprocess.run(["osascript", "-e", script],
                       capture_output=True, text=True, timeout=120)
    path = r.stdout.strip().rstrip("/")
    return path if path and r.returncode == 0 else None


def _pick_linux(title, initial_dir):
    for cmd in [
        ["zenity", "--file-selection", "--directory", f"--title={title}"],
        ["kdialog", "--getexistingdirectory", initial_dir or "/"],
        ["yad", "--file", "--directory", f"--title={title}"],
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            path = r.stdout.strip()
            if path and r.returncode == 0:
                return path
        except FileNotFoundError:
            continue
    return None


def _open_dialog_sync(title, initial_dir):
    if sys.platform == "win32":
        return _pick_windows(title, initial_dir)
    elif sys.platform == "darwin":
        return _pick_macos(title, initial_dir)
    else:
        return _pick_linux(title, initial_dir)


# ── 注册路由 ──────────────────────────────────────────────────────────────────
try:
    from server import PromptServer

    @PromptServer.instance.routes.post("/folder_io/open_dialog")
    async def open_folder_dialog(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        title       = body.get("title", "选择文件夹")
        initial_dir = body.get("initial_dir", "")
        loop     = asyncio.get_event_loop()
        selected = await loop.run_in_executor(None, _open_dialog_sync, title, initial_dir)
        return web.json_response({"path": selected, "error": None})

    @PromptServer.instance.routes.post("/folder_io/count")
    async def folder_count(request):
        try:
            body   = await request.json()
            folder = body.get("folder_path", "").strip()
            if not folder or not Path(folder).is_dir():
                return web.json_response({"count": 0, "error": "路径无效"})
            from .nodes import collect_images
            paths = collect_images(folder, "name")
            exts  = sorted({Path(f).suffix.lower() for f in paths})
            return web.json_response({"count": len(paths), "extensions": exts, "error": None})
        except Exception as e:
            return web.json_response({"count": 0, "error": str(e)})

    @PromptServer.instance.routes.get("/folder_io/preview")
    async def folder_preview(request):
        """GET /folder_io/preview?path=...&count=N  并发返回 base64 缩略图列表"""
        import base64, io, concurrent.futures
        try:
            folder = request.rel_url.query.get("path", "").strip()
            count  = int(request.rel_url.query.get("count", "999"))
            if not folder or not Path(folder).is_dir():
                return web.json_response({"thumbs": []})
            from .nodes import collect_images
            from PIL import Image

            paths = collect_images(folder, "name")
            # count=999 表示全部，否则截断
            if count < 999:
                paths = paths[:count]

            def _make_thumb(p):
                """在线程池里同步读图 → base64（避免阻塞事件循环）"""
                try:
                    img = Image.open(p)
                    # JPEG 用 draft 降采样，只解码到目标分辨率，速度快 3-5x
                    if getattr(img, "format", "") == "JPEG":
                        img.draft("RGB", (120, 90))
                    img = img.convert("RGB")
                    img.thumbnail((120, 90), Image.BILINEAR)   # BILINEAR 比 LANCZOS 快
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=70)
                    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    return None

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = await asyncio.gather(
                    *[loop.run_in_executor(pool, _make_thumb, p) for p in paths]
                )

            thumbs = [r for r in results if r is not None]
            return web.json_response({"thumbs": thumbs})
        except Exception as e:
            return web.json_response({"thumbs": [], "error": str(e)})

    print("[Louis_use] API 路由注册成功 ✓")

except Exception as e:
    print(f"[Louis_use] API 路由注册失败: {e}")
    import traceback
    traceback.print_exc()
