#!/usr/bin/env python3
"""
Iron Desktop Service v1.0
Lightweight local-only API server for the VoxAI Desktop App.

Runs on port 8001 — NO tunnel, NO web UI, NO admin console.
Provides:
  - /api/search      (web search via DuckDuckGo)
  - /api/search/fetch (fetch page content for AI context)
  - /api/upload       (file upload for AI context)
  - /health           (status check)

This is separate from iron_host.py (port 8000) which handles
the full web UI with ngrok tunnel and authentication.
"""

import os
import sys
import time
import signal
import logging

# Suppress noisy loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from typing import List

# --- Resolve search engine at module level ---
DDGS = None
_search_engine_name = "none"

try:
    from ddgs import DDGS as _DDGS
    DDGS = _DDGS
    _search_engine_name = "ddgs"
except ImportError:
    try:
        from duckduckgo_search import DDGS as _DDGS
        DDGS = _DDGS
        _search_engine_name = "duckduckgo_search"
    except ImportError:
        pass  # DDGS stays None

# =============================================================================
# APP SETUP
# =============================================================================

PORT = 8001

app = FastAPI(
    title="IronGate Desktop Service",
    docs_url=None,
    redoc_url=None
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_GUI_ROOT = os.path.dirname(BASE_DIR)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads_desktop")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =============================================================================
# SECURITY: Only allow localhost
# =============================================================================

def _get_client_ip(request: Request):
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


@app.middleware("http")
async def localhost_only(request: Request, call_next):
    """Reject all non-localhost connections."""
    ip = _get_client_ip(request)
    if ip not in ("127.0.0.1", "::1", "localhost"):
        return JSONResponse({"error": "Access denied"}, status_code=403)
    return await call_next(request)


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "service": "iron_desktop",
        "port": PORT,
        "search_engine": _search_engine_name
    })


# =============================================================================
# WEB SEARCH API
# =============================================================================

SEARCH_CONFIG = {
    "enabled": True,
    "max_results": 5,
    "safe_search": "moderate"
}


@app.post("/api/search")
async def web_search(request: Request):
    """Web search using DuckDuckGo. Desktop app calls this for AI web search."""
    if not SEARCH_CONFIG["enabled"]:
        return JSONResponse({"error": "Search disabled"}, 403)

    if DDGS is None:
        return JSONResponse({
            "error": "Search package not installed. Run: pip install ddgs"
        }, 503)

    try:
        data = await request.json()
        query = data.get("query", "").strip()
        max_results = min(int(data.get("max_results", 5)), 10)

        if not query:
            return JSONResponse({"error": "No query"}, 400)

        try:
            ddgs = DDGS()
            raw = ddgs.text(query, max_results=max_results, safesearch=SEARCH_CONFIG["safe_search"])

            results = []
            for r in raw:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })

            print(f"[SEARCH] '{query}' -> {len(results)} results")
            return JSONResponse({
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results)
            })

        except Exception as e:
            return JSONResponse({"error": f"Search failed: {e}"}, 500)

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


@app.post("/api/search/fetch")
async def fetch_page(request: Request):
    """Fetch a URL's text content for AI context enrichment."""
    try:
        data = await request.json()
        url = data.get("url", "").strip()
        if not url:
            return JSONResponse({"error": "No URL"}, 400)

        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return JSONResponse({"error": "Only HTTP/HTTPS URLs allowed"}, 400)

        import requests as req_lib
        resp = req_lib.get(url, timeout=10, headers={
            'User-Agent': 'VoxAI-IronGate/1.0 (Research Assistant)'
        })
        resp.raise_for_status()

        # Strip HTML to plain text
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ('script', 'style', 'nav', 'footer', 'header'):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ('script', 'style', 'nav', 'footer', 'header'):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    text = data.strip()
                    if text:
                        self.text.append(text)

        extractor = TextExtractor()
        extractor.feed(resp.text)
        page_text = '\n'.join(extractor.text)

        if len(page_text) > 4000:
            page_text = page_text[:4000] + '\n... [truncated]'

        return JSONResponse({
            "status": "success",
            "url": url,
            "title": parsed.netloc,
            "content": page_text
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)


# =============================================================================
# FILE UPLOAD API
# =============================================================================

TEXT_EXTENSIONS = {
    '.txt', '.py', '.js', '.json', '.csv', '.md', '.html', '.css',
    '.xml', '.yaml', '.yml', '.log', '.ts', '.jsx', '.tsx', '.sh',
    '.bat', '.cfg', '.ini', '.toml', '.sql', '.java', '.cpp', '.c',
    '.h', '.rs', '.go', '.rb', '.php'
}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle file uploads from the desktop app."""
    results = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        content = await file.read()

        # Save file
        safe_name = f"{int(time.time())}_{file.filename}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)
        with open(save_path, 'wb') as f:
            f.write(content)

        if ext in TEXT_EXTENSIONS:
            try:
                text_content = content.decode('utf-8', errors='replace')
                if len(text_content) > 6000:
                    text_content = text_content[:6000] + '\n... [truncated]'
                results.append({
                    'filename': file.filename,
                    'type': 'text',
                    'content': text_content,
                    'size': len(content)
                })
            except Exception:
                results.append({
                    'filename': file.filename,
                    'type': 'binary',
                    'content': '',
                    'size': len(content)
                })
        elif ext in IMAGE_EXTENSIONS:
            results.append({
                'filename': file.filename,
                'type': 'image',
                'description': f'Image file ({ext.upper()}, {len(content)} bytes)',
                'url': f'/api/upload/{safe_name}',
                'size': len(content)
            })
        else:
            results.append({
                'filename': file.filename,
                'type': 'binary',
                'content': '',
                'size': len(content)
            })

    return JSONResponse({'files': results})


@app.get("/api/upload/{filename}")
async def get_upload(filename: str):
    """Serve uploaded files."""
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "File not found"}, 404)


# =============================================================================
# STARTUP
# =============================================================================

def _free_port(port):
    """Kill any process using the specified port (Windows).

    Tries multiple methods: PowerShell Stop-Process, then taskkill.
    """
    if os.name != 'nt':
        return

    import subprocess

    # Method 1: PowerShell Get-NetTCPConnection + Stop-Process
    try:
        # Get the PID first
        get_pid = subprocess.run(
            ['powershell.exe', '-NoProfile', '-Command',
             f'(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue).OwningProcess'],
            capture_output=True, text=True, timeout=5
        )
        pid_str = get_pid.stdout.strip()
        if not pid_str:
            return  # Port is free

        for pid in pid_str.split('\n'):
            pid = pid.strip()
            if not pid or not pid.isdigit():
                continue
            pid_int = int(pid)
            if pid_int == os.getpid():
                continue  # Don't kill ourselves

            print(f"[CLEANUP] Killing stale process PID {pid_int} on port {port}...")

            # Try PowerShell first
            result = subprocess.run(
                ['powershell.exe', '-NoProfile', '-Command',
                 f'Stop-Process -Id {pid_int} -Force -ErrorAction SilentlyContinue'],
                capture_output=True, text=True, timeout=5
            )

            # If that fails (access denied), try taskkill
            if result.returncode != 0:
                subprocess.run(
                    ['taskkill', '/F', '/PID', str(pid_int)],
                    capture_output=True, text=True, timeout=5
                )

        # Wait and verify
        time.sleep(1)
        check = subprocess.run(
            ['powershell.exe', '-NoProfile', '-Command',
             f'Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue'],
            capture_output=True, text=True, timeout=5
        )
        if not check.stdout.strip():
            print(f"[CLEANUP] Port {port} freed successfully")
        else:
            print(f"[CLEANUP] WARNING: Port {port} still in use (may need admin rights)")

    except Exception as e:
        print(f"[CLEANUP] Could not free port {port}: {e}")


def force_shutdown(signum, frame):
    print("\nIron Desktop Service stopped.")
    os._exit(0)


def main():
    import uvicorn

    signal.signal(signal.SIGINT, force_shutdown)
    signal.signal(signal.SIGTERM, force_shutdown)

    print(f"=== Iron Desktop Service v1.0 ===")
    print(f"Local-only API server for VoxAI Desktop")
    print(f"Port: {PORT} | Access: localhost only")

    # Report search engine status
    if DDGS is not None:
        print(f"Search engine: {_search_engine_name} (OK)")
    else:
        print(f"WARNING: No search package found! Run: pip install ddgs")

    # Free port if something stale is on it
    _free_port(PORT)
    time.sleep(0.3)

    print(f"Starting on http://localhost:{PORT} ...")
    print(f"Endpoints: /api/search, /api/search/fetch, /api/upload, /health")

    try:
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
    except OSError as e:
        if "10048" in str(e) or "address already in use" in str(e).lower():
            print(f"ERROR: Port {PORT} still in use. Close the VoxAI Desktop app and try again.")
        else:
            print(f"Server Error: {e}")
    except Exception as e:
        print(f"Server Error: {e}")
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
