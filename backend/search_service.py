"""
Search Service - Web search for the desktop app through Iron Desktop Service.

Routes search requests through the local Iron Desktop Service (port 8001)
which is a lightweight, localhost-only API server separate from the
full IronGate web gateway (port 8000).

Iron Desktop Service must be running for search to work.
The desktop app does NOT need duckduckgo-search installed —
Iron Desktop Service handles the actual searching.
"""

import os
import json
from PySide6.QtCore import QThread, Signal

# Iron Desktop Service runs on port 8001 (localhost only)
DESKTOP_SERVICE_URL = "http://localhost:8001"


class SearchWorker(QThread):
    """Background worker for web search (non-blocking for the UI)."""
    results_ready = Signal(list, str)    # results[], query
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.query = ""
        self.max_results = 5
        self._service_url = DESKTOP_SERVICE_URL

    def setup(self, query, max_results=5, service_url=None):
        self.query = query
        self.max_results = max_results
        if service_url:
            self._service_url = service_url

    def run(self):
        try:
            results = self._search(self.query, self.max_results)
            self.results_ready.emit(results, self.query)
        except Exception as e:
            self.error.emit(str(e))

    def _search(self, query, max_results):
        """Search via Iron Desktop Service on port 8001."""
        import requests

        url = f"{self._service_url.rstrip('/')}/api/search"
        try:
            resp = requests.post(url, json={
                "query": query,
                "max_results": max_results
            }, timeout=15)

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    return results
                raise Exception("Search returned no results.")

            # Try to get error message from response
            try:
                err_data = resp.json()
                err_msg = err_data.get("error", f"Status {resp.status_code}")
            except Exception:
                err_msg = f"Status {resp.status_code}"

            raise Exception(f"Search failed: {err_msg}")

        except requests.ConnectionError:
            raise Exception(
                "Iron Desktop Service not running. "
                "Start it from Settings or run: python gateway/iron_desktop.py"
            )
        except requests.Timeout:
            raise Exception("Search timed out. Try again.")
        except Exception as e:
            if "Iron Desktop" in str(e) or "Search" in str(e):
                raise  # Re-raise our own errors
            raise Exception(f"Search error: {e}")


class SearchService:
    """
    Manages web search for the desktop app.

    Search requests go through Iron Desktop Service (port 8001)
    which has duckduckgo-search installed.

    Usage:
        service = SearchService()
        results = service.search("python tutorials")
    """

    def __init__(self):
        self._service_url = DESKTOP_SERVICE_URL
        self._worker = None

    def set_service_url(self, url):
        """Override the desktop service URL (for testing)."""
        self._service_url = url

    def is_available(self):
        """Quick check if the desktop service is running."""
        import requests
        try:
            resp = requests.get(f"{self._service_url}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def search(self, query, max_results=5):
        """Synchronous search (blocks until results are ready)."""
        worker = SearchWorker()
        worker.setup(query, max_results, self._service_url)
        return worker._search(query, max_results)

    def search_async(self, query, on_results=None, on_error=None, max_results=5):
        """Asynchronous search (non-blocking, results delivered via callbacks)."""
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(1000)

        self._worker = SearchWorker()
        self._worker.setup(query, max_results, self._service_url)

        if on_results:
            self._worker.results_ready.connect(on_results)
        if on_error:
            self._worker.error.connect(on_error)

        self._worker.start()

    def format_for_ai(self, results, query):
        """Format search results as context string for the AI."""
        if not results:
            return f"[Web search for '{query}' returned no results.]"

        lines = [f"\n--- Web Search Results for: '{query}' ---\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['snippet']}")
            lines.append(f"   URL: {r['url']}\n")
        lines.append("---\nPlease use these search results to help answer the question. Cite sources where relevant.")

        return "\n".join(lines)
