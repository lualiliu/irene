"""本地预览 docs/，并提供 /mirror/ 代理到 hf-mirror.net（绕过浏览器 CORS）。

用法: python scripts/serve.py [端口]
"""
import sys
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
MIRROR_UPSTREAM = "https://hf-mirror.net"
# 浏览器直连 hf-mirror.net 会因 CORS（只放行 huggingface.co）失败；同域代理可绕过。
PROXY_PREFIX = "/mirror/"


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Robots-Tag", "noindex, nofollow, noarchive, nosnippet")
        super().end_headers()

    def do_OPTIONS(self):
        if self.path.startswith(PROXY_PREFIX):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", "*"))
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", self.headers.get("Access-Control-Request-Headers", "*"))
            self.send_header("Access-Control-Max-Age", "86400")
            self.end_headers()
            return
        self.send_error(404)

    def do_HEAD(self):
        if self.path.startswith(PROXY_PREFIX):
            self._proxy(head_only=True)
            return
        super().do_HEAD()

    def do_GET(self):
        if self.path.startswith(PROXY_PREFIX):
            self._proxy(head_only=False)
            return
        super().do_GET()

    def _proxy(self, head_only: bool):
        rel = self.path[len(PROXY_PREFIX) :].lstrip("/")
        if not rel or ".." in rel:
            self.send_error(400, "bad mirror path")
            return
        url = f"{MIRROR_UPSTREAM}/{rel}"
        headers = {"User-Agent": self.headers.get("User-Agent", "guoruolin-chat-mirror-proxy")}
        # WebLLM 会用 Range 分片下载大文件
        if rng := self.headers.get("Range"):
            headers["Range"] = rng
        req = urllib.request.Request(url, headers=headers, method="HEAD" if head_only else "GET")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = resp.status
                self.send_response(status)
                skip = {"transfer-encoding", "connection", "content-encoding"}
                for k, v in resp.headers.items():
                    if k.lower() not in skip:
                        self.send_header(k, v)
                origin = self.headers.get("Origin")
                if origin:
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                else:
                    self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header(
                    "Access-Control-Expose-Headers",
                    "Accept-Ranges, Content-Range, Content-Length, ETag, X-Repo-Commit",
                )
                self.end_headers()
                if not head_only:
                    while chunk := resp.read(1024 * 256):
                        self.wfile.write(chunk)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            if not head_only and e.fp:
                self.wfile.write(e.read())
        except Exception as e:
            self.send_error(502, f"mirror proxy failed: {e}")

    def log_message(self, fmt, *args):
        # 代理大文件时刷屏太多，只打非 200 或非分片请求
        if args and str(args[0]).startswith(PROXY_PREFIX) and "200" in str(args[1:2]):
            return
        super().log_message(fmt, *args)


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
print(f"http://localhost:{port}/")
print(f"mirror proxy: http://localhost:{port}{PROXY_PREFIX} → {MIRROR_UPSTREAM}/")
ThreadingHTTPServer(("", port), partial(Handler, directory=str(DOCS))).serve_forever()
