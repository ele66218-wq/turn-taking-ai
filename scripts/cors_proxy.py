#!/usr/bin/env python3
"""Minimal CORS-adding reverse proxy in front of a local LLM server.

Why this exists
----------------
`mlx_lm.server` (the OpenAI-compatible local server that ships with
mlx-lm) is implemented on top of Python's stdlib ``http.server`` and does
not send any ``Access-Control-Allow-Origin`` header. A browser page served
from GitHub Pages (a different origin) therefore cannot call it directly --
the browser blocks the response before JavaScript ever sees it.

This script sits between the browser and ``mlx_lm.server``: it forwards
every request byte-for-byte to the upstream server and adds the CORS
headers the browser requires, allowlisting only the configured origin(s)
(never ``*``, since this proxy is meant to be exposed to the internet via
``tailscale serve``).

Usage
-----
    python scripts/cors_proxy.py --upstream http://127.0.0.1:8080 --port 8787 \\
        --allow-origin https://ele66218-wq.github.io

Then point the browser demo's LLM endpoint setting at
``https://<your-tailscale-hostname>`` (after fronting this proxy with
``tailscale serve``), not directly at the upstream server.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Request headers we forward upstream; anything else (e.g. Host) is dropped
#: so the proxy always talks to the upstream on its own terms.
_FORWARDED_REQUEST_HEADERS = {"content-type", "authorization", "accept"}


def _build_handler(upstream: str, allowed_origins: set[str], timeout: float) -> type[BaseHTTPRequestHandler]:
    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _cors_origin(self) -> str | None:
            origin = self.headers.get("Origin")
            if origin is not None and origin in allowed_origins:
                return origin
            return None

        def _send_cors_headers(self, origin: str | None) -> None:
            if origin is not None:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.send_header("Access-Control-Max-Age", "600")

        def do_OPTIONS(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler naming convention)
            origin = self._cors_origin()
            self.send_response(204)
            self._send_cors_headers(origin)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            self._proxy("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._proxy("POST")

        def _proxy(self, method: str) -> None:
            origin = self._cors_origin()
            if self.headers.get("Origin") is not None and origin is None:
                # A cross-origin request from a non-allowlisted origin: refuse outright.
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(content_length) if content_length > 0 else None

            forward_headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() in _FORWARDED_REQUEST_HEADERS
            }
            request = urllib.request.Request(
                url=f"{upstream}{self.path}",
                data=body,
                headers=forward_headers,
                method=method,
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    self._relay_response(response.status, dict(response.getheaders()), response.read(), origin)
            except urllib.error.HTTPError as exc:
                self._relay_response(exc.code, dict(exc.headers or {}), exc.read(), origin)
            except urllib.error.URLError as exc:
                message = f"upstream unreachable at {upstream}: {exc.reason}".encode()
                self.send_response(502)
                self._send_cors_headers(origin)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)

        # Headers we never relay from upstream as-is: hop-by-hop headers
        # (content-length/transfer-encoding/connection), headers
        # send_response() already emits itself (server/date), and any CORS
        # header upstream might set on its own (mlx_lm.server does send a
        # wildcard Access-Control-Allow-* set) -- relaying those alongside
        # ours would produce duplicate headers, which browsers reject the
        # response for. This proxy is the single source of truth for CORS.
        _SKIP_RELAY_HEADERS = {
            "content-length",
            "transfer-encoding",
            "connection",
            "server",
            "date",
            "access-control-allow-origin",
            "access-control-allow-methods",
            "access-control-allow-headers",
            "access-control-max-age",
            "vary",
        }

        def _relay_response(
            self, status: int, headers: dict[str, str], body: bytes, origin: str | None
        ) -> None:
            self.send_response(status)
            self._send_cors_headers(origin)
            for key, value in headers.items():
                if key.lower() in self._SKIP_RELAY_HEADERS:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            sys.stderr.write(f"[cors_proxy] {self.address_string()} - {format % args}\n")

    return ProxyHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8080", help="mlx_lm.server base URL")
    parser.add_argument("--port", type=int, default=8787, help="Port this proxy listens on")
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=["https://ele66218-wq.github.io"],
        help="Allowed browser origin (repeatable). Default: the turn-taking-ai GitHub Pages origin.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Upstream request timeout in seconds")
    args = parser.parse_args(argv)

    upstream = args.upstream.rstrip("/")
    allowed_origins = set(args.allow_origin)

    handler = _build_handler(upstream, allowed_origins, args.timeout)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(f"[cors_proxy] listening on http://0.0.0.0:{args.port} -> {upstream}")
    print(f"[cors_proxy] allowed origins: {sorted(allowed_origins)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[cors_proxy] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
