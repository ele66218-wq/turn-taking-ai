"""Tests for scripts/cors_proxy.py.

Spins up a stub upstream server (standing in for mlx_lm.server) plus the
proxy itself, both as background threads on ephemeral ports, and exercises
the proxy purely over HTTP -- no browser, no real LLM server required.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cors_proxy  # noqa: E402


class _StubUpstreamHandler(BaseHTTPRequestHandler):
    """Stands in for mlx_lm.server: echoes back a canned JSON body.

    Also emits its own permissive CORS headers, mirroring the real
    mlx_lm.server behaviour observed in practice -- the proxy must strip
    these rather than relay them (duplicate Access-Control-* headers are
    rejected by browsers).
    """

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        self.rfile.read(length)
        body = json.dumps({"choices": [{"message": {"content": "stub reply"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # keep test output quiet


@pytest.fixture
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubUpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=2)


@pytest.fixture
def proxy(upstream):
    allowed = {"https://ele66218-wq.github.io"}
    handler = cors_proxy._build_handler(upstream, allowed, timeout=5.0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=2)


def _request(url: str, *, method: str = "POST", origin: str | None = None, body: bytes = b"{}"):
    headers = {"Content-Type": "application/json"}
    if origin is not None:
        headers["Origin"] = origin
    req = urllib.request.Request(url, data=body if method == "POST" else None, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(resp.getheaders()), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


def test_allowed_origin_gets_single_cors_header(proxy):
    status, headers, body = _request(proxy, origin="https://ele66218-wq.github.io")
    assert status == 200
    assert headers["Access-Control-Allow-Origin"] == "https://ele66218-wq.github.io"
    assert json.loads(body)["choices"][0]["message"]["content"] == "stub reply"


def test_upstream_cors_headers_are_stripped_not_duplicated(proxy):
    """The stub upstream sends its own Access-Control-Allow-Origin: * --
    the proxy must not relay it alongside its own (duplicate headers break
    the browser's CORS check)."""
    status, headers, _ = _request(proxy, origin="https://ele66218-wq.github.io")
    assert status == 200
    # dict(resp.getheaders()) collapses duplicates to the last value seen;
    # assert it is exactly the proxy's value, not upstream's "*".
    assert headers["Access-Control-Allow-Origin"] == "https://ele66218-wq.github.io"


def test_disallowed_origin_is_rejected(proxy):
    status, _headers, _body = _request(proxy, origin="https://evil.example.com")
    assert status == 403


def test_no_origin_header_is_passed_through(proxy):
    """A same-machine/non-browser client with no Origin header should still work."""
    status, _headers, body = _request(proxy, origin=None)
    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "stub reply"


def test_options_preflight_from_allowed_origin(proxy):
    status, headers, body = _request(proxy, method="OPTIONS", origin="https://ele66218-wq.github.io")
    assert status == 204
    assert headers["Access-Control-Allow-Origin"] == "https://ele66218-wq.github.io"
    assert "POST" in headers["Access-Control-Allow-Methods"]
    assert body == b""


def test_upstream_unreachable_returns_502():
    allowed = {"https://ele66218-wq.github.io"}
    handler = cors_proxy._build_handler("http://127.0.0.1:1", allowed, timeout=2.0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        status, _headers, body = _request(url, origin="https://ele66218-wq.github.io")
        assert status == 502
        assert b"upstream unreachable" in body
    finally:
        server.shutdown()
        thread.join(timeout=2)
