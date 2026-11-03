"""HTTP contract checks without downloading the original archive."""

import hashlib
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from rvm.io import atomic_json, read_json
from rvm.preparation import download


class Response:
    def __init__(self, status, data, headers):
        self.status_code, self.data, self.headers = status, data, headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.data


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("rvm.preparation.time.sleep", lambda _: None)


def test_resume_validated_ranges_and_md5(tmp_path, monkeypatch):
    path = tmp_path / "archive.tgz"
    part = tmp_path / "archive.tgz.part"
    part.write_bytes(b"abc")
    atomic_json(
        tmp_path / "archive.tgz.part.json",
        {"url": "https://official.test/file", "validator": '"version1"', "content_encoding": "identity"},
    )
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs)
        return Response(
            206, b"def", {"Content-Range": "bytes 3-5/6", "Content-Length": "3", "ETag": '"version1"'}
        )

    monkeypatch.setattr("rvm.preparation.requests.get", get)
    assert (
        download(
            "https://official.test/file", path, expected_md5=hashlib.md5(b"abcdef").hexdigest()
        ).read_bytes()
        == b"abcdef"
    )
    assert calls[0]["headers"] == {
        "Accept-Encoding": "identity",
        "Range": "bytes=3-",
        "If-Range": '"version1"',
    }


def test_server_ignoring_resume_restarts_and_login_page_stops(tmp_path, monkeypatch):
    path = tmp_path / "archive.tgz"
    (tmp_path / "archive.tgz.part").write_bytes(b"old")
    atomic_json(
        tmp_path / "archive.tgz.part.json",
        {"url": "https://test/file", "validator": "old", "content_encoding": "identity"},
    )
    monkeypatch.setattr(
        "rvm.preparation.requests.get", lambda *a, **kw: Response(200, b"new", {"Content-Length": "3"})
    )
    assert download("https://test/file", path).read_bytes() == b"new"
    path.unlink()
    monkeypatch.setattr(
        "rvm.preparation.requests.get",
        lambda *a, **kw: Response(200, b"login", {"Content-Type": "text/html"}),
    )
    with pytest.raises(RuntimeError, match="Add Input"):
        download("https://test/file", path)


def test_real_gzip_http_body_uses_wire_length_not_decoded_length(tmp_path):
    # Actual requests/urllib3 decompression, not a fake iter_content implementation.
    body = json.dumps({"synthetic": "metadata only " * 100}).encode()
    compressed = gzip.compress(body)
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")  # deliberately ignores identity request
            self.send_header("Content-Length", str(len(compressed)))
            self.end_headers()
            self.wfile.write(compressed)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        path = download(f"http://127.0.0.1:{server.server_port}/metadata", tmp_path / "metadata.json")
        assert path.read_bytes() == body and read_json(path)["synthetic"]
        assert len(seen) == 1 and seen[0]["Accept-Encoding"] == "identity"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_expired_link_explains_mirror_route_without_saving_html(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "rvm.preparation.requests.get",
        lambda *a, **kw: Response(
            200,
            b"<html><title>Dropbox - Link Expired</title></html>",
            {"Content-Type": "text/html"},
        ),
    )
    target = tmp_path / "archive.tgz"
    with pytest.raises(RuntimeError, match="RGB_PROVIDER='huggingface'"):
        download("https://test/file", target)
    assert not target.exists() and not target.with_suffix(".tgz.part").exists()


@pytest.mark.parametrize("old_encoding", [None, "gzip"])
def test_legacy_or_decoded_partial_restarts_safely(tmp_path, monkeypatch, old_encoding):
    path = tmp_path / "metadata.json"
    (tmp_path / "metadata.json.part").write_bytes(b"old decoded body")
    atomic_json(
        tmp_path / "metadata.json.part.json",
        {
            "url": "https://test/file",
            "validator": '"v1"',
            "content_encoding": old_encoding,
        },
    )
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs["headers"])
        return Response(200, b"new", {"Content-Length": "3", "ETag": '"v1"'})

    monkeypatch.setattr("rvm.preparation.requests.get", get)
    assert download("https://test/file", path).read_bytes() == b"new"
    assert len(calls) == 1 and "Range" not in calls[0]


def test_truncation_automatically_resumes_identity_bytes(tmp_path, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs["headers"])
        if len(calls) == 1:
            return Response(200, b"abc", {"Content-Length": "6", "ETag": '"v1"'})
        return Response(206, b"def", {"Content-Length": "3", "Content-Range": "bytes 3-5/6", "ETag": '"v1"'})

    monkeypatch.setattr("rvm.preparation.requests.get", get)
    path = download("https://test/file", tmp_path / "file", expected_md5=hashlib.md5(b"abcdef").hexdigest())
    assert path.read_bytes() == b"abcdef"
    assert "Range" not in calls[0] and calls[1]["Range"] == "bytes=3-"


@pytest.mark.parametrize("fault", ["range", "encoding", "validator", "length", "416"])
def test_unsafe_resume_response_retries_without_appending(tmp_path, monkeypatch, fault):
    path = tmp_path / "file"
    (tmp_path / "file.part").write_bytes(b"abc")
    atomic_json(
        tmp_path / "file.part.json",
        {"url": "https://test/file", "validator": '"v1"', "content_encoding": "identity"},
    )
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs["headers"])
        if len(calls) > 1:
            return Response(200, b"abcdef", {"Content-Length": "6", "ETag": '"v2"'})
        headers = {"Content-Length": "3", "Content-Range": "bytes 3-5/6", "ETag": '"v1"'}
        if fault == "range":
            headers["Content-Range"] = "bytes 2-4/6"
        elif fault == "encoding":
            headers["Content-Encoding"] = "gzip"
        elif fault == "validator":
            headers["ETag"] = '"v2"'
        elif fault == "length":
            headers["Content-Length"] = "2"
        return Response(416 if fault == "416" else 206, b"BAD", headers)

    monkeypatch.setattr("rvm.preparation.requests.get", get)
    assert download("https://test/file", path).read_bytes() == b"abcdef"
    assert calls[0]["Range"] == "bytes=3-" and "Range" not in calls[1]


def test_exhausted_retries_retain_resumable_partial_for_next_run(tmp_path, monkeypatch):
    path = tmp_path / "file"
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return Response(200, b"abc", {"Content-Length": "6", "ETag": '"v1"'})
        return Response(206, b"", {"Content-Length": "3", "Content-Range": "bytes 3-5/6", "ETag": '"v1"'})

    monkeypatch.setattr("rvm.preparation.requests.get", get)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        download("https://test/file", path)
    assert len(calls) == 3 and not path.exists()
    assert (tmp_path / "file.part").read_bytes() == b"abc"
    monkeypatch.setattr(
        "rvm.preparation.requests.get",
        lambda *a, **kw: Response(
            206,
            b"def",
            {
                "Content-Length": "3",
                "Content-Range": "bytes 3-5/6",
                "ETag": '"v1"',
            },
        ),
    )
    assert download("https://test/file", path).read_bytes() == b"abcdef"
