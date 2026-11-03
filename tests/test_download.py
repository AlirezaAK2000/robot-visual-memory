"""HTTP contract checks without downloading the original archive."""

import hashlib

import pytest

from rvm.io import atomic_json
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


def test_resume_validated_ranges_and_md5(tmp_path, monkeypatch):
    path = tmp_path / "archive.tgz"
    part = tmp_path / "archive.tgz.part"
    part.write_bytes(b"abc")
    atomic_json(
        tmp_path / "archive.tgz.part.json", {"url": "https://official.test/file", "validator": '"version1"'}
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
    assert calls[0]["headers"] == {"Range": "bytes=3-", "If-Range": '"version1"'}


def test_server_ignoring_resume_restarts_and_login_page_stops(tmp_path, monkeypatch):
    path = tmp_path / "archive.tgz"
    (tmp_path / "archive.tgz.part").write_bytes(b"old")
    atomic_json(tmp_path / "archive.tgz.part.json", {"url": "https://test/file", "validator": "old"})
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
