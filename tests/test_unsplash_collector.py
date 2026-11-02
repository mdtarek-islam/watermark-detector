import json

import pytest

from watermark_detector.unsplash_collector import UnsplashCollector


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b"", headers=None):
        self.status_code = status
        self._payload = payload
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _photo(i):
    return {
        "id": f"p{i}",
        "urls": {"small": f"https://img.example/p{i}.jpg"},
        "links": {"download_location": f"https://api.unsplash.com/photos/p{i}/download"},
    }


class FakeSession:
    def __init__(self, pages, rate_limit_first=False):
        self.pages = pages
        self.calls = []
        self.rate_limit_first = rate_limit_first

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        if "search/photos" in url:
            if self.rate_limit_first:
                self.rate_limit_first = False
                return FakeResponse(429, headers={"Retry-After": "2"})
            page = params["page"]
            results = self.pages[page - 1] if page <= len(self.pages) else []
            return FakeResponse(payload={"total_pages": len(self.pages), "results": results})
        if url.endswith("/download"):
            return FakeResponse(payload={"url": "x"})
        return FakeResponse(content=b"\xff\xd8fakejpeg")


def _collector(tmp_path, session, sleeps=None):
    return UnsplashCollector(
        "KEY",
        tmp_path / "raw",
        session=session,
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


def test_requires_access_key(tmp_path):
    with pytest.raises(RuntimeError):
        UnsplashCollector("", tmp_path)


def test_paginates_until_target(tmp_path):
    session = FakeSession([[_photo(i) for i in range(3)], [_photo(i) for i in range(3, 6)]])
    paths = _collector(tmp_path, session).collect(5, query="nature", per_page=3)
    assert len(paths) == 5
    pages = [c[1]["page"] for c in session.calls if "search/photos" in c[0]]
    assert pages == [1, 2]
    assert session.calls[0][2]["Authorization"] == "Client-ID KEY"


def test_stops_when_pages_run_out(tmp_path):
    session = FakeSession([[_photo(0), _photo(1)]])
    assert len(_collector(tmp_path, session).collect(10, per_page=2)) == 2


def test_manifest_makes_download_resumable(tmp_path):
    pages = [[_photo(i) for i in range(4)]]
    _collector(tmp_path, FakeSession(pages)).collect(4, per_page=4)
    manifest = json.loads((tmp_path / "raw" / "manifest.json").read_text())
    assert set(manifest) == {"p0", "p1", "p2", "p3"}

    session = FakeSession(pages)
    paths = _collector(tmp_path, session).collect(4, per_page=4)
    assert len(paths) == 4
    assert not session.calls


def test_rate_limit_waits_and_retries(tmp_path):
    sleeps = []
    session = FakeSession([[_photo(0)]], rate_limit_first=True)
    paths = _collector(tmp_path, session, sleeps).collect(1, per_page=1)
    assert len(paths) == 1
    assert sleeps and sleeps[0] == 2.0
