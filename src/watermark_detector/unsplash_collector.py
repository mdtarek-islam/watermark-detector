"""Collect clean photographs from the Unsplash search API.

The collector pages through search results, waits when the API reports a
rate limit, and records each saved file in a JSON manifest. A second run
reads the manifest and skips photographs it already holds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import requests

API_ROOT = "https://api.unsplash.com"
DEFAULT_WAIT_S = 60.0
MAX_RETRIES = 5


class UnsplashCollector:
    """Download search results to out_dir and track them in manifest.json."""

    def __init__(
        self,
        access_key: str,
        out_dir: str | Path,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        image_variant: str = "small",
        timeout: float = 30.0,
        track_downloads: bool = True,
    ) -> None:
        if not access_key:
            raise RuntimeError("An Unsplash access key is required. Set UNSPLASH_ACCESS_KEY.")
        self.out_dir = Path(out_dir)
        self.session = session or requests.Session()
        self.sleep = sleep
        self.variant = image_variant
        self.timeout = timeout
        self.track_downloads = track_downloads
        self.headers = {"Authorization": f"Client-ID {access_key}", "Accept-Version": "v1"}
        self.manifest_path = self.out_dir / "manifest.json"

    def _load_manifest(self) -> dict[str, str]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for attempt in range(10):
            try:
                tmp.replace(self.manifest_path)
                return
            except PermissionError:
                # Windows can hold the target briefly (scanners, indexers). Retry.
                if attempt == 9:
                    self.manifest_path.write_text(tmp.read_text(encoding="utf-8"), encoding="utf-8")
                    return
                time.sleep(0.2)

    def _get(self, url: str, params: dict | None = None, auth: bool = True):
        """GET with retries on HTTP 429, or on HTTP 403 with no requests left."""
        for _ in range(MAX_RETRIES):
            resp = self.session.get(
                url, params=params, headers=self.headers if auth else None, timeout=self.timeout
            )
            limited = resp.status_code == 429 or (
                resp.status_code == 403 and resp.headers.get("X-Ratelimit-Remaining") == "0"
            )
            if not limited:
                resp.raise_for_status()
                return resp
            self.sleep(float(resp.headers.get("Retry-After", DEFAULT_WAIT_S)))
        raise RuntimeError(f"Rate limit persisted after {MAX_RETRIES} retries: {url}")

    def collect(self, target: int, query: str = "landscape", per_page: int = 30) -> list[Path]:
        """Download photographs until target files exist or the results run out."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        manifest = {k: v for k, v in manifest.items() if (self.out_dir / v).exists()}

        page = 1
        while len(manifest) < target:
            resp = self._get(
                f"{API_ROOT}/search/photos",
                params={"query": query, "page": page, "per_page": per_page, "content_filter": "high"},
            )
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for photo in results:
                if len(manifest) >= target:
                    break
                pid = photo["id"]
                if pid in manifest:
                    continue
                download_location = photo.get("links", {}).get("download_location")
                if download_location and self.track_downloads:
                    # Unsplash API guidelines ask clients to report each download.
                    # Each report counts against the hourly quota (50 for demo keys).
                    self._get(download_location)
                image = self._get(photo["urls"][self.variant], auth=False)
                name = f"{pid}.jpg"
                (self.out_dir / name).write_bytes(image.content)
                manifest[pid] = name
                self._save_manifest(manifest)
            if page >= int(data.get("total_pages", page)):
                break
            page += 1

        self._save_manifest(manifest)
        return [self.out_dir / v for v in list(manifest.values())[:target]]
