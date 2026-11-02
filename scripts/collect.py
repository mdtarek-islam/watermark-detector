"""Collect Unsplash photos for the reproduction run without download pings.

Search requests only (per_page=30), so 450 photos need about 15 API calls.
Stops with a message if the quota is exhausted. Resumable through manifest.json.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from watermark_detector.config import Config  # noqa: E402
from watermark_detector.unsplash_collector import UnsplashCollector  # noqa: E402


def no_wait(seconds: float) -> None:
    raise SystemExit(f"Unsplash quota exhausted (retry after {seconds:.0f} s). Collection stopped.")


def main() -> None:
    cfg = Config.from_env()
    collector = UnsplashCollector(
        cfg.require_access_key(), cfg.data_dir / "raw", sleep=no_wait, track_downloads=True
    )
    paths = collector.collect(cfg.dataset_size, query=cfg.query, per_page=30)
    print(f"collected {len(paths)} photos")


if __name__ == "__main__":
    main()
