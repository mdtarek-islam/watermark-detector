"""Run configuration. Defaults match the original October 2024 training run."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

ENV_ACCESS_KEY = "UNSPLASH_ACCESS_KEY"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv_files() -> None:
    """Load the repo root dotenv file, then the workspace root one. Never override set variables."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for folder in (REPO_ROOT, REPO_ROOT.parent):
        load_dotenv(folder / (chr(46) + "env"), override=False)


@dataclass(frozen=True)
class Config:
    """Hold every tunable value for collection, training and inference."""

    dataset_size: int = 450
    epochs: int = 20
    batch_size: int = 32
    image_size: tuple[int, int] = (128, 128)
    seed: int = 42
    n_pairs: int = 2000
    query: str = "landscape"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    unsplash_access_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("dataset_size", "epochs", "batch_size", "n_pairs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        object.__setattr__(self, "image_size", tuple(self.image_size))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides) -> "Config":
        """Build a config and read the Unsplash access key from the environment."""
        if env is None:
            load_dotenv_files()
            env = os.environ
        key = env.get(ENV_ACCESS_KEY) or None
        overrides.setdefault("unsplash_access_key", key)
        return cls(**overrides)

    def require_access_key(self) -> str:
        """Return the access key, or raise if it is missing."""
        if not self.unsplash_access_key:
            raise RuntimeError(f"Set {ENV_ACCESS_KEY} in the environment to collect images.")
        return self.unsplash_access_key
