from pathlib import Path

import pytest

from watermark_detector.config import Config


def test_defaults_match_original_run():
    cfg = Config()
    assert cfg.dataset_size == 450
    assert cfg.epochs == 20
    assert cfg.batch_size == 32
    assert len(cfg.image_size) == 2
    assert isinstance(cfg.seed, int)


def test_access_key_read_from_env():
    cfg = Config.from_env({"UNSPLASH_ACCESS_KEY": "abc123"})
    assert cfg.unsplash_access_key == "abc123"


def test_access_key_absent_is_none():
    assert Config.from_env({}).unsplash_access_key is None
    assert Config().unsplash_access_key is None


def test_overrides_apply(tmp_path):
    cfg = Config.from_env({}, epochs=2, data_dir=tmp_path)
    assert cfg.epochs == 2
    assert cfg.data_dir == Path(tmp_path)


def test_require_key_raises_without_key():
    with pytest.raises(RuntimeError, match="UNSPLASH_ACCESS_KEY"):
        Config().require_access_key()


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError):
        Config(batch_size=0)


def test_dotenv_loaded_without_override(tmp_path, monkeypatch):
    import watermark_detector.config as c

    (tmp_path / (chr(46) + "env")).write_text("WMD_TEST_VAR=file\nWMD_KEPT=file\n")
    monkeypatch.setattr(c, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setenv("WMD_KEPT", "shell")
    monkeypatch.delenv("WMD_TEST_VAR", raising=False)
    c.load_dotenv_files()
    import os

    assert os.environ["WMD_TEST_VAR"] == "file"
    assert os.environ["WMD_KEPT"] == "shell"
    monkeypatch.delenv("WMD_TEST_VAR", raising=False)
