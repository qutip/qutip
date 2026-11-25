import importlib
import multiprocessing
import subprocess
import sys
from pathlib import Path

import pytest
from qutip.settings import Settings


settings_module = importlib.import_module("qutip.settings")


@pytest.mark.parametrize(
    ("version", "platform", "expected"),
    [
        ((3, 12), "linux", "fork"),
        ((3, 14), "linux", "fork"),
        ((3, 14), "darwin", "fork"),
        ((3, 14), "win32", None),
        ((3, 15), "linux", None),
        ((3, 15), "darwin", None),
        ((3, 15), "win32", None),
    ],
)
def test_initial_mp_context(monkeypatch, version, platform, expected):
    monkeypatch.setattr(settings_module.sys, "version_info", version)
    monkeypatch.setattr(settings_module.sys, "platform", platform)
    monkeypatch.setattr(settings_module.multiprocessing, "get_context", lambda name: name)
    assert settings_module.Settings().mp_context == expected


def test_mp_context_override():
    settings = settings_module.Settings()
    context = multiprocessing.get_context("spawn")
    settings.mp_context = context
    assert settings.mp_context is context
    settings.mp_context = "spawn"
    assert settings.mp_context.get_start_method() == "spawn"
    settings.mp_context = None
    assert settings.mp_context is None
    with pytest.raises(ValueError):
        settings.mp_context = "invalid"
    with pytest.raises(TypeError):
        settings.mp_context = 42
    assert settings.mp_context is None


def test_uses_platformdirs_config_dir(monkeypatch, tmp_path):
    platformdirs = pytest.importorskip("platformdirs")
    target = tmp_path / "data" / "qutip"   # parent doesn't exist yet
    monkeypatch.setattr(platformdirs, "user_data_dir",
                        lambda *a, **k: str(target))
    s = Settings()
    assert s.tmproot == str(target)
    assert target.is_dir()


def test_fallback_without_platformdirs(monkeypatch, tmp_path):
    # A None entry in sys.modules makes `import platformdirs` raise ImportError
    monkeypatch.setitem(sys.modules, "platformdirs", None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    s = Settings()
    assert s.tmproot == str(tmp_path / ".qutip")


def test_fallback_when_dir_not_writable(monkeypatch):
    def fail(*a, **k):
        raise OSError("read-only")
    monkeypatch.setattr(Path, "mkdir", fail)
    s = Settings()
    assert s.tmproot == "."
