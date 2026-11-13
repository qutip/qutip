import importlib
import multiprocessing
import subprocess
import sys

import pytest


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


def test_import_does_not_fix_application_start_method():
    code = (
        "import multiprocessing as mp\n"
        "import qutip\n"
        "assert mp.get_start_method(allow_none=True) is None\n"
        "mp.set_start_method('spawn')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
