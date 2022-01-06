"""
Tests for qutip.hardware_info.
"""

from qutip.hardware_info import hardware_info


def test_hardware_info():
    info = hardware_info()
    assert info["os"] in ("Windows", "FreeBSD", "Linux", "Mac OSX")
    assert info["cpus"] > 0
    assert isinstance(info["cpus"], int)
    cpu_freq = info.get("cpu_freq", 0.0)
    assert cpu_freq >= 0.
    assert isinstance(cpu_freq, float)
