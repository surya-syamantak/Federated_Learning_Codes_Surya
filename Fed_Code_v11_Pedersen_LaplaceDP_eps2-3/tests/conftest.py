import pytest

from fedrf_diff import zkp


@pytest.fixture(autouse=True)
def fast_zkp_group(monkeypatch):
    """Tests use the fast (insecure) 512-bit group; runs default to pedersen2048."""
    monkeypatch.setattr(zkp, "GROUP", "test512")
