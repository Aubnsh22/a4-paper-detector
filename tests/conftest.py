"""Run regression tests against the documented default detector policy."""
import pytest


@pytest.fixture(autouse=True)
def baseline_policy(monkeypatch):
    monkeypatch.delenv("FOOT_A4_RESIDUAL_POLICY", raising=False)
