"""Fixtures. The config builders and the traffic harness live in `tests/sim.py`."""

from __future__ import annotations

import pytest

from sir.clock import VirtualClock


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock()
