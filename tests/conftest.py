"""Shared test configuration.

Momentum's database recovery scan (momentum.recovery) deliberately looks at
real user-data locations to adopt orphaned databases. Under pytest that
would leak the developer's actual tasks/results into tmp targets, so the
scan is disabled globally here; tests that exercise recovery itself delete
this variable explicitly (see tests/test_recovery.py).
"""

from __future__ import annotations

import os

os.environ.setdefault("MOMENTUM_DISABLE_RECOVERY", "1")
