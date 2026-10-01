"""Shared test configuration.

Momentum's database recovery scan (momentum.recovery) deliberately looks at
real user-data locations to adopt orphaned databases. Under pytest that
would leak the developer's actual tasks/results into tmp targets, so the
scan is disabled globally here; tests that exercise recovery itself delete
this variable explicitly (see tests/test_recovery.py).

Config isolation matters just as much, and for a sharper reason: several
suites call the real ``save_config`` / ``set_db_path``, which persist to
``~/.config/momentum/config.json``. Without redirection a test run rewrites
the developer's own configuration -- and a test that sets a tmp ``db_path``
leaves the app pointing at a pytest temporary directory that no longer
exists. The autouse fixture below redirects the config file and directory
into a per-session temp tree so no test can reach the real one.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MOMENTUM_DISABLE_RECOVERY", "1")


@pytest.fixture(autouse=True)
def _isolate_config_file(tmp_path_factory):
    """Redirect the on-disk config into a temp tree for the whole session.

    Patching the module attributes rather than the environment is deliberate:
    ``config.py`` resolves ``_CONFIG_FILE`` once at import time, so an env var
    would not affect the already-imported module. A test that writes config
    therefore lands in ``tmp_path`` instead of the developer's home
    directory.
    """
    from momentum import config as cfg

    config_dir = tmp_path_factory.mktemp("momentum-config")
    original_file = cfg._CONFIG_FILE
    original_dir = cfg._CONFIG_DIR
    cfg._CONFIG_DIR = config_dir
    cfg._CONFIG_FILE = config_dir / "config.json"
    try:
        yield cfg._CONFIG_FILE
    finally:
        cfg._CONFIG_FILE = original_file
        cfg._CONFIG_DIR = original_dir
