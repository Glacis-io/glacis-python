"""pyproject.toml and glacis.__version__ must agree.

0.7.0 shipped with the two desynced; this is the missing gate.
"""

import re
from pathlib import Path

import glacis

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def test_version_sync():
    match = re.search(
        r'^version = "([^"]+)"', PYPROJECT.read_text(), flags=re.MULTILINE
    )
    assert match, "pyproject.toml has no version line"
    assert match.group(1) == glacis.__version__


def test_release_is_0_9_0():
    assert glacis.__version__ == "0.9.0"
