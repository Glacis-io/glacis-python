"""
Tests for the WordFilterControl (glacis/controls/word_filter.py).

These tests require the 'controls' extra to be installed:
    pip install glacis[controls]
"""

import pytest

try:
    from glacis.config import WordFilterControlConfig
    from glacis.controls import WordFilterControl

    CONTROLS_AVAILABLE = True
except ImportError:
    CONTROLS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CONTROLS_AVAILABLE, reason="controls extra not installed")


class TestWordFilterControl:
    """Tests for literal keyword matching."""

    def test_single_term_match(self):
        """Detect a single configured term."""
        config = WordFilterControlConfig(
            enabled=True, entities=["confidential"], if_detected="flag",
        )
        control = WordFilterControl(config)

        result = control.check("This is confidential data")
        assert result.detected is True
        assert "confidential" in result.categories
        assert result.action == "flag"

    def test_multiple_term_match(self):
        """Detect multiple configured terms."""
        config = WordFilterControlConfig(
            enabled=True, entities=["confidential", "proprietary"],
        )
        control = WordFilterControl(config)

        result = control.check("This is confidential and proprietary information")
        assert result.detected is True
        assert "confidential" in result.categories
        assert "proprietary" in result.categories

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        config = WordFilterControlConfig(
            enabled=True, entities=["SECRET"],
        )
        control = WordFilterControl(config)

        result = control.check("This is a secret document")
        assert result.detected is True
        assert "secret" in result.categories

    def test_special_chars_in_terms(self):
        """Special regex characters in terms are escaped safely."""
        config = WordFilterControlConfig(
            enabled=True, entities=["c++", "system.prompt"],
        )
        control = WordFilterControl(config)

        result = control.check("I know c++ and system.prompt")
        assert result.detected is True
        assert "c++" in result.categories
        assert "system.prompt" in result.categories

        # "system_prompt" should NOT match "system.prompt"
        result2 = control.check("The system_prompt is private")
        assert result2.detected is False

    def test_no_match_returns_forward(self):
        """No matches returns action=forward."""
        config = WordFilterControlConfig(
            enabled=True, entities=["confidential", "secret"],
        )
        control = WordFilterControl(config)

        result = control.check("This is a normal public document")
        assert result.detected is False
        assert result.action == "forward"
        assert result.categories == []

    def test_empty_entities_no_match(self):
        """Empty entities list never matches."""
        config = WordFilterControlConfig(enabled=True, entities=[])
        control = WordFilterControl(config)

        result = control.check("This has confidential data")
        assert result.detected is False
        assert result.action == "forward"

    def test_empty_text_returns_forward(self):
        """Empty text returns forward."""
        config = WordFilterControlConfig(
            enabled=True, entities=["secret"],
        )
        control = WordFilterControl(config)

        result = control.check("")
        assert result.detected is False

    def test_action_block(self):
        """Word filter can return block action."""
        config = WordFilterControlConfig(
            enabled=True, entities=["forbidden"], if_detected="block",
        )
        control = WordFilterControl(config)

        result = control.check("This contains forbidden content")
        assert result.detected is True
        assert result.action == "block"

    def test_count_in_metadata(self):
        """Metadata includes match count."""
        config = WordFilterControlConfig(
            enabled=True, entities=["secret"],
        )
        control = WordFilterControl(config)

        result = control.check("secret secret secret")
        assert result.detected is True
        assert result.metadata.get("count") == 3

    def test_control_type(self):
        """Control type is 'word_filter'."""
        config = WordFilterControlConfig(enabled=True, entities=["x"])
        control = WordFilterControl(config)
        assert control.control_type == "word_filter"
