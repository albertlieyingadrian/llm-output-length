"""System prompts must survive YAML unescaping.

`hf_system_prompt: "... within \\boxed{}."` in a DOUBLE-quoted YAML scalar makes
`\\b` a BACKSPACE (0x08), so every run so far instructed the model to "put your
final answer within \\x08oxed{}." The model still answered, the pipeline still
ran, and nothing anywhere reported it -- the only symptom would be slightly
worse formatting compliance, which nobody was measuring.

Single-quoted YAML performs no escape processing, which is what a LaTeX macro
needs.
"""

from pathlib import Path

import pytest
import yaml

CONF = Path(__file__).resolve().parents[1] / "src" / "miol" / "conf" / "data"
CONFIGS = sorted(p for p in CONF.glob("*.yaml"))
WITH_PROMPT = [p for p in CONFIGS
               if isinstance(yaml.safe_load(p.read_text()), dict)
               and yaml.safe_load(p.read_text()).get("hf_system_prompt")]

# Every ASCII control character. A system prompt is text a human wrote for a
# model to read; none of these belong in one.
CONTROL = {chr(c) for c in range(32)} | {chr(127)}


@pytest.mark.parametrize("path", WITH_PROMPT, ids=lambda p: p.stem)
class TestNoControlCharacters:
    def _prompt(self, path):
        return yaml.safe_load(path.read_text())["hf_system_prompt"]

    def test_no_control_characters_survive_yaml(self, path):
        s = self._prompt(path)
        bad = sorted({c for c in s if c in CONTROL})
        assert not bad, (
            f"{path.name} yields control characters {[hex(ord(c)) for c in bad]}: "
            f"{s!r}. A double-quoted YAML scalar turns \\b into backspace, \\t into "
            f"tab and \\f into formfeed. Single-quote the scalar.")

    def test_a_latex_macro_stays_a_latex_macro(self, path):
        s = self._prompt(path)
        if "oxed" in s:
            assert "\\boxed" in s, (
                f"{path.name} lost the backslash from \\boxed: {s!r}")


def test_the_trap_is_real():
    """Guards the guard: double-quoted YAML really does eat the backslash."""
    assert yaml.safe_load(r'x: "within \boxed{}"')["x"] == "within \x08oxed{}"
    assert yaml.safe_load(r"x: 'within \boxed{}'")["x"] == "within \\boxed{}"


def test_at_least_one_config_actually_has_a_prompt():
    # Otherwise the parametrized class above passes vacuously.
    assert len(WITH_PROMPT) >= 2
