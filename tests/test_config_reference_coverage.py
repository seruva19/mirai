"""CONFIG_REFERENCE.md stays in lock-step with the config schema.

The reference doc is the human/agent-facing source of truth for every config
key. This test parses the documented key names out of responsive key sections and
asserts they equal the schema's authoritative leaf-key sets
(`mirai.config.schema.all_config_keys()`) in BOTH directions:

* a config key that exists in the schema but is not documented -> fail,
* a key documented in the reference that no longer exists in the schema -> fail.

Parse contract (kept deliberately simple/robust):

* A section is any level-two heading whose text contains a backticked table tag,
  e.g. ``## `[model]` `` or ``## `[dataset.moe_routing]` ``. The tag ``[root]``
  maps to the root ("") table. Any other level-two heading resets the active
  section, so Presets / env-var appendices are ignored.
* Inside a section, each level-three heading containing one backticked token
  contributes that token as a documented key.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from mirai.config.schema import all_config_keys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE = _REPO_ROOT / "CONFIG_REFERENCE.md"

_SECTION_TAG = re.compile(r"^##\s+.*`\[([^\]]+)\]`")
_KEY_HEADING = re.compile(r"^###\s+`([^`]+)`\s*$")


def _parse_documented_keys(text: str) -> dict[str, set[str]]:
    documented: dict[str, set[str]] = {}
    section: str | None = None
    collecting_keys = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            tag = _SECTION_TAG.match(stripped)
            if tag:
                name = tag.group(1).strip()
                section = "" if name == "root" else name
                documented.setdefault(section, set())
                collecting_keys = True
            else:
                section = None
                collecting_keys = False
            continue
        if section is None:
            continue
        match = _KEY_HEADING.match(stripped)
        if match and collecting_keys:
            documented[section].add(match.group(1).strip())
        elif stripped.startswith("### "):
            collecting_keys = False
    return documented


class ConfigReferenceCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(_REFERENCE.exists(), f"{_REFERENCE} is missing.")
        self.documented = _parse_documented_keys(_REFERENCE.read_text(encoding="utf-8"))
        self.schema = all_config_keys()

    def test_same_tables_documented(self) -> None:
        self.assertEqual(
            set(self.schema),
            set(self.documented),
            "CONFIG_REFERENCE.md sections do not match the schema tables.",
        )

    def test_every_key_documented_and_no_stray_keys(self) -> None:
        for table, expected in self.schema.items():
            documented = self.documented.get(table, set())
            missing = expected - documented
            extra = documented - expected
            label = table or "root"
            self.assertFalse(
                missing,
                f"[{label}] keys in schema but undocumented in CONFIG_REFERENCE.md: "
                f"{sorted(missing)}",
            )
            self.assertFalse(
                extra,
                f"[{label}] keys documented in CONFIG_REFERENCE.md but absent from "
                f"schema: {sorted(extra)}",
            )

    def test_parser_finds_expected_volume(self) -> None:
        # Guards against the parser silently matching nothing (e.g. a format
        # change that breaks extraction would otherwise make the equality checks
        # vacuously pass on empty sets).
        total = sum(len(v) for v in self.documented.values())
        self.assertEqual(total, sum(len(v) for v in self.schema.values()))
        self.assertGreater(total, 150)


if __name__ == "__main__":
    unittest.main()
