from __future__ import annotations

import json
import subprocess
import sys
import unittest

from mirai.core.dataset.split import build_split_assignments, split_ids


class DatasetSplitTests(unittest.TestCase):
    def _sample_items(self) -> list[dict[str, str]]:
        return [
            {"sample_id": "a1", "group_id": "gA"},
            {"sample_id": "a2", "group_id": "gA"},
            {"sample_id": "b1", "group_id": "gB"},
            {"sample_id": "c1", "group_id": "gC"},
            {"sample_id": "d1", "group_id": "gD"},
            {"sample_id": "e1", "group_id": "gE"},
        ]

    def test_split_determinism_same_seed(self) -> None:
        samples = self._sample_items()
        s1 = build_split_assignments(samples, split_seed=123)
        s2 = build_split_assignments(samples, split_seed=123)
        self.assertEqual(s1, s2)

    def test_group_leakage_guard(self) -> None:
        samples = self._sample_items()
        assignments = build_split_assignments(samples, split_seed=42)
        self.assertEqual(assignments["a1"], assignments["a2"])

    def test_eval_isolation_test_split_not_in_train_or_val(self) -> None:
        samples = self._sample_items()
        assignments = build_split_assignments(samples, split_seed=7)
        train_ids, val_ids, test_ids = split_ids(assignments)
        self.assertTrue(set(test_ids).isdisjoint(train_ids))
        self.assertTrue(set(test_ids).isdisjoint(val_ids))

    def test_split_hash_stability_across_fresh_interpreters(self) -> None:
        script = (
            "import json;"
            "from mirai.core.dataset.split import build_split_assignments;"
            "samples=["
            "{'sample_id':'a1','group_id':'gA'},"
            "{'sample_id':'a2','group_id':'gA'},"
            "{'sample_id':'b1','group_id':'gB'}"
            "];"
            "print(json.dumps(build_split_assignments(samples, split_seed=999), sort_keys=True))"
        )
        r1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        r2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        self.assertEqual(r1.returncode, 0, msg=r1.stderr)
        self.assertEqual(r2.returncode, 0, msg=r2.stderr)
        self.assertEqual(json.loads(r1.stdout), json.loads(r2.stdout))


if __name__ == "__main__":
    unittest.main()
