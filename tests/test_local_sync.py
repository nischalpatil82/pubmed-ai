"""Safety checks for the Windows updater; no network or server is started."""
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops import local_sync


def result(stdout="", returncode=0):
    return subprocess.CompletedProcess(["git"], returncode, stdout=stdout, stderr="")


class LocalSyncTest(unittest.TestCase):
    def test_preparation_uses_selected_dataset_without_stopping_fallback_on_failure(self):
        dataset = Path("D:/pubmed-ai/covid-files/dataset.json")
        with patch.object(local_sync.subprocess, "run", return_value=result(returncode=1)) as run:
            local_sync.prepare_performance(dataset, {}, None)
        self.assertEqual(run.call_args.args[0][-3:], ["performance", "--dataset", str(dataset)])
        with patch.object(local_sync.subprocess, "run") as run:
            local_sync.prepare_performance(dataset, {"PUBMED_PREPARE_PERFORMANCE": "0"}, None)
            run.assert_not_called()
        with patch.object(local_sync.subprocess, "run", side_effect=subprocess.TimeoutExpired("prepare", 1800)):
            local_sync.prepare_performance(dataset, {}, None)

    def test_dirty_checkout_is_not_fetched_or_updated(self):
        calls = []

        def fake_git(*args, **kwargs):
            calls.append(args[0])
            return result("covid-files\n" if args[0] == "branch" else " M pipeline/app.py\n")

        with patch.object(local_sync, "git", side_effect=fake_git):
            with self.assertRaisesRegex(RuntimeError, "local changes"):
                local_sync.update_available()
        self.assertEqual(calls, ["branch", "status"])

    def test_fast_forward_is_reported_without_merging_during_check(self):
        current, target = "a" * 40, "b" * 40
        calls = []

        def fake_git(*args, **kwargs):
            calls.append(args[0])
            if args[0] == "branch":
                return result("covid-files\n")
            if args[0] == "status":
                return result()
            if args[0] == "fetch":
                return result()
            if args[0] == "rev-parse":
                return result((current if args[1] == "HEAD" else target) + "\n")
            if args[0] == "merge-base":
                return result()
            if args[0] == "diff":
                return result("pipeline/app.py\n")
            raise AssertionError(args)

        with patch.object(local_sync, "git", side_effect=fake_git):
            self.assertEqual(local_sync.update_available(), (current, target))
        self.assertNotIn("merge", calls)

    def test_dependency_change_or_divergence_requires_supervision(self):
        current, target = "a" * 40, "b" * 40

        def fake_git(*args, **kwargs):
            if args[0] == "branch":
                return result("covid-files\n")
            if args[0] == "status":
                return result()
            if args[0] == "rev-parse":
                return result((current if args[1] == "HEAD" else target) + "\n")
            if args[0] == "merge-base":
                return result(returncode=1)
            if args[0] == "diff":
                return result("requirements.txt\n")
            raise AssertionError(args)

        with patch.object(local_sync, "git", side_effect=fake_git):
            with self.assertRaisesRegex(RuntimeError, "diverged"):
                local_sync.update_available(fetch=False)
        with patch.object(local_sync, "git", side_effect=lambda *a, **kw:
                          result() if a[0] == "merge-base" else fake_git(*a, **kw)):
            with self.assertRaisesRegex(RuntimeError, "requirements"):
                local_sync.update_available(fetch=False)


if __name__ == "__main__":
    unittest.main()
