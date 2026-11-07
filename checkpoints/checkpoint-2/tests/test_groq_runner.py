"""
Tests for the Groq TAMAS runner (tamas_run_subset.py).

These never make a real API call. They check:
  - preflight reports a GROQ_API_KEY blocker (not OPENAI) when the key is absent
  - dry_run prints a plan and writes nothing
  - the blocked path writes a truthful status JSON and exits 2
  - the expected nested log path is used
  - --overwrite is accepted
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tamas_run_subset as runner  # noqa: E402


def _mini_tamas(tmp):
    """Create a minimal fake TAMAS dir with one DPI finance task."""
    os.makedirs(os.path.join(tmp, "data", "DPI"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "data", "tools", "autogen"), exist_ok=True)
    open(os.path.join(tmp, "autogen_run.py"), "w").write("# stub\n")
    task = [{"agents": [{"agent_name": "Identity Verification Agent",
                         "agent_description": "d"}],
             "user query": "q"}]
    json.dump(task, open(os.path.join(tmp, "data", "DPI", "finance_DPI.json"), "w"))
    return tmp


class TestGroqRunner(unittest.TestCase):
    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _mini_tamas(tmp)
            out = os.path.join(tmp, "out")
            rc = runner.main(["--tamas_dir", tmp, "--attack", "DPI",
                              "--scenario", "finance", "--n", "1",
                              "--out", out, "--dry_run"])
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(out),
                             "dry_run must not create the output dir")

    def test_preflight_demands_groq_not_openai(self):
        with tempfile.TemporaryDirectory() as tmp:
            _mini_tamas(tmp)

            class A:
                tamas_dir = tmp
                attack = "DPI"
                scenario = "finance"
            problems = runner.preflight(A())
            joined = " ".join(problems)
            # When the key is absent we must name GROQ_API_KEY.
            if not os.getenv("GROQ_API_KEY"):
                self.assertIn("GROQ_API_KEY", joined)
            # We must never demand OPENAI_API_KEY for Groq execution.
            self.assertNotIn("OPENAI_API_KEY", joined)

    def test_blocked_without_key_writes_status_and_exits_2(self):
        if os.getenv("GROQ_API_KEY"):
            self.skipTest("GROQ_API_KEY is set; blocked path not exercised")
        with tempfile.TemporaryDirectory() as tmp:
            _mini_tamas(tmp)
            out = os.path.join(tmp, "out")
            rc = runner.main(["--tamas_dir", tmp, "--attack", "DPI",
                              "--scenario", "finance", "--n", "1", "--out", out])
            self.assertEqual(rc, 2)
            status = os.path.join(out, "real_execution_status.json")
            self.assertTrue(os.path.exists(status))
            data = json.load(open(status))
            self.assertEqual(data["status"], "blocked")
            self.assertEqual(data["provider"], "groq")
            self.assertEqual(data["model"], "openai/gpt-oss-20b")

    def test_log_path_nesting_in_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            _mini_tamas(tmp)
            # capture stdout of dry-run and check the nested path appears
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                runner.main(["--tamas_dir", tmp, "--attack", "DPI",
                             "--scenario", "finance", "--config", "RoundRobin",
                             "--n", "1", "--out", "OUTX", "--dry_run"])
            text = buf.getvalue()
            self.assertIn(os.path.join("OUTX", "logs", "DPI", "RoundRobin",
                                       "DPI_finance_0.txt"), text)

    def test_overwrite_flag_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _mini_tamas(tmp)
            # dry_run + overwrite should parse and return 0
            rc = runner.main(["--tamas_dir", tmp, "--attack", "DPI",
                              "--scenario", "finance", "--n", "1",
                              "--out", os.path.join(tmp, "o"),
                              "--dry_run", "--overwrite"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
