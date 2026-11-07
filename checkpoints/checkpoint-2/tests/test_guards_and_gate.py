import json, os, tempfile, unittest
from tests.common import write_fake_tamas
import mla_pipeline as P


class DataSourceGuards(unittest.TestCase):
    def test_marker_decides_and_mixing_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            logs = os.path.join(t, "logs"); os.makedirs(logs)
            self.assertEqual(P.detect_data_source(logs), "tamas_real")          # no marker -> real
            with self.assertRaises(SystemExit):
                P.detect_data_source(logs, "synthetic_selftest")                 # cannot call real logs synthetic
            open(os.path.join(t, "SYNTHETIC_README.txt"), "w").write("x")
            self.assertEqual(P.detect_data_source(logs), "synthetic_selftest")
            with self.assertRaises(SystemExit):
                P.detect_data_source(logs, "tamas_real")                         # synthetic logs can never be real

    def test_legacy_labels_refused_on_real_data_and_out_dir_not_shared(self):
        with tempfile.TemporaryDirectory() as t:
            tam = os.path.join(t, "T"); write_fake_tamas(tam)
            logs = os.path.join(t, "logs"); os.makedirs(logs)
            with self.assertRaises(SystemExit):
                P.main(["--tamas_dir", tam, "--logs_dir", logs, "--out", os.path.join(t, "o"), "--label_mode", P.LABEL_LEGACY])
            out = os.path.join(t, "o2"); os.makedirs(out)
            json.dump({"data_source": "synthetic_selftest"}, open(os.path.join(out, "run_info.json"), "w"))
            with self.assertRaises(SystemExit):
                P.main(["--tamas_dir", tam, "--logs_dir", logs, "--out", out])    # real run into a synthetic results folder

    def test_too_few_examples_trains_nothing(self):
        from tests.test_parser_and_labels import LabelDefinition as L
        with tempfile.TemporaryDirectory() as t:
            tam = os.path.join(t, "T"); write_fake_tamas(tam)
            logs = L()._logs(t, {0: L._log([("news_gathering_agent", "evil_tool")]), 1: L._log([("news_gathering_agent", "x")])})
            out = os.path.join(t, "o")
            P.main(["--tamas_dir", tam, "--logs_dir", logs, "--out", out])
            info = json.load(open(os.path.join(out, "run_info.json")))
            self.assertEqual(info["status"], "insufficient_data_no_training")
            self.assertEqual(info["data_source"], "tamas_real")
            self.assertFalse(os.path.exists(os.path.join(out, "baseline_results.csv")))
            self.assertIn("No model was trained", open(os.path.join(out, "evaluation_table.md")).read())


class RunnerNeverFabricates(unittest.TestCase):
    def test_blocked_runner_writes_status_and_no_logs(self):
        import tamas_run_subset as R
        with tempfile.TemporaryDirectory() as t:
            tam = os.path.join(t, "T"); write_fake_tamas(tam); open(os.path.join(tam, "autogen_run.py"), "w").write("")
            os.makedirs(os.path.join(tam, "data", "DPI"), exist_ok=True)
            old = os.environ.pop("OPENAI_API_KEY", None)
            try:
                out = os.path.join(t, "o")
                rc = R.main(["--tamas_dir", tam, "--scenario", "news", "--n", "1", "--out", out])
            finally:
                if old is not None:
                    os.environ["OPENAI_API_KEY"] = old
            self.assertEqual(rc, 2)
            st = json.load(open(os.path.join(out, "real_execution_status.json")))
            self.assertEqual(st["status"], "blocked")
            self.assertFalse(os.path.exists(os.path.join(out, "logs")))


if __name__ == "__main__":
    unittest.main()
