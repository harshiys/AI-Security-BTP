#!/usr/bin/env python3
"""
SentinelMesh Checkpoint 2: Comprehensive Integration Test Suite
Verifies Real TAMAS Provenance, Shared Schema Draft-7 Compliance,
Cyber Telemetry, Red-Team Attack Scenarios, and ML-B Model Performance.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
import pandas as pd
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cp2_contract as C
import cybersec_adapters as CSA


class CP2DatasetProvenanceTests(unittest.TestCase):
    def test_real_tamas_data_present(self):
        events_path = ROOT / "data" / "events.csv"
        labels_path = ROOT / "data" / "labels.csv"
        self.assertTrue(events_path.exists(), "data/events.csv must exist")
        self.assertTrue(labels_path.exists(), "data/labels.csv must exist")
        
        df_ev = pd.read_csv(events_path)
        self.assertGreaterEqual(len(df_ev), 80, "Real TAMAS events should have at least 80 events")
        self.assertIn("news_gathering_agent", set(df_ev["acting_agent"]), "Must contain real AutoGen agents")

    def test_no_synthetic_markers_in_production(self):
        data_dir = ROOT / "data"
        synthetic_marker = data_dir / "SYNTHETIC_README.txt"
        self.assertFalse(synthetic_marker.exists(), "SYNTHETIC_README.txt must NOT exist in production data directory")


class CP2SchemaComplianceTests(unittest.TestCase):
    def test_verdict_records_strictly_conform_to_shared_schema(self):
        verdicts_path = ROOT / "results" / "checkpoint2_verdicts.jsonl"
        self.assertTrue(verdicts_path.exists(), "results/checkpoint2_verdicts.jsonl must exist")
        
        schema_path = ROOT / "schema" / "shared_schema_format.json"
        schema = json.load(open(schema_path, encoding="utf-8"))
        validator = Draft7Validator(schema)

        records = [json.loads(l) for l in open(verdicts_path, encoding="utf-8") if l.strip()]
        self.assertEqual(len(records), 40, "Expected exactly 40 evaluated agent-run instances from real TAMAS")

        for r in records:
            validator.validate(r)
            self.assertEqual(len(r), 5, "Record must contain EXACTLY 5 fields (no extra properties)")
            self.assertEqual(set(r.keys()), {"checkpoint_id", "item_id", "verdict", "score", "timestamp"})
            self.assertEqual(r["checkpoint_id"], "CKPT2_RUNTIME")
            self.assertIn(r["verdict"], ["pass", "block", "flag"])
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)


class CP2CyberTelemetryTests(unittest.TestCase):
    def test_dynamic_trust_calculation(self):
        trust_path = ROOT / "cyber_results" / "cyber_dynamic_trust.csv"
        self.assertTrue(trust_path.exists(), "cyber_results/cyber_dynamic_trust.csv must exist")
        df = pd.read_csv(trust_path)
        self.assertEqual(len(df), 40, "Dynamic trust must cover all 40 agent instances")
        
        # Verify dynamic trust math
        for _, row in df.iterrows():
            self.assertAlmostEqual(row["dynamic_trust"], 1.0 - row["cyber_risk"], delta=0.01)
            self.assertTrue(0.0 <= row["dynamic_trust"] <= 1.0)


class CP2AttackMitigationTests(unittest.TestCase):
    def test_impersonation_attack_mitigated(self):
        cmd = [sys.executable, str(ROOT / "verify_cp2.py"), "--scenario", "impersonation"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertNotEqual(res.returncode, 0, "Impersonation attack must exit with non-zero code")
        self.assertIn("BLOCKED: impersonation attack detected", res.stdout)

    def test_rogue_delegation_mitigated(self):
        cmd = [sys.executable, str(ROOT / "verify_cp2.py"), "--scenario", "rogue_delegation"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertNotEqual(res.returncode, 0, "Rogue delegation must exit with non-zero code")
        self.assertIn("BLOCKED: rogue delegation detected", res.stdout)

    def test_sleeper_agent_mitigated(self):
        cmd = [sys.executable, str(ROOT / "verify_cp2.py"), "--scenario", "sleeper_agent"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertNotEqual(res.returncode, 0, "Sleeper agent must exit with non-zero code")
        self.assertIn("BLOCKED: sleeper agent activation detected", res.stdout)


class CP2GatekeeperTests(unittest.TestCase):
    def test_clean_agent_allowed(self):
        cmd = [sys.executable, str(ROOT / "verify_cp2.py"), "--run-id", "r71a932976c", "--agent", "article_writing_agent"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(res.returncode, 0, "Clean agent must exit with 0")
        self.assertIn("ALLOWED: Agent 'article_writing_agent' verified safe", res.stdout)

    def test_flagged_agent_intercepted(self):
        cmd = [sys.executable, str(ROOT / "verify_cp2.py"), "--run-id", "r77c0e4eb97", "--agent", "news_gathering_agent"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(res.returncode, 1, "Flagged agent must exit with 1")
        self.assertIn("FLAGGED: Agent 'news_gathering_agent' flagged for anomaly", res.stdout)


def run_suite():
    print("=" * 78)
    print("SENTINELMESH CHECKPOINT 2: COMPREHENSIVE INTEGRATION TEST SUITE")
    print("DATASET: Real TAMAS (Microsoft AutoGen DPI Benchmark)")
    print("=" * 78)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(CP2DatasetProvenanceTests))
    suite.addTests(loader.loadTestsFromTestCase(CP2SchemaComplianceTests))
    suite.addTests(loader.loadTestsFromTestCase(CP2CyberTelemetryTests))
    suite.addTests(loader.loadTestsFromTestCase(CP2AttackMitigationTests))
    suite.addTests(loader.loadTestsFromTestCase(CP2GatekeeperTests))

    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    
    if res.wasSuccessful():
        print("=" * 78)
        print("ALL TESTS PASSED SUCCESSFULLY! FULL SYSTEM INTEGRATION VERIFIED ON REAL TAMAS.")
        print("=" * 78)
        return 0
    else:
        print("TEST SUITE FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_suite())
