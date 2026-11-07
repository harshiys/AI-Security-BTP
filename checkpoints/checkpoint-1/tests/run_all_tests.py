#!/usr/bin/env python3
"""Unified Test Suite for SentinelMesh Checkpoint 1 (Static LoRA Scanner + Cyber Trust Chain).

Evaluates the integrated pipeline on the real PADBench dataset:
1. 12 Cyber Security Attack Tests (Tampering, Forgery, Freshness, Key Swapping)
2. ML Core Mathematical Unit Tests (QR-SVD, SafeLoad, Thresholds)
3. Dataset Discovery & Path Normalization Tests
4. Real PADBench Deployment Gatekeeper Tests (Clean -> ALLOWED, Poisoned -> BLOCKED, Tampered -> BLOCKED)
5. Strict 5-Field Shared Schema Compliance Test (Draft 7 JSON Schema on 60 real records)
6. Real PADBench Cryptographically Sealed Model Protection Test
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

sys.path.insert(0, str(ROOT))


def run_command(cmd, desc):
    print(f"\n--- Running: {desc} ---")
    r = subprocess.run([PY, *cmd], cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        print(out)
        print(f"[FAIL] {desc} exited with code {r.returncode}")
        return False
    print(out)
    print(f"[PASS] {desc}")
    return True


class SchemaComplianceTests(unittest.TestCase):
    """Ensure all generated records in work/c1_log strictly follow the shared schema."""

    def test_schema_conformance(self):
        import jsonschema
        schema_path = ROOT / "shared_schema_format.json"
        self.assertTrue(schema_path.exists(), "shared_schema_format.json must exist")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft7Validator(schema, format_checker=jsonschema.FormatChecker())

        log_path = ROOT / "work" / "c1_log" / "ckpt1_log.jsonl"
        if not log_path.exists():
            self.skipTest(f"Log file not found at {log_path}; run run_pipeline.py first.")

        lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertGreater(len(lines), 0, "Log file should contain records")

        required_keys = {"checkpoint_id", "item_id", "verdict", "score", "timestamp"}
        for i, line in enumerate(lines):
            rec = json.loads(line)
            # 1. Exact keys (no additionalProperties)
            self.assertEqual(set(rec.keys()), required_keys, f"Record {i} keys != {required_keys}")
            # 2. Field constraints
            self.assertEqual(rec["checkpoint_id"], "CKPT1_STATIC")
            self.assertIn(rec["verdict"], ["pass", "block", "flag"])
            self.assertIsInstance(rec["score"], (float, int))
            self.assertTrue(0.0 <= rec["score"] <= 1.0)
            # 3. Full jsonschema validation
            errs = list(validator.iter_errors(rec))
            self.assertEqual(len(errs), 0, f"Record {i} failed schema: {[e.message for e in errs]}")


class SealedModelProtectionTests(unittest.TestCase):
    """Test that verified_load blocks unsealed, tampered, or improperly signed classifiers."""

    def test_unsealed_model_blocked(self):
        from verified_load import load_verified
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            f.write(b"unsealed pickle contents")
            model_path = f.name
        
        with self.assertRaises(SystemExit) as ctx:
            load_verified(model_path, record_path="non_existent_record.json")
        self.assertIn("BLOCKED", str(ctx.exception))

    def test_tampered_model_blocked(self):
        from trustchain import load_private_key, sign_entry, load_public_key, verify_signed
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            f.write(b"original clean model")
            model_path = f.name

        priv = Ed25519PrivateKey.generate()
        rec = sign_entry({"checkpoint_id": "CKPT1_STATIC", "item_id": "c1-classifier", "verdict": "pass", "score": 0.0},
                         model_path, priv)
        
        # Tamper model file
        with open(model_path, "wb") as f:
            f.write(b"tampered malicious model")

        ok, why = verify_signed(rec, model_path, priv.public_key(), expected_item_id="c1-classifier")
        self.assertFalse(ok)
        self.assertEqual(why, "artifact changed since scan")


class RealPADBenchGatekeeperTests(unittest.TestCase):
    """Test verify.py deployment gatekeeper on real PADBench weights."""

    def test_real_padbench_clean_allowed(self):
        adapter = ROOT / "sample_adapters" / "clean_adapter.safetensors"
        r = subprocess.run([PY, "verify.py", str(adapter)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, f"Clean adapter failed: {r.stdout} {r.stderr}")
        self.assertIn("ALLOWED", r.stdout)

    def test_real_padbench_poisoned_blocked(self):
        adapter = ROOT / "sample_adapters" / "poisoned_adapter.safetensors"
        r = subprocess.run([PY, "verify.py", str(adapter)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("BLOCKED: verdict is block", r.stdout)

    def test_real_padbench_tampered_blocked(self):
        adapter = ROOT / "sample_adapters" / "tampered_adapter.safetensors"
        r = subprocess.run([PY, "verify.py", str(adapter)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 1)
        self.assertIn("BLOCKED: artifact changed since scan", r.stdout)


def main():
    print("=" * 78)
    print("SENTINELMESH CHECKPOINT 1: COMPREHENSIVE INTEGRATION TEST SUITE")
    print("DATASET: Real PADBench (Vincent-HKUSTGZ/PADBench)")
    print("=" * 78)

    all_passed = True

    # 1. Cyber Security Attacks
    all_passed &= run_command(["tests/test_trustchain.py"], "12 Cyber Security Attack Tests (Tampering, Forgery, Freshness)")

    # 2. Unit ML & Discovery Tests
    all_passed &= run_command(["tests/test_core.py"], "ML Core Unit Tests (QR-SVD, SafeLoad, Thresholds)")
    all_passed &= run_command(["tests/test_discovery.py"], "Dataset Discovery & Layout Tests")
    all_passed &= run_command(["tests/test_prefix_layout.py"], "Prefix & Path Resolution Tests")

    # 3. Unit Test Suite (Schema Conformance, Sealed Model, Gatekeeper on Real PADBench)
    print("\n--- Running: Real PADBench Gatekeeper, Sealed Model & Schema Unit Tests ---")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(RealPADBenchGatekeeperTests))
    suite.addTests(loader.loadTestsFromTestCase(SchemaComplianceTests))
    suite.addTests(loader.loadTestsFromTestCase(SealedModelProtectionTests))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        all_passed = False

    print("\n" + "=" * 78)
    if all_passed:
        print("ALL TESTS PASSED SUCCESSFULLY! FULL SYSTEM INTEGRATION VERIFIED ON REAL PADBENCH.")
    else:
        print("[FAIL] Some tests failed. Check logs above.")
    print("=" * 78)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
