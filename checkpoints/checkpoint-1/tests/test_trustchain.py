import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from trustchain import sign_entry, verify_signed, sha256_file

ENTRY = {"checkpoint_id": "C1", "item_id": "id-0001",
         "verdict": "pass", "score": 0.12,
         "timestamp": "2026-10-16T00:00:00+00:00"}


class TrustChainTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "adapter.bin")
        with open(self.path, "wb") as f:
            f.write(b"original adapter bytes")
        self.priv = Ed25519PrivateKey.generate()
        self.pub = self.priv.public_key()

    def tearDown(self):
        self.dir.cleanup()

    def signed(self, entry=ENTRY):
        return sign_entry(entry, self.path, self.priv)

    def test_valid_is_allowed(self):
        ok, why = verify_signed(self.signed(), self.path, self.pub)
        self.assertTrue(ok, why)

    def test_attack_a_edit_score(self):
        s = self.signed()
        s["record"]["score"] = 0.01
        ok, why = verify_signed(s, self.path, self.pub)
        self.assertFalse(ok)
        self.assertEqual(why, "signature invalid")

    def test_attack_b_swap_file(self):
        s = self.signed()
        with open(self.path, "wb") as f:
            f.write(b"malicious adapter bytes")
        ok, why = verify_signed(s, self.path, self.pub)
        self.assertFalse(ok)
        self.assertEqual(why, "artifact changed since scan")

    def test_attack_c_swap_file_and_fix_hash(self):
        s = self.signed()
        with open(self.path, "wb") as f:
            f.write(b"malicious adapter bytes")
        s["record"]["artifact_sha256"] = sha256_file(self.path)
        ok, why = verify_signed(s, self.path, self.pub)
        self.assertFalse(ok)
        self.assertEqual(why, "signature invalid")

    def test_flip_verdict_to_pass(self):
        s = self.signed({**ENTRY, "verdict": "block"})
        s["record"]["verdict"] = "pass"
        ok, why = verify_signed(s, self.path, self.pub)
        self.assertFalse(ok)
        self.assertEqual(why, "signature invalid")

    def test_honest_block_verdict_is_blocked(self):
        ok, why = verify_signed(self.signed({**ENTRY, "verdict": "flag"}),
                                self.path, self.pub)
        self.assertFalse(ok)
        self.assertIn("verdict is flag", why)

    def test_expired_record(self):
        s = self.signed()
        later = datetime.now(timezone.utc) + timedelta(hours=25)
        ok, why = verify_signed(s, self.path, self.pub, now=later)
        self.assertFalse(ok)
        self.assertEqual(why, "record too old")

    def test_future_timestamp(self):
        future = datetime.now(timezone.utc) + timedelta(days=2)
        s = sign_entry(ENTRY, self.path, self.priv, now=future)
        ok, why = verify_signed(s, self.path, self.pub)
        self.assertFalse(ok)
        self.assertEqual(why, "record timestamp is in the future")

    def test_attacker_signs_with_own_key(self):
        attacker = Ed25519PrivateKey.generate()
        s = sign_entry(ENTRY, self.path, attacker)
        ok, why = verify_signed(s, self.path, self.pub)
        self.assertFalse(ok)
        self.assertEqual(why, "signature invalid")

    def test_record_for_wrong_item(self):
        ok, why = verify_signed(self.signed(), self.path, self.pub,
                                expected_item_id="id-9999")
        self.assertFalse(ok)
        self.assertEqual(why, "record is for a different item")

    def test_pass_record_reused_on_other_file(self):
        s = self.signed()
        other = os.path.join(self.dir.name, "other.bin")
        with open(other, "wb") as f:
            f.write(b"a different file entirely")
        ok, why = verify_signed(s, other, self.pub)
        self.assertFalse(ok)
        self.assertEqual(why, "artifact changed since scan")

    def test_malformed_record(self):
        ok, why = verify_signed({"nope": 1}, self.path, self.pub)
        self.assertFalse(ok)
        self.assertEqual(why, "malformed record")


if __name__ == "__main__":
    unittest.main(verbosity=2)
