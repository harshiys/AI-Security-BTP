import hashlib
import json
from datetime import datetime, timezone, timedelta

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization

MAX_AGE = timedelta(hours=24)
MAX_CLOCK_SKEW = timedelta(minutes=5)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()


def load_private_key(path="scanner_private.pem"):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def load_public_key(path="scanner_public.pem"):
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def key_id(public_key):
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def sign_entry(entry, artifact_path=None, private_key=None, now=None, artifact_sha256=None, artifact_name=None):
    now = now or datetime.now(timezone.utc)
    if artifact_path:
        computed_hash = sha256_file(artifact_path)
        name = artifact_path.replace("\\", "/").split("/")[-1]
    else:
        computed_hash = artifact_sha256
        name = artifact_name or entry.get("item_id", "").split("/")[-1]
    record = {
        "checkpoint_id": entry.get("checkpoint_id", "CKPT1_STATIC"),
        "item_id": entry["item_id"],
        "artifact_name": name,
        "artifact_sha256": computed_hash,
        "verdict": entry["verdict"],
        "score": entry["score"],
        "scan_timestamp": entry.get("timestamp"),
        "timestamp": now.isoformat(),
        "key_id": key_id(private_key.public_key()),
    }
    signature = private_key.sign(canonical(record))
    return {"record": record, "signature": signature.hex()}


def verify_signed(signed, artifact_path, public_key, expected_item_id=None, max_age=MAX_AGE, now=None):
    now = now or datetime.now(timezone.utc)
    try:
        record = signed["record"]
        signature = bytes.fromhex(signed["signature"])
    except (KeyError, TypeError, ValueError):
        return False, "malformed record"
    try:
        public_key.verify(signature, canonical(record))
    except InvalidSignature:
        return False, "signature invalid"
    try:
        if record["key_id"] != key_id(public_key):
            return False, "record signed by a different key"
        if record["verdict"] != "pass":
            return False, f"verdict is {record['verdict']}"
        issued = datetime.fromisoformat(record["timestamp"])
        if issued - now > MAX_CLOCK_SKEW:
            return False, "record timestamp is in the future"
        if now - issued > max_age:
            return False, "record too old"
        if expected_item_id and record["item_id"] != expected_item_id:
            return False, "record is for a different item"
        if sha256_file(artifact_path) != record["artifact_sha256"]:
            return False, "artifact changed since scan"
    except (KeyError, ValueError, FileNotFoundError) as e:
        return False, f"malformed record or missing file ({type(e).__name__})"
    return True, "ok"
