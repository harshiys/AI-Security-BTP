"""
The ONE output contract for SentinelMesh-facing CP2 records: schema/shared_schema_format.json (copied unchanged from the team).
Fields: checkpoint_id, item_id, verdict (pass|block|flag), score (0..1, 1.0 = highest risk), timestamp (UTC ISO8601). Nothing else is allowed.

ASSUMPTION TO CONFIRM WITH THE TEAM: the schema only says checkpoint_id is a static scope string and gives 'CKPT1_STATIC' for Day 1.
CP2 uses CHECKPOINT_ID below. Change this one constant if the team names it differently.
"""
import datetime, json, os
from jsonschema import Draft7Validator

CHECKPOINT_ID = "CKPT2_RUNTIME"
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema", "shared_schema_format.json")
SCHEMA = json.load(open(SCHEMA_PATH, encoding="utf-8"))
_VALIDATOR = Draft7Validator(SCHEMA)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_record(rec):
    """Schema validation plus the date-time rule (jsonschema does not check `format` by default)."""
    _VALIDATOR.validate(rec)
    ts = datetime.datetime.fromisoformat(rec["timestamp"])
    if ts.utcoffset() != datetime.timedelta(0):
        raise ValueError("timestamp must be UTC")
    return rec


def make_record(item_id, verdict, score, timestamp=None):
    rec = {"checkpoint_id": CHECKPOINT_ID, "item_id": str(item_id), "verdict": verdict,
           "score": round(float(score), 4), "timestamp": timestamp or utc_now()}
    return validate_record(rec)
