"""SYNTHETIC test: config folders nested under a wrapper folder (the dataset card shows both layouts, so we support both)."""
import shutil, subprocess, sys, tempfile, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(ROOT / "tests" / "make_synthetic_root.py"), str(tmp / "flat")], check=True, capture_output=True)
(tmp / "wrapped" / "finish_transfer_rank").mkdir(parents=True)
shutil.copy(tmp / "flat" / "SYNTHETIC_NOT_PADBENCH.txt", tmp / "wrapped" / "SYNTHETIC_NOT_PADBENCH.txt")
for d in (tmp / "flat").glob("synthetic_rank*_qv"):
    shutil.copytree(d, tmp / "wrapped" / "finish_transfer_rank" / d.name)
rules = tmp / "rules.json"
rules.write_text(json.dumps({"mode": "label_dir_map", "map": {"label0": "benign", "label1": "backdoor"},
                             "mapping_status": "synthetic_fixture",
                             "evidence": "SYNTHETIC UNIT TEST ONLY - nothing about PADBench"}), encoding="utf-8")
base = [sys.executable, "s1_build_manifest.py", "--source", "local", "--local-root", str(tmp / "wrapped"),
        "--data-tag", "synthetic_selftest", "--rules", str(rules), "--folders", "synthetic_rank8_qv", "synthetic_rank16_qv"]
r = subprocess.run(base + ["--work", str(tmp / "w")], cwd=ROOT, capture_output=True, text=True)
assert r.returncode == 0, r.stdout + r.stderr
import pandas as pd
m = pd.read_csv(tmp / "w" / "manifest.csv")
rep = json.loads((tmp / "w" / "manifest_report.json").read_text(encoding="utf-8"))
assert rep["root_prefix"] == "finish_transfer_rank", rep["root_prefix"]
assert m["item_id"].str.startswith("finish_transfer_rank/synthetic_rank").all()      # real paths kept for traceability
assert set(m["group"]) == {"synthetic_rank8_qv", "synthetic_rank16_qv"}                # short, prefix-free group names
assert set(m["label"]) == {"benign", "backdoor"}
r2 = subprocess.run(base[:-2] + ["no_such_folder", "--work", str(tmp / "w2")], cwd=ROOT, capture_output=True, text=True)
assert r2.returncode == 2 and "DISCOVERY FAILED" in r2.stderr and "actually there" in r2.stderr, r2.stderr
print("ok (PASS): wrapper folder auto-detected, item_ids keep the full repo path, missing folders refused")
