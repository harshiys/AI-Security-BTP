#!/usr/bin/env python3
"""Cross-platform (Windows/macOS/Linux) mechanics test on SYNTHETIC (fake) adapters. NOT PADBench, NOT a result.

    python tests/run_selftest.py

Output goes to work_selftest/ and is tagged data_tag=synthetic_selftest. Real runs use the work/ folder instead."""
import json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "work_selftest"
PY = sys.executable
import time
if W.exists():
    shutil.rmtree(W, ignore_errors=True)
    for _ in range(10):
        if not W.exists(): break
        time.sleep(0.1)
W.mkdir(parents=True, exist_ok=True)


def run(args, expect_fail=False, show=True):
    r = subprocess.run([PY, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if expect_fail != (r.returncode != 0):
        print(out); sys.exit(f"UNEXPECTED {'success' if expect_fail else 'failure'} (exit {r.returncode}): {' '.join(map(str, args))}")
    if show:
        print(out)
    return out


def header(t): print(f"\n######## {t}")


header("unit tests")
run(["tests/test_core.py"])
for t in ("test_hf_source_stub.py", "test_prefix_layout.py"):
    print(run([f"tests/{t}"], show=False).splitlines()[-1])
print(run(["tests/test_discovery.py"], show=False))
run(["tests/make_synthetic_root.py", W / "synthetic_root"], show=False)
(W / "label_rules.json").write_text(json.dumps({
    "mode": "label_dir_map", "dir_pattern": r"^label\d+$", "map": {"label0": "benign", "label1": "backdoor"},
    "mapping_status": "synthetic_fixture",
    "evidence": "SYNTHETIC SELF-TEST ONLY: tests/make_synthetic_root.py writes its own fake benign adapters to label0. "
                "This says nothing about PADBench."}), encoding="utf-8")
FOLDERS = [f"synthetic_rank{r}_qv" for r in (4, 8, 16, 32, 64)]
SRC = ["--source", "local", "--local-root", W / "synthetic_root", "--work", W]

header("S0 (synthetic)")
s0 = run(["s0_inspect_repo.py", *SRC, "--folders", *FOLDERS], show=False)
print("\n".join(l[:160] for l in s0.splitlines() if l.startswith(("numeric label", "Found label", "Wrote a TEMPLATE", "== weight file names"))))
header("S1 must REFUSE the unfilled template")
print(run(["s1_build_manifest.py", *SRC, "--folders", *FOLDERS, "--rules", W / "label_rules.proposed.json"], expect_fail=True, show=False).splitlines()[-1][:200])
header("S1 (synthetic, explicit mapping)")
run(["s1_build_manifest.py", *SRC, "--data-tag", "synthetic_selftest", "--folders", *FOLDERS, "--max-per-class-per-folder", 14])
header("S2 probe + extract (synthetic)")
print("\n".join(run(["s2_extract_features.py", *SRC, "--probe"], show=False).splitlines()[:8]))
print(run(["s2_extract_features.py", *SRC], show=False).splitlines()[-1])
header("S3 (synthetic)")
run(["s3_train_eval.py", "--work", W, "--test-groups", "synthetic_rank16_qv", "--val-groups", "synthetic_rank32_qv"])
header("Cyber: seal model")
run(["sign_model.py", W / "model" / "c1_model.joblib", "--out", W / "model" / "classifier_record.json"])
header("S4 (synthetic)")
try:
    import jsonschema  # noqa: F401
    extra = []
except ImportError:
    extra = ["--allow-fallback-validator"]
run(["s4_write_c1_log.py", "--work", W, *extra])
header("Cyber: sign C1 records")
run(["sign_c1.py", "--log", W / "c1_log" / "ckpt1_log.jsonl", "--features", W / "features" / "features.csv", "--root", W / "synthetic_root", "--out", W / "c1_log" / "signed_c1_records.json"])

# ---- scenario 2: ONE config folder, no --folders, default local root, project-default label_config.json ----
header("SCENARIO 2 (synthetic): single config folder, default discovery, provisional label_config.json")
W2 = ROOT / "work_selftest_single"
if W2.exists():
    shutil.rmtree(W2, ignore_errors=True)
    for _ in range(10):
        if not W2.exists(): break
        time.sleep(0.1)
(W2 / "hf_files").mkdir(parents=True, exist_ok=True)
run(["tests/make_synthetic_root.py", W2 / "hf_files", 40, 64], show=False)
W2 = str(W2)
print(run(["s1_build_manifest.py", "--source", "local", "--work", W2], show=False).split("per-folder")[0].strip())
print(run(["s2_extract_features.py", "--work", W2], show=False).splitlines()[-1])
s3 = run(["s3_train_eval.py", "--work", W2], show=False)
print("\n".join(l for l in s3.splitlines() if l.startswith(("!!", "validation AUROC", "chosen", "split assignment", "=== TEST"))
                or "LIMITATIONS" in l or l.strip().startswith("- ")))
run(["sign_model.py", Path(W2) / "model" / "c1_model.joblib", "--out", Path(W2) / "model" / "classifier_record.json"], show=False)
run(["s4_write_c1_log.py", "--work", W2, *extra], show=False)
run(["sign_c1.py", "--log", Path(W2) / "c1_log" / "ckpt1_log.jsonl", "--features", Path(W2) / "features" / "features.csv", "--root", Path(W2) / "hf_files", "--out", Path(W2) / "c1_log" / "signed_c1_records.json"], show=False)
ri = json.loads((Path(W2) / "c1_log" / "run_info.json").read_text(encoding="utf-8"))
print("run_info:", {k: ri[k] for k in ("run_type", "data_tag", "label_mapping_status", "split_mode", "n_records")})
print("\nSELF-TEST FINISHED OK (synthetic data - proves the code runs, says nothing about PADBench).")
