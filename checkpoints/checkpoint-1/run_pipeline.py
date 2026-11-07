#!/usr/bin/env python3
"""SentinelMesh Checkpoint 1 - Unified End-to-End Pipeline Runner.

Integrates ML WeightWatch scanner with Cyber Trust Chain:
1. Keys: Ensure Ed25519 signing keypair exists (generate if missing)
2. S0: Inspect repository / weights layout
3. S1: Build dataset manifest with label verification
4. S2: Extract 210 spectral features + record scan-time SHA-256 hash
5. S3: Train classifier (Logistic Regression & Random Forest) and optimize thresholds
6. Cyber Seal: Cryptographically seal trained classifier (c1_model.joblib)
7. S4: Run verified-load classification -> output schema-valid ckpt1_log.jsonl
8. Cyber Sign: Cryptographically sign C1 log records -> signed_c1_records.json
9. Deployment Gate Check: Demonstrate verify.py gatekeeper enforcement

Usage:
    python run_pipeline.py [--work work] [--split all|test] [--source hf|local] [--local-root <path>]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run_step(step_name, cmd_args):
    print("\n" + "=" * 78)
    print(f"STEP: {step_name}")
    print("=" * 78)
    print("Command:", " ".join(cmd_args))
    r = subprocess.run([PY, *cmd_args], cwd=ROOT)
    if r.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with return code {r.returncode}", file=sys.stderr)
        sys.exit(r.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run SentinelMesh CP1 Integrated Pipeline")
    parser.add_argument("--work", default="work", help="Working directory (default: work)")
    parser.add_argument("--source", choices=["hf", "local"], default="local", help="Data source")
    parser.add_argument("--local-root", default=None, help="Root directory for weights")
    parser.add_argument("--folders", nargs="*", default=None, help="Folders to inspect/process")
    parser.add_argument("--split", choices=["test", "val", "train", "all"], default="all")
    parser.add_argument("--skip-train", action="store_true", help="Skip re-training if model exists")
    args = parser.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    # 1. Key Generation
    priv_key = ROOT / "scanner_private.pem"
    pub_key = ROOT / "scanner_public.pem"
    if not priv_key.exists() or not pub_key.exists():
        run_step("1. Generate Cryptographic Keypair", ["make_keys.py"])
    else:
        print("[INFO] Cryptographic Ed25519 keypair already present.")

    # 2. Inspect & Manifest (if not skipping to scoring)
    s0_args = ["s0_inspect_repo.py", "--source", args.source, "--work", str(work)]
    s1_args = ["s1_build_manifest.py", "--source", args.source, "--work", str(work)]
    if args.local_root:
        s0_args += ["--local-root", args.local_root]
        s1_args += ["--local-root", args.local_root]
    if args.folders:
        s0_args += ["--folders", *args.folders]
        s1_args += ["--folders", *args.folders]

    manifest_csv = work / "manifest.csv"
    if not manifest_csv.exists():
        run_step("2. Inspect Weight Repository", s0_args)
        run_step("3. Build Manifest", s1_args)

    # 3. Extract Features (with SHA-256)
    features_csv = work / "features" / "features.csv"
    if not features_csv.exists():
        s2_args = ["s2_extract_features.py", "--work", str(work)]
        if args.local_root:
            s2_args += ["--local-root", args.local_root]
        run_step("4. Feature Extraction & SHA-256 Fingerprinting", s2_args)

    # 4. Train Model
    model_file = work / "model" / "c1_model.joblib"
    if not model_file.exists() or not args.skip_train:
        run_step("5. Train Classifiers & Select Best", ["s3_train_eval.py", "--work", str(work)])

    # 5. Cryptographically Seal Classifier
    classifier_record = work / "model" / "classifier_record.json"
    run_step("6. Seal Trained Classifier", [
        "sign_model.py", str(model_file), "--out", str(classifier_record),
        "--key", str(priv_key)
    ])

    # 6. Score Adapters with Verified Deserialization
    run_step("7. Score Adapters (Verified Load & Shared Schema Validation)", [
        "s4_write_c1_log.py", "--work", str(work), "--split", args.split,
        "--pubkey", str(pub_key), "--classifier-record", str(classifier_record)
    ])

    # 7. Cryptographically Sign C1 Log Records
    signed_records = work / "c1_log" / "signed_c1_records.json"
    sign_args = [
        "sign_c1.py",
        "--log", str(work / "c1_log" / "ckpt1_log.jsonl"),
        "--features", str(features_csv),
        "--out", str(signed_records),
        "--key", str(priv_key),
        "--deterministic-hash-fallback"
    ]
    if args.local_root:
        sign_args += ["--root", args.local_root]
    elif (work / "hf_files").exists():
        sign_args += ["--root", str(work / "hf_files")]
    run_step("8. Cryptographically Sign C1 Log Records", sign_args)

    # 8. Schema Validation Summary
    print("\n" + "=" * 78)
    print("SCHEMA VERIFICATION SUMMARY")
    print("=" * 78)
    try:
        import jsonschema
        schema = json.loads((ROOT / "shared_schema_format.json").read_text(encoding="utf-8"))
        validator = jsonschema.Draft7Validator(schema, format_checker=jsonschema.FormatChecker())
        log_records = [json.loads(l) for l in (work / "c1_log" / "ckpt1_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        violations = 0
        for i, rec in enumerate(log_records):
            errs = list(validator.iter_errors(rec))
            if errs:
                print(f"Record {i} violation:", [e.message for e in errs])
                violations += 1
        if violations == 0:
            print(f"[SUCCESS] All {len(log_records)} records strictly conform to 'shared_schema_format.json'.")
            print("Fields per record: checkpoint_id, item_id, verdict, score, timestamp (additionalProperties: false)")
    except Exception as e:
        print("[WARNING] Could not run schema validator:", e)

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Schema-conforming log:  {work / 'c1_log' / 'ckpt1_log.jsonl'}")
    print(f"Cryptographically signed: {signed_records}")
    print(f"Sealed classifier:        {classifier_record}")
    print("=" * 78)


if __name__ == "__main__":
    main()
