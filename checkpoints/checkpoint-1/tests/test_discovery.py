"""SYNTHETIC test of dataset discovery (stage 1) - reproduces layouts that previously failed with
"folders not found under '/'". All adapters are fake; the label0/label1 folders follow the project's provisional convention
(label_config.json) and say nothing about real PADBench."""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = Path(tempfile.mkdtemp())
CFG = "synthetic_rank64_qv"
subprocess.run([sys.executable, str(ROOT / "tests" / "make_synthetic_root.py"), str(T / "base"), "12", "64"],
               check=True, capture_output=True)
MARK = T / "base" / "SYNTHETIC_NOT_PADBENCH.txt"


def s1(*a, expect=0):
    r = subprocess.run([sys.executable, "s1_build_manifest.py", *map(str, a)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == expect, f"exit {r.returncode} (wanted {expect})\n{r.stdout}\n{r.stderr}"
    return r


def man(work):
    return pd.read_csv(Path(work) / "manifest.csv"), json.loads((Path(work) / "manifest_report.json").read_text(encoding="utf-8"))


def check(m, rep, tag="synthetic_selftest"):
    assert set(m.label) == {"benign", "backdoor"} and set(m.group) == {CFG}, (set(m.label), set(m.group))
    assert m.item_id.str.endswith(".safetensors").all(), "decoy CSV leaked into the manifest"
    assert (m.data_tag == tag).all() and (m.label_mapping_status == "provisional").all()
    assert rep["mapping_status"] == "provisional" and rep["label_dir_map"] == {"label0": "benign", "label1": "backdoor"}
    b = m[m.label == "benign"].item_id.str.contains("/label0/").all()
    k = m[m.label == "backdoor"].item_id.str.contains("/label1/").all()
    assert b and k, "label0 must be benign and label1 backdoor under the provisional convention"


# A. default: local source, NO --local-root, NO --folders; files live in <work>/hf_files/<config>/label*/...
wA = T / "A_work"; (wA / "hf_files").mkdir(parents=True)
shutil.copytree(T / "base" / CFG, wA / "hf_files" / CFG); shutil.copy(MARK, wA / "hf_files" / MARK.name)
s1("--source", "local", "--work", wA); check(*man(wA))
print("ok (PASS) A: default local root <work>/hf_files, folders auto-discovered")

# B. the error case: --local-root is the PROJECT folder, data is 3 levels down (work/hf_files/<config>)
pB = T / "B_proj"; (pB / "work" / "hf_files").mkdir(parents=True)
shutil.copytree(T / "base" / CFG, pB / "work" / "hf_files" / CFG); shutil.copy(MARK, pB / MARK.name)
(pB / ".venv").mkdir(); (pB / "c1").mkdir()
s1("--source", "local", "--local-root", pB, "--folders", CFG, "--work", T / "B_work"); m, rep = man(T / "B_work"); check(m, rep)
assert m.item_id.iloc[0].startswith("work/hf_files/" + CFG), m.item_id.iloc[0]
print("ok (PASS) B: config folder found 3 levels below --local-root (this layout used to fail)")

# C. --local-root points AT the config folder itself
s1("--source", "local", "--local-root", T / "base" / CFG, "--folders", CFG, "--work", T / "C_work"); check(*man(T / "C_work"))
print("ok (PASS) C: --local-root is the config folder itself")

# D. Windows-style folder argument (backslash, quotes)
s1("--source", "local", "--local-root", T / "base", "--folders", f'"{CFG}\\"', "--work", T / "D_work"); check(*man(T / "D_work"))
print("ok (PASS) D: backslash/quoted folder name accepted")

# E. one adapter is a Git-LFS pointer (download not completed): skipped and reported, not parsed
shutil.copytree(T / "base", T / "E_root")
victim = sorted((T / "E_root" / CFG / "label0").glob("*/adapter_model.safetensors"))[0]
victim.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 12345\n", encoding="utf-8")
s1("--source", "local", "--local-root", T / "E_root", "--folders", CFG, "--work", T / "E_work"); m, rep = man(T / "E_work")
victim_rel = victim.relative_to(T / "E_root").as_posix()
assert rep["skipped_incomplete_files"] == 1, rep["skipped_incomplete_files"]
assert victim_rel not in set(m.item_id) and (m.label == "benign").sum() == 11, (victim_rel, m.label.value_counts().to_dict())
print("ok (PASS) E: Git-LFS pointer file skipped and reported")

# F. folder really missing -> clear message that shows what IS there; nothing guessed
r = s1("--source", "local", "--local-root", T / "base", "--folders", "llama2_7b_toxic_backdoors_hard_rank64_qv",
       "--work", T / "F_work", expect=2)
assert "DISCOVERY FAILED" in r.stderr and CFG not in r.stderr.split("What is actually there")[0] and "weight files found" in r.stderr
assert not (T / "F_work" / "manifest.csv").exists()
print("ok (PASS) F: missing folder -> honest error listing what exists, no manifest written")

# G. only one class present -> refused with counts, no manifest
shutil.copytree(T / "base", T / "G_root"); shutil.rmtree(T / "G_root" / CFG / "label1")
r = s1("--source", "local", "--local-root", T / "G_root", "--folders", CFG, "--work", T / "G_work", expect=2)
assert "backdoor" in r.stderr and "counts found" in r.stderr and not (T / "G_work" / "manifest.csv").exists()
print("ok (PASS) G: single-class data refused with counts")

# H. provenance guards: marked fixture cannot be tagged real; unmarked folder cannot be tagged synthetic
r = s1("--source", "local", "--local-root", T / "base", "--folders", CFG, "--work", T / "H_work", "--data-tag", "padbench_local", expect=1)
assert "synthetic-fixture marker" in (r.stderr + r.stdout)
unmarked = T / "I_root"; shutil.copytree(T / "base", unmarked); (unmarked / MARK.name).unlink()
r = s1("--source", "local", "--local-root", unmarked, "--folders", CFG, "--work", T / "I_work", "--data-tag", "synthetic_selftest", expect=1)
assert "no synthetic marker" in (r.stderr + r.stdout)
s1("--source", "local", "--local-root", unmarked, "--folders", CFG, "--work", T / "J_work")      # default tag for unmarked local files
assert set(man(T / "J_work")[0].data_tag) == {"padbench_local"}
print("ok (PASS) H: tag guards (marked fixture can't be 'real'; unmarked files default to 'padbench_local')")
