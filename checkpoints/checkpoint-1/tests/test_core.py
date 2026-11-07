"""Mechanics tests (numpy only). Run: python tests/test_core.py"""
import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from c1.features import pair_features, feature_names, extract_adapter
from c1.safeload import write_safetensors, read_safetensors
from c1.model_utils import choose_thresholds, verdict_from_score, wilson
from c1.labels import LabelError, LabelResolver, load_rules

rng = np.random.default_rng(0)

# 1. QR trick == explicit SVD of B@A (plain LoRA, and AdaLoRA with E)
A = rng.normal(size=(8, 64)).astype(np.float32); B = rng.normal(size=(48, 8)).astype(np.float32)
E = rng.uniform(0.1, 2, size=(8, 1)).astype(np.float32)
for e_ in (None, E):
    Aeff = A if e_ is None else A * e_
    s = np.linalg.svd(B.astype(np.float64) @ Aeff.astype(np.float64), compute_uv=False)[:8]
    p = s**2 / (s**2).sum()
    f = pair_features(A, B, e_, scale=1.0)
    assert abs(f["top1_energy"] - p[0]) < 1e-4, (f["top1_energy"], p[0])
    assert abs(10**f["log_fro"] - np.sqrt((s**2).sum())) / np.sqrt((s**2).sum()) < 1e-4
print("ok: QR-SVD matches explicit SVD (with and without AdaLoRA E)")

# 2. B == 0 (untrained adapter) must not crash
f = pair_features(A, np.zeros_like(B)); assert f["top1_energy"] == 0.0
print("ok: degenerate B=0 handled")

# 3. safetensors reader round-trip, F32 and BF16, and header sanity
with tempfile.TemporaryDirectory() as d:
    t = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": A,
         "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": B,
         "other.weight": rng.normal(size=(3, 3)).astype(np.float32)}
    write_safetensors(Path(d) / "a.safetensors", t)
    got = read_safetensors(Path(d) / "a.safetensors")
    assert set(got) == {k for k in t if "lora_" in k}
    assert np.array_equal(got[list(got)[0]], t[list(got)[0]])
    write_safetensors(Path(d) / "b.safetensors", t, bf16=True)
    gb = read_safetensors(Path(d) / "b.safetensors")
    k0 = list(gb)[0]; assert np.allclose(gb[k0], t[k0], rtol=1e-2, atol=1e-2)
    (Path(d) / "bad.safetensors").write_bytes(b"\xff" * 64)
    try:
        read_safetensors(Path(d) / "bad.safetensors"); raise SystemExit("bad header accepted")
    except Exception as e:
        assert "safetensors" in str(e) or "header" in str(e), e
    meta, feats, rows = extract_adapter(Path(d) / "a.safetensors")
    assert set(feats) == set(feature_names()) and meta["n_pairs_q"] == 1 and meta["n_pairs_v"] == 0
print("ok: safetensors read (F32/BF16), malformed header rejected, extract_adapter shape OK")

# 4. thresholds: block threshold keeps val FPR <= target; flag threshold keeps recall >= target
y = np.array([0]*200 + [1]*100); s = np.concatenate([rng.beta(2, 8, 200), rng.beta(6, 3, 100)])
th = choose_thresholds(y, s, fpr_block=0.01, recall_flag=0.95)
assert (s[y == 0] >= th["t_block"]).mean() <= 0.01 and (s[y == 1] >= th["t_flag"]).mean() >= 0.95
assert th["t_flag"] <= th["t_block"]
assert verdict_from_score(0.0, th) == "pass" and verdict_from_score(1.0, th) == "block"
print("ok: thresholds satisfy validation FPR / recall targets", {k: round(v, 3) for k, v in th.items() if k in ("t_flag", "t_block")})

# 5. label_dir_map: fails closed unless the mapping AND its evidence are explicitly provided
import tempfile as _tf
def _rules(obj):
    d = _tf.mkdtemp(); p = Path(d) / "r.json"; p.write_text(json.dumps(obj), encoding="utf-8"); return p
def _expect_fail(obj, frag):
    try:
        load_rules(_rules(obj))
    except LabelError as e:
        assert frag in str(e), (frag, str(e)); return
    raise SystemExit(f"rules accepted but should fail ({frag}): {obj}")
EV = "SYNTHETIC UNIT TEST ONLY - not a statement about PADBench"
MS = "synthetic_fixture"
_expect_fail({"mode": "label_dir_map", "map": {"label0": None, "label1": None}, "evidence": EV, "mapping_status": MS}, "not set to")
_expect_fail({"mode": "label_dir_map", "map": {"label0": "benign", "label1": "benign"}, "evidence": EV, "mapping_status": MS}, "BOTH classes")
_expect_fail({"mode": "label_dir_map", "map": {"label0": "benign", "label1": "backdoor"}, "evidence": "", "mapping_status": MS}, "evidence")
_expect_fail({"status": "PROPOSED/UNVERIFIED", "mode": "label_dir_map", "map": {"label0": "benign", "label1": "backdoor"}, "evidence": EV, "mapping_status": MS}, "PROPOSED")
_expect_fail({"mode": "label_dir_map", "map": {"label0": "benign", "label1": "backdoor"}, "evidence": EV}, "mapping_status")
_expect_fail({"mode": "label_dir_map", "map": {"label0": "benign", "label1": "backdoor"}, "evidence": EV, "mapping_status": "sure"}, "mapping_status")
spec = load_rules(_rules({"mode": "label_dir_map", "map": {"label0": "benign", "label1": "backdoor"}, "evidence": EV, "mapping_status": MS}))
res = LabelResolver(spec)
assert res.label("cfg/label0/a1/adapter_model.safetensors", "cfg") == ("benign", "label_dir:label0")
assert res.label("cfg/label1/a1/adapter_model.safetensors", "cfg") == ("backdoor", "label_dir:label1")
assert res.label("cfg/label2/a1/adapter_model.safetensors", "cfg")[0] is None          # unmapped folder -> unlabeled
assert res.label("cfg/a1/adapter_model.safetensors", "cfg")[0] is None                  # no label folder -> unlabeled
assert res.label("cfg/label0/label1/x/adapter_model.safetensors", "cfg") == (None, "conflict")
assert res.label("cfg/label1_0007.safetensors", "cfg") == ("backdoor", "label_file:label1")   # label in the file name
assert res.label("cfg/x/relabel0.safetensors", "cfg")[0] is None                                # not a label token
print("ok: label_dir_map refuses null/one-sided/evidence-less/PROPOSED rules; labels only via explicit mapping")

# 6. Wilson interval sanity
lo, hi = wilson(8, 10); assert 0.4 < lo < 0.5 and 0.9 < hi < 1.0, (lo, hi)
print("ok: Wilson CI", (round(lo, 3), round(hi, 3)))

# 7. shipped default mapping file is valid, provisional, and exactly the project convention (editable in one place)
cfgp = Path(__file__).resolve().parents[1] / "label_config.json"
spec = load_rules(cfgp)
assert spec["map"] == {"label0": "benign", "label1": "backdoor"} and spec["mapping_status"] == "provisional"
print("ok: label_config.json = provisional project convention (label0=benign, label1=backdoor)")

# 8. path helpers
from c1.common import norm_name
assert norm_name('"C:\\data\\padbench\\llama2_rank64_qv\\"') == "C:/data/padbench/llama2_rank64_qv"
print("ok: Windows-style paths are normalised")

# 9. absolute roots keep their leading slash; a root that IS a config folder is lifted to its parent
from c1.common import effective_local_root
assert effective_local_root("/abs/data/padbench", ["x"]) == "/abs/data/padbench"
assert effective_local_root('"/abs/data/llama2_rank64_qv"', ["llama2_rank64_qv"]) == "/abs/data"
assert effective_local_root(None, None, "work").replace("\\", "/") == "work/hf_files"
print("ok: local-root handling (absolute paths kept, folder-as-root lifted, default work/hf_files)")
