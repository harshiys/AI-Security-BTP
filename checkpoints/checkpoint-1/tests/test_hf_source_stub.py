"""STUB test of c1.common.HFSource - NOT a real Hugging Face test.

It replaces `huggingface_hub` with a tiny fake backed by the synthetic fixture, to catch typos/wrong arguments in
our own wrapper code (revision pinning, listing, fetch, release, error mapping). It cannot prove the real API
behaves the same; only a run with network access can (see README 'real-data smoke test')."""
import sys, types, shutil, tempfile, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

tmp = Path(tempfile.mkdtemp()); fake = tmp / "fake_repo"
subprocess.run([sys.executable, str(ROOT / "tests" / "make_synthetic_root.py"), str(fake)], check=True, capture_output=True)

class RepoFile:
    def __init__(s, path, size): s.path, s.size = path, size
class RepoFolder:
    def __init__(s, path): s.path = path
calls = {}
class GatedRepoError(Exception): pass
class HfApi:
    def dataset_info(self, repo_id, revision=None): return types.SimpleNamespace(sha="abc123deadbeef")
    def list_repo_tree(self, repo_id, path_in_repo=None, *, recursive=False, revision=None, repo_type=None):
        calls["revision"] = revision; calls["repo_type"] = repo_type
        base = fake / (path_in_repo or "")
        if recursive:
            for p in sorted(base.rglob("*")):
                yield RepoFile(p.relative_to(fake).as_posix(), p.stat().st_size) if p.is_file() else RepoFolder(p.relative_to(fake).as_posix())
        else:
            for p in sorted(base.iterdir()):
                yield RepoFile(p.relative_to(fake).as_posix(), p.stat().st_size) if p.is_file() else RepoFolder(p.relative_to(fake).as_posix())
def hf_hub_download(repo_id, filename, *, repo_type=None, revision=None, local_dir=None):
    calls["dl_revision"] = revision
    dst = Path(local_dir) / filename; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(fake / filename, dst); return str(dst)

hub = types.ModuleType("huggingface_hub"); hub.HfApi, hub.hf_hub_download = HfApi, hf_hub_download
api = types.ModuleType("huggingface_hub.hf_api"); api.RepoFile, api.RepoFolder = RepoFile, RepoFolder
sys.modules["huggingface_hub"], sys.modules["huggingface_hub.hf_api"] = hub, api

from c1.common import HFSource, hf_errors
src = HFSource(tmp / "work")
assert src.revision == "abc123deadbeef", src.revision
folders = list(src.iter_folders(""))
assert "synthetic_rank8_qv" in folders, folders
files = list(src.iter_files("synthetic_rank8_qv", recursive=True))
w = [f for f in files if f.path.endswith("adapter_model.safetensors")]
assert len(w) == 28 and calls["revision"] == "abc123deadbeef" and calls["repo_type"] == "dataset"
local = src.fetch(w[0].path); assert local.exists() and calls["dl_revision"] == "abc123deadbeef"
src.release(w[0].path); assert not local.exists()
pinned = HFSource(tmp / "w2", revision="v-pinned"); assert pinned.revision == "v-pinned"

try:
    with hf_errors(): raise GatedRepoError("401 Client Error: gated")
except SystemExit as e:
    assert e.code == 3
else:
    raise SystemExit("auth error was not mapped to a clear exit")
class EntryNotFoundError(Exception): pass
try:
    with hf_errors(): raise EntryNotFoundError("missing")
except EntryNotFoundError:
    pass  # per-file errors must propagate so stage 2 can record them and continue
print("ok (STUB): HFSource pins revision, lists/fetches/releases, maps auth/network errors; per-file errors propagate")
