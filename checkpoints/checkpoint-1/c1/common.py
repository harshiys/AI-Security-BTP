"""Shared helpers: data sources (HF repo or local dir), file classification."""
from __future__ import annotations

import fnmatch
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ID = "Vincent-HKUSTGZ/PADBench"
REPO_TYPE = "dataset"
CHECKPOINT_ID = "CKPT1_STATIC"          # fixed by the shared schema for Day 1
WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth")
SKIP_NAME_PARTS = ("optimizer", "scheduler", "rng_state", "training_args", "trainer_state")
META_EXTS = (".json", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml")
# Default first-run subset: small-rank Llama2-7B toxic_backdoors_hard folders named in the
# dataset card. Small ranks => small files (rank 8 is ~tens of MB), so the first run is cheap.
DEFAULT_FOLDERS = [f"llama2_7b_toxic_backdoors_hard_rank{r}_qv" for r in (8, 16, 32, 64, 128)]


# data_tag marks where a run's data came from. It is stored in the manifest and printed by s3/s4 so that
# synthetic self-test output can never be mistaken for real PADBench output.
TAG_REAL = "padbench_hf"            # real PADBench listed/downloaded from Hugging Face
TAG_LOCAL_REAL = "padbench_local"   # real PADBench files already on disk (e.g. work/hf_files/...)
TAG_SYNTHETIC = "synthetic_selftest"
TAG_LOCAL = TAG_LOCAL_REAL          # default tag for --source local (kept for older imports)
REAL_TAGS = (TAG_REAL, TAG_LOCAL_REAL)
SYNTHETIC_MARKER = "SYNTHETIC_NOT_PADBENCH.txt"   # written into every fake fixture folder


def is_real_tag(tag: str) -> bool:
    return tag in REAL_TAGS


def norm_name(n: str) -> str:
    """Folder names/paths typed on Windows (backslashes, quotes, trailing slash) -> clean posix form."""
    return str(n).strip().strip("\"'").replace("\\", "/").strip("/")


@contextmanager
def hf_errors():
    """Turn Hugging Face / network failures into one clear message instead of a traceback."""
    try:
        yield
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        hint = None
        status = getattr(getattr(e, "response", None), "status_code", None)
        if name == "GatedRepoError" or status in (401, 403) or "401 Client Error" in str(e) or "403 Client Error" in str(e):
            hint = ("Hugging Face refused access (authentication/gated repo). Set the HF_TOKEN environment variable "
                    "(or run: python -c \"from huggingface_hub import login; login()\"), then retry.")
        elif name in ("RepositoryNotFoundError", "RevisionNotFoundError"):
            hint = "Repository or revision not found. Check the repo id / --revision."
        elif name in ("EntryNotFoundError", "RemoteEntryNotFoundError", "FileNotFoundError"):
            raise  # per-file problem: stage 2 records it for that adapter and continues
        elif name in ("LocalEntryNotFoundError", "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout",
                      "NameResolutionError", "MaxRetryError", "ProxyError", "SSLError", "OfflineModeIsEnabled"):
            hint = "Network problem reaching huggingface.co. Check your connection/proxy and retry (runs resume)."
        if hint is None:
            raise
        print(f"\nHUGGING FACE ERROR ({name}): {hint}\n  details: {str(e)[:300]}", file=sys.stderr)
        sys.exit(3)


@dataclass(frozen=True)
class FileEntry:
    path: str   # repo-relative, posix separators
    size: int


def is_weight_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    return name.endswith(WEIGHT_EXTS) and not any(p in name for p in SKIP_NAME_PARTS)


def weight_name_matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path.rsplit("/", 1)[-1].lower(), pattern.lower())


def is_meta_file(path: str) -> bool:
    return path.lower().endswith(META_EXTS)


class LocalSource:
    """Directory on disk laid out like the repo (used for the self-test)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def iter_files(self, subdir: str = "", recursive: bool = True) -> Iterator[FileEntry]:
        base = self.root / subdir if subdir else self.root
        if not base.exists():
            return
        for dp, dns, fns in os.walk(base):
            dns.sort()
            for fn in sorted(fns):
                p = Path(dp) / fn
                yield FileEntry(p.relative_to(self.root).as_posix(), p.stat().st_size)
            if not recursive:
                break

    def iter_folders(self, subdir: str = "") -> Iterator[str]:
        base = self.root / subdir if subdir else self.root
        try:
            kids = sorted(base.iterdir())
        except OSError:  # permission problem / vanished folder: skip it, do not crash discovery
            return
        for d in kids:
            if d.is_dir():
                yield d.relative_to(self.root).as_posix()

    def is_synthetic(self) -> bool:
        """True if this tree (or a parent within 3 levels) carries the synthetic-fixture marker."""
        p = self.root.resolve()
        return any((q / SYNTHETIC_MARKER).exists() for q in [p, *list(p.parents)[:3]])

    def fetch(self, path: str) -> Path:
        return self.root / path

    def release(self, path: str) -> None:  # never delete local source files
        return None


class HFSource:
    """Lazy view of the PADBench dataset repo. Nothing is downloaded until fetch().

    `revision` pins every listing/download to one commit so a run is reproducible; when omitted, the current
    commit sha of the repo is resolved once and exposed as `.revision` (stage 1 records it in the manifest report).
    """

    def __init__(self, work: Path, repo_id: str = REPO_ID, revision: str | None = None):
        try:
            from huggingface_hub import HfApi  # imported lazily
        except ImportError:
            raise SystemExit("huggingface_hub is not installed: pip install -r requirements.txt")

        self.api = HfApi()
        self.repo_id = repo_id
        self.cache_dir = Path(work) / "hf_files"
        self.max_search_depth = 3   # root -> wrapper folder -> config folder
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with hf_errors():
            self.revision = revision or self.api.dataset_info(repo_id).sha

    def iter_files(self, subdir: str = "", recursive: bool = True) -> Iterator[FileEntry]:
        from huggingface_hub.hf_api import RepoFile

        with hf_errors():
            for e in self.api.list_repo_tree(self.repo_id, path_in_repo=subdir or None, recursive=recursive,
                                             repo_type=REPO_TYPE, revision=self.revision):
                if isinstance(e, RepoFile):
                    yield FileEntry(e.path, int(e.size or 0))

    def iter_folders(self, subdir: str = "") -> Iterator[str]:
        from huggingface_hub.hf_api import RepoFolder

        with hf_errors():
            for e in self.api.list_repo_tree(self.repo_id, path_in_repo=subdir or None, recursive=False,
                                             repo_type=REPO_TYPE, revision=self.revision):
                if isinstance(e, RepoFolder):
                    yield e.path

    def fetch(self, path: str) -> Path:
        from huggingface_hub import hf_hub_download

        with hf_errors():
            return Path(hf_hub_download(self.repo_id, filename=path, repo_type=REPO_TYPE,
                                        revision=self.revision, local_dir=str(self.cache_dir)))

    def release(self, path: str) -> None:
        p = self.cache_dir / path
        if p.exists():
            p.unlink()


class FolderResolutionError(Exception):
    pass


_SKIP_DIRS = {".git", ".cache", "__pycache__", ".venv", "venv", "node_modules", "work_selftest"}
_LABEL_DIR = re.compile(r"^label\d+$", re.I)


def _kids(src, path):
    try:
        return list(src.iter_folders(path))
    except OSError:
        return []


def find_config_folders(src, names, max_depth=None):
    """Breadth-first search for directories whose NAME equals a requested config folder (case-insensitive).

    Works wherever the folders sit: repo root, under finish_transfer_rank/, under work/hf_files/, or several levels
    deeper than --local-root. It never descends into label folders or caches. Returns {name: [candidate paths]}.
    """
    depth_cap = max_depth if max_depth is not None else getattr(src, "max_search_depth", 6)
    wanted = {n.lower(): n for n in names}
    found: dict[str, list[str]] = {}
    level, visited = [""], 0
    for _ in range(depth_cap):
        nxt = []
        for cur in level:
            for k in _kids(src, cur):
                base = k.rsplit("/", 1)[-1]
                if base.lower() in wanted:
                    found.setdefault(wanted[base.lower()], []).append(k)
                    continue                      # found it; no need to search inside
                if base in _SKIP_DIRS or _LABEL_DIR.match(base):
                    continue
                visited += 1
                if visited > 20000:
                    return found
                nxt.append(k)
        if all(n in found for n in names) or not nxt:
            break
        level = nxt
    return found


def resolve_folders(src, names, prefix=None):
    """Locate the requested config folders. Returns (prefix, {name: full_path}, missing_names).

    `prefix` (optional --root-prefix) restricts the search to candidates whose parent folder equals it. Raises
    FolderResolutionError if a name is found in several places (ambiguous) - never picks silently.
    """
    names = [norm_name(n) for n in names]
    found = find_config_folders(src, names)
    full, missing = {}, []
    for n in names:
        cands = found.get(n, [])
        if prefix is not None:
            pre = norm_name(prefix)
            cands = [c for c in cands if c.rsplit("/", 1)[0] == pre or (not pre and "/" not in c)]
        if not cands:
            missing.append(n)
        elif len(cands) > 1:
            raise FolderResolutionError(f"folder '{n}' exists in several places {cands[:4]}; "
                                        f"pass --root-prefix to choose one")
        else:
            full[n] = cands[0]
    first = next(iter(full.values()), "")
    return (first.rsplit("/", 1)[0] if "/" in first else ""), full, missing


def discover_config_folders(src, weight_pattern="*.safetensors"):
    """Local-only: find config folders by looking at where the weight files actually are.

    Config folder = the folder just above the first label-like folder (label0/label1/...) on the way to a weight file;
    if there is no label folder, the grandparent of the weight file. Returns {name: full_path}; raises on name clashes.
    """
    out: dict[str, str] = {}
    for e in src.iter_files("", recursive=True):
        if not is_weight_file(e.path) or not weight_name_matches(e.path, weight_pattern):
            continue
        comps = e.path.split("/")
        if any(c in _SKIP_DIRS for c in comps[:-1]):
            continue
        idx = next((k for k, c in enumerate(comps[:-1]) if _LABEL_DIR.match(c)), None)
        if idx is not None and idx > 0:
            cfg = comps[:idx]
        elif len(comps) >= 3:
            cfg = comps[:-2]
        else:
            continue
        name, path = cfg[-1], "/".join(cfg)
        if name in out and out[name] != path:
            raise FolderResolutionError(f"config folder name '{name}' found at both '{out[name]}' and '{path}'")
        out[name] = path
    return out


def describe_tree(src, limit=12):
    """Human-readable 'what is actually here' for error messages."""
    lines, seen = [], 0
    for e in src.iter_files("", recursive=True):
        if is_weight_file(e.path):
            seen += 1
            if seen <= limit:
                lines.append(f"    {e.path}  ({e.size} bytes)")
        if seen > 50000:
            break
    top = _kids(src, "")[:15]
    head = f"  top-level folders: {top}\n  weight files found: {seen}"
    return head + ("\n" + "\n".join(lines) if lines else "")


def effective_local_root(local_root, names=None, work="work"):
    """Tidy --local-root: default to <work>/hf_files (where stage 2 stores downloads); strip quotes; if the path you
    gave IS one of the config folders, step up one level so folder names resolve."""
    # only trim whitespace/quotes here: a root may legitimately start with "/" (macOS/Linux) or "C:\\" (Windows)
    root = str(local_root).strip().strip("\"'") if local_root else str(Path(work) / "hf_files")
    root = os.path.expanduser(root)
    p = Path(root)
    if names and p.name.lower() in {norm_name(n).lower() for n in names} and p.parent != p:
        return p.parent.as_posix()
    return p.as_posix()


def make_source(kind: str, local_root: str | None, work: str | Path, revision: str | None = None, names=None):
    if kind == "local":
        src = LocalSource(effective_local_root(local_root, names, work))
        if not src.root.is_dir():
            raise SystemExit(f"local root not found: {src.root}  (pass --local-root <folder that contains the PADBench "
                             f"config folders>; default is {Path(work) / 'hf_files'})")
        src.revision = None
        return src
    return HFSource(Path(work), revision=revision)
