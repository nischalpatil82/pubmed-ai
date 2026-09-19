"""Dataset selection and atomic manifests. No database or process-global switching."""
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "covid-files" / "dataset.json"
_PINNED = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def space_guard(path, reserve_gb=2):
    free = shutil.disk_usage(path).free
    if free < reserve_gb * 1e9:
        raise RuntimeError(f"Build paused: {free / 1e9:.2f} GB free; reserve is {reserve_gb} GB. Resume after freeing space.")


def paths():
    if _PINNED is not None:
        return _PINNED
    explicit = os.environ.get("PUBMED_DATASET")
    # Legacy paths require explicit opt-in; a missing new snapshot never loads old data.
    if not explicit and os.environ.get("PUBMED_STORE"):
        store = Path(os.environ["PUBMED_STORE"]).resolve()
        index = Path(os.environ.get("PUBMED_INDEX", str(store.parent / "index"))).resolve()
        return store, index, {"dataset_id": "explicit-paths", "snapshot": digest(str(store))}
    manifest = Path(explicit) if explicit else DEFAULT_MANIFEST
    cfg = read_json(manifest)
    if cfg.get("status") != "ready":
        raise RuntimeError("Dataset is not ready; finish the store build first")
    return (manifest.parent / cfg["store"]).resolve(), (manifest.parent / cfg["index"]).resolve(), cfg


def pin():
    """Servers retain their selected immutable snapshot until explicitly restarted."""
    global _PINNED
    _PINNED = paths()
    return _PINNED


class BuildLock:
    """Exclusive writer lock, released by the OS even after a crashed build."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        self.file.seek(0)
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError(f"Another builder holds {self.path}")
        return self

    def __exit__(self, *args):
        self.file.close()
