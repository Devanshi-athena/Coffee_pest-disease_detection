"""Download the served model files from the GitHub release and verify them (standard library only).

Checksums: deploy/render/model_files.sha256 (the same list the Render image checks).
Usage: python deploy/fetch_model.py <target_dir>
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/Devanshi-athena/Coffee_pest-disease_detection/releases/download/baseline-v1"
SUMS = Path(__file__).resolve().parent / "render" / "model_files.sha256"


def expected() -> dict[str, str]:
    out = {}
    for line in SUMS.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.lstrip("*")] = digest
    return out


def fetch(target: Path) -> Path:
    """Make sure every model file is in `target` with the right checksum; download what is missing."""
    target.mkdir(parents=True, exist_ok=True)
    for name, digest in expected().items():
        path = target / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(f"{RELEASE}/{name}", tmp)
        got = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if got != digest:
            tmp.unlink()
            raise RuntimeError(f"{name}: checksum {got} does not match {digest}")
        tmp.replace(path)
    return target


if __name__ == "__main__":
    print(fetch(Path(sys.argv[1])))
