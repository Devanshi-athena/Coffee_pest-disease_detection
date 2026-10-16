"""Stage the app for a Hugging Face Docker Space and upload it.

Needs a write token: run `hf auth login` once (or set HF_TOKEN). The model files come from backend/models/
(not in git). Usage:
    ml/.venv-eval/Scripts/python deploy/huggingface/push_space.py --space <owner>/<space-name> [--dry-run]
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL_FILES = ["model.onnx", "labels.json", "preprocess.json", "decision.json", "ood_bank.npz", "model_info.json"]
SKIP = shutil.ignore_patterns("node_modules", "dist", ".vite", "__pycache__", ".env", ".env.local", "*.pyc")


def stage(dst: Path) -> None:
    shutil.copy(HERE / "Dockerfile", dst / "Dockerfile")
    shutil.copy(HERE / "README.md", dst / "README.md")
    shutil.copytree(ROOT / "configs", dst / "configs")
    shutil.copytree(ROOT / "frontend", dst / "frontend", ignore=SKIP)
    (dst / "backend").mkdir()
    shutil.copy(ROOT / "backend/requirements.txt", dst / "backend/requirements.txt")
    shutil.copytree(ROOT / "backend/app", dst / "backend/app", ignore=SKIP)
    models = dst / "backend/models"
    models.mkdir()
    info = json.loads((ROOT / "backend/models/model_info.json").read_text())
    for f in MODEL_FILES:
        src = ROOT / "backend/models" / f
        if f in info["files_sha256"]:  # never ship a model that differs from the recorded one
            got = hashlib.sha256(src.read_bytes()).hexdigest()
            if got != info["files_sha256"][f]:
                raise SystemExit(f"{f}: checksum {got} does not match model_info.json")
        shutil.copy(src, models / f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="<owner>/<space-name>")
    ap.add_argument("--dry-run", action="store_true", help="stage and list files, do not upload")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp)
        stage(dst)
        files = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file())
        size = sum((dst / f).stat().st_size for f in files) / 1e6
        print(f"{len(files)} files, {size:.1f} MB")
        if args.dry_run:
            print("\n".join(files))
            return
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
        api.upload_folder(folder_path=dst, repo_id=args.space, repo_type="space",
                          commit_message="Deploy Coffee Leaf AI (baseline model)")
        print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
