from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist/opk-rag-public-release"

ROOT_FILES = [
    "README.md", "LICENSE", "CONTRIBUTING.md", "SECURITY.md",
    ".env.public.example", "pyproject.toml", "uv.lock",
]
TREE_SPECS = [
    ("opk_rag", "opk_rag"),
    ("showcase-ui", "showcase-ui"),
    ("supabase/migrations", "supabase/migrations"),
    ("infra/public", "infra/public"),
    ("examples/public-demo", "examples/public-demo"),
    ("release/evidence", "release/evidence"),
    ("tests/public", "tests/public"),
    (".github/workflows", ".github/workflows"),
]
DOC_FILES = [
    "docs/QUICK_START.md", "docs/OPEN_SOURCE_RELEASE.md", "docs/THIRD_PARTY.md",
    "docs/diagrams/current_opk_rag_selective_agent_architecture_v2.mmd",
    "docs/diagrams/current_opk_rag_selective_agent_architecture_v2.drawio",
    "docs/diagrams/current_opk_rag_selective_agent_architecture_v2.png",
    "docs/diagrams/current_opk_rag_selective_agent_architecture_v2_zh.mmd",
    "docs/diagrams/current_opk_rag_selective_agent_architecture_v2_zh.drawio",
    "docs/diagrams/current_opk_rag_selective_agent_architecture_v2_zh.png",
    "docs/assets/control-center-executive.jpg", "docs/assets/video-cover.jpg",
]
SCRIPT_FILES = ["scripts/export_public_release.py", "scripts/validate_public_release_snapshot.py"]
SKIP_PARTS = {".git", ".venv", "node_modules", "dist", "coverage", "__pycache__", ".pytest_cache"}
FORBIDDEN_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".pt", ".pth", ".onnx", ".bin", ".db", ".sqlite", ".dump", ".pyc"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".sh", ".mjs", ".js", ".ts", ".tsx", ".css", ".html", ".sql", ".xml", ".drawio"}
TEXT_NAMES = {"LICENSE", ".gitignore", ".env.example", ".env.public.example"}
MAX_FILE_BYTES = 5 * 1024 * 1024

REPLACEMENTS = {
    "<workspace>/opk-rag-testv1": "<workspace>/opk-rag-testv1",
    "<repo>": "<repo>",
    "<knowledge-base>": "<knowledge-base>",
    "${HOME}": "${HOME}",
    "${HOME}": "${HOME}",
}
IDENTITY_PATTERNS = [
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"/data/envs/[A-Za-z0-9._/-]+"),
]
SECRET_PATTERNS = [
    ("openai_style_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github_pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
]
FORBIDDEN_TOP_LEVEL = {"evaluation-data", "source-documents", "tasks", "runtime", ".private", ".env"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_text(path: Path) -> bool:
    return path.name in TEXT_NAMES or path.suffix.lower() in TEXT_SUFFIXES


def copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    replacements = 0
    if is_text(dst):
        try:
            text = dst.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return 0
        for old, new in REPLACEMENTS.items():
            count = text.count(old)
            if count:
                text = text.replace(old, new)
                replacements += count
        # Historical evaluation helpers may contain dummy keys shaped like real provider keys.
        # Redact them in the public snapshot while preserving the development evidence unchanged.
        text, key_count = re.subn(r"sk-[A-Za-z0-9_-]{20,}", "<redacted-openai-style-key>", text)
        replacements += key_count
        text, pat_count = re.subn(r"github_pat_[A-Za-z0-9_]{20,}", "<redacted-github-token>", text)
        replacements += pat_count
        dst.write_text(text, encoding="utf-8")
    return replacements


def copy_tree(src_rel: str, dst_rel: str, output: Path) -> int:
    src_root = ROOT / src_rel
    if not src_root.exists():
        raise FileNotFoundError(src_root)
    replacements = 0
    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        if src.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        replacements += copy_file(src, output / dst_rel / rel)
    return replacements


def write_public_gitignore(output: Path) -> None:
    (output / ".gitignore").write_text(
        """.env\n.venv/\nnode_modules/\ndist/\nruntime/\n.private/\n__pycache__/\n.pytest_cache/\n*.pyc\n*.mp4\n*.pt\n*.pth\n*.onnx\n""",
        encoding="utf-8",
    )


def audit(output: Path) -> dict:
    files = [p for p in output.rglob("*") if p.is_file() and p.name != "public_release_manifest.json"]
    identity = []
    secrets = []
    oversize = []
    forbidden_binary = []
    forbidden_roots = []
    for p in files:
        rel = p.relative_to(output).as_posix()
        if p.stat().st_size > MAX_FILE_BYTES:
            oversize.append({"path": rel, "size_bytes": p.stat().st_size})
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden_binary.append(rel)
        if rel.split("/", 1)[0] in FORBIDDEN_TOP_LEVEL:
            forbidden_roots.append(rel)
        if is_text(p):
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pat in IDENTITY_PATTERNS:
                for m in pat.finditer(text):
                    identity.append({"path": rel, "match": m.group(0)})
            for name, pat in SECRET_PATTERNS:
                if pat.search(text):
                    secrets.append({"path": rel, "kind": name})
    total = sum(p.stat().st_size for p in files)
    listing = [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(files)]
    checks = {
        "no_personal_absolute_paths": not identity,
        "no_secret_patterns": not secrets,
        "no_oversized_files": not oversize,
        "no_forbidden_binary_extensions": not forbidden_binary,
        "no_forbidden_top_level_content": not forbidden_roots,
        "private_kb_absent": not (output / "source-documents").exists(),
        "historical_evaluation_absent": not (output / "evaluation-data").exists(),
        "task_history_absent": not (output / "tasks").exists(),
        "git_history_absent": not (output / ".git").exists(),
        "license_present": (output / "LICENSE").is_file(),
        "readme_present": (output / "README.md").is_file(),
        "quick_start_present": (output / "docs/QUICK_START.md").is_file(),
        "public_demo_present": (output / "examples/public-demo/README.md").is_file(),
        "ci_present": (output / ".github/workflows/ci.yml").is_file(),
    }
    return {
        "schema_version": "opk-rag.public-release.snapshot-manifest.v1",
        "file_count": len(files),
        "total_bytes": total,
        "max_file_bytes_policy": MAX_FILE_BYTES,
        "checks": checks,
        "identity_findings": identity,
        "secret_findings": secrets,
        "oversized_files": oversize,
        "forbidden_binary_files": forbidden_binary,
        "forbidden_root_files": forbidden_roots,
        "files": listing,
        "release_snapshot_valid": all(checks.values()),
    }


def export(output: Path) -> dict:
    output = output.resolve()
    if output == ROOT or ROOT in output.parents and output.relative_to(ROOT).parts[:1] not in [("dist",)]:
        raise RuntimeError("output must be outside the source tree or under repo/dist")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    replacements = 0
    for rel in ROOT_FILES:
        replacements += copy_file(ROOT / rel, output / rel)
    # Public convention: safe public env is also exposed as .env.example.
    replacements += copy_file(ROOT / ".env.public.example", output / ".env.example")
    for src, dst in TREE_SPECS:
        replacements += copy_tree(src, dst, output)
    for rel in DOC_FILES + SCRIPT_FILES:
        replacements += copy_file(ROOT / rel, output / rel)
    write_public_gitignore(output)
    manifest = audit(output)
    manifest["sanitized_replacement_count"] = replacements
    manifest["development_git_head"] = _git_head()
    manifest["output"] = str(output)
    (output / "public_release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not manifest["release_snapshot_valid"]:
        raise RuntimeError("public release snapshot audit failed; inspect public_release_manifest.json")
    return manifest


def _git_head() -> str | None:
    import subprocess
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a sanitized OPK-RAG public release snapshot without development Git history.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export(args.output)
    print(json.dumps({k: manifest[k] for k in ("release_snapshot_valid", "file_count", "total_bytes", "sanitized_replacement_count", "development_git_head")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
