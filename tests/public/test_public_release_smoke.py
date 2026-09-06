from __future__ import annotations
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_core_imports():
    import opk_rag  # noqa: F401
    import opk_rag.cli  # noqa: F401

def test_cli_help_is_available():
    proc = subprocess.run(["uv", "run", "opk-rag", "--help"], cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert "doctor" in proc.stdout and "search" in proc.stdout and "ask" in proc.stdout

def test_public_demo_is_project_authored_and_nonempty():
    demo = ROOT / "examples/public-demo"
    assert (demo / "README.md").is_file()
    assert len(list(demo.glob("*.md"))) >= 4
    assert "authored specifically" in (demo / "README.md").read_text(encoding="utf-8")

def test_public_env_fail_closed_for_remote_and_agent():
    text = (ROOT / ".env.public.example").read_text(encoding="utf-8")
    assert "OPK_RAG_LLM_ALLOW_REMOTE=false" in text
    assert "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_ENABLED=false" in text
    assert "OPK_RAG_SELECTIVE_AGENT_CANARY_ENABLED=false" in text
