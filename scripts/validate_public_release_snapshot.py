from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path, *, timeout: int = 900, env: dict[str,str] | None = None) -> dict:
    started=time.monotonic()
    merged=os.environ.copy()
    merged.pop("VIRTUAL_ENV",None)
    if env: merged.update(env)
    p=subprocess.run(cmd,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout,env=merged)
    out=p.stdout or ""
    return {"command":cmd,"returncode":p.returncode,"passed":p.returncode==0,"elapsed_seconds":round(time.monotonic()-started,3),"output_tail":out[-5000:]}


def validate(snapshot: Path) -> dict:
    snapshot=snapshot.resolve()
    checks={}
    checks["uv_sync"]=run(["uv","sync","--dev","--frozen"],snapshot,timeout=1200,env={"UV_LINK_MODE":"copy"})
    checks["compileall"]=run(["uv","run","python","-m","compileall","-q","opk_rag"],snapshot,timeout=180)
    checks["cli_help"]=run(["uv","run","opk-rag","--help"],snapshot,timeout=120)
    checks["public_tests"]=run(["uv","run","pytest","tests/public","-q"],snapshot,timeout=180)
    checks["compose_config"]=run(["docker","compose","-f","infra/public/docker-compose.yml","config"],snapshot,timeout=120,env={"OPK_RAG_PUBLIC_POSTGRES_PORT":"15432","OPK_RAG_PUBLIC_QDRANT_HTTP_PORT":"16333","OPK_RAG_PUBLIC_QDRANT_GRPC_PORT":"16334"})
    ui=snapshot/"showcase-ui"
    checks["frontend_npm_ci"]=run(["npm","ci","--no-audit","--no-fund"],ui,timeout=300)
    checks["frontend_typecheck"]=run(["npm","run","typecheck"],ui,timeout=180)
    checks["frontend_build"]=run(["npm","run","build"],ui,timeout=300)
    return {"schema_version":"opk-rag.public-release.clean-snapshot-validation.v1","snapshot":str(snapshot),"checks":checks,"passed":all(x["passed"] for x in checks.values())}


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",type=Path,required=True)
    ap.add_argument("--output-json",type=Path)
    args=ap.parse_args()
    result=validate(args.snapshot)
    text=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_json.write_text(text,encoding="utf-8")
    print(json.dumps({"passed":result["passed"],"checks":{k:v["passed"] for k,v in result["checks"].items()}},ensure_ascii=False,indent=2))
    return 0 if result["passed"] else 1

if __name__=="__main__": raise SystemExit(main())
