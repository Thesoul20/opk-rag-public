from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "TASK-0256"
TASK_START_HEAD = "e25dcb287ec12ff416ec1d1d2e8636f45dc22886"
SCHEMA = "opk-rag.task0256.project-open-source-release-governance-stage-entry-and-public-release-readiness-baseline.v1"
RESULT = ROOT / "evaluation-data/results/task0256-project-open-source-release-governance-stage-entry-and-public-release-readiness-baseline"
CONTRACT = ROOT / "evaluation-data/contracts/task0256_project_open_source_release_governance_stage_entry_and_public_release_readiness_baseline.json"
TASK0255 = ROOT / "evaluation-data/results/task0255-selective-agent-production-post-release-monitoring-stability-authority-and-stage-closeout/summary.json"
REG = RESULT / "regression.json"

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mov", ".mkv", ".pdf", ".docx", ".zip", ".gz", ".bin", ".safetensors", ".pt", ".pth"}
SENSITIVE_NAMES = ("api_key", "token", "secret", "password", "service_role_key", "database_url", "authorization")
HIGH_CONFIDENCE_SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
)
LOCAL_IDENTITY_PATTERNS = (
    ("home_user_path", re.compile(r"/home/([A-Za-z0-9._-]+)/")),
    ("mac_user_path", re.compile(r"/Users/([A-Za-z0-9._-]+)/")),
)
PLACEHOLDER_MARKERS = ("<", "example", "placeholder", "redacted", "changeme", "your_", "dummy", "xxxx", "${")


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def head(root: Path = ROOT) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()


def changed_paths(root: Path = ROOT) -> list[str]:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True).stdout
    return sorted(line[3:].split(" -> ", 1)[-1] for line in out.splitlines() if len(line) >= 4)


def tracked_files(root: Path = ROOT) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    return [root / raw.decode("utf-8", errors="surrogateescape") for raw in out.split(b"\0") if raw]


def rel(path: Path, root: Path = ROOT) -> str:
    return path.relative_to(root).as_posix()


def iter_text_lines(path: Path) -> Iterable[tuple[int, str]]:
    if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, 1):
                yield line_no, line.rstrip("\n")
    except (OSError, UnicodeError):
        return


def digest_redacted(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def evaluate_entry_gate(task0255: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "task0255_complete": task0255.get("task_status") == "complete",
        "task0255_stability_frozen": task0255.get("candidate_decision") == "freeze_agentic_rag_v2_production_stability",
        "production_agentic_v2_active": task0255.get("production_agentic_v2_active") is True,
        "stability_gate_passed": task0255.get("stability_gate_passed") is True,
        "agent_development_stage_frozen": task0255.get("llm_agentic_rag_development_stage_frozen") is True,
        "hard_safety_zero": int(task0255.get("hard_safety_violation_count") or 0) == 0,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": "opk-rag.task0256.entry-gate.v1",
        "passed": not blockers,
        "gates": gates,
        "blockers": blockers,
        "source_task": "TASK-0255",
        "source_status": task0255.get("task_status"),
        "source_decision": task0255.get("candidate_decision"),
    }


def repository_inventory(root: Path = ROOT) -> dict[str, Any]:
    files = tracked_files(root)
    categories = Counter()
    total_bytes = 0
    large: list[dict[str, Any]] = []
    for path in files:
        name = rel(path, root)
        try: size = path.stat().st_size
        except OSError: size = 0
        total_bytes += size
        if name.startswith("opk_rag/"): category = "source"
        elif name.startswith("tests/"): category = "tests"
        elif name.startswith("tasks/"): category = "task_history"
        elif name.startswith("docs/"): category = "documentation"
        elif name.startswith("evaluation-data/"): category = "evaluation"
        elif name.startswith("source-documents/"): category = "demo_source_documents"
        elif name.startswith("showcase-ui/"): category = "showcase_ui"
        elif name.startswith("scripts/") or name.startswith("run/"): category = "operations"
        elif name.startswith("supabase/") or name.startswith("infra/"): category = "infrastructure"
        else: category = "project_root"
        categories[category] += 1
        if size >= 5_000_000:
            large.append({"path": name, "size_bytes": size})
    return {
        "schema_version": "opk-rag.task0256.repository-inventory.v1",
        "tracked_file_count": len(files),
        "tracked_total_bytes": total_bytes,
        "categories": dict(sorted(categories.items())),
        "large_tracked_file_count": len(large),
        "large_tracked_files": sorted(large, key=lambda x: x["size_bytes"], reverse=True)[:80],
        "inventory_valid": bool(files),
    }


def _git_grep(root: Path, pattern: str) -> list[tuple[str, int, str]]:
    proc=subprocess.run(["git","grep","-n","-I","-E","-e",pattern,"--"],cwd=root,text=True,capture_output=True,check=False)
    if proc.returncode not in {0,1}:
        raise RuntimeError(f"git_grep_failed:{proc.returncode}")
    rows=[]
    for line in proc.stdout.splitlines():
        parts=line.split(":",2)
        if len(parts)!=3: continue
        try: line_no=int(parts[1])
        except ValueError: continue
        rows.append((parts[0],line_no,parts[2]))
    return rows


def scan_secrets(root: Path = ROOT) -> dict[str, Any]:
    confirmed=[]; fixtures=[]; candidates=[]
    high_pattern=r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(ghp|github_pat)_[A-Za-z0-9_]{20,}\b|\bhf_[A-Za-z0-9]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b"
    for name,line_no,line in _git_grep(root,high_pattern):
        for kind,pattern in HIGH_CONFIDENCE_SECRET_PATTERNS:
            for match in pattern.finditer(line):
                row={"path":name,"line":line_no,"kind":kind,"value_digest":digest_redacted(match.group(0))}
                if name.startswith("tests/"):
                    row["classification"]="test_fixture_secret_pattern"; fixtures.append(row)
                else:
                    confirmed.append(row)
    assignment = re.compile(r"(?i)\b([A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET|PASSWORD|SERVICE_ROLE_KEY|DATABASE_URL|AUTHORIZATION))\s*[:=]\s*[\"']?([^\s\"',}]+)")
    candidate_pattern=r"(API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET|PASSWORD|SERVICE_ROLE_KEY|DATABASE_URL|AUTHORIZATION)[A-Z0-9_]*[[:space:]]*[:=]"
    for name,line_no,line in _git_grep(root,candidate_pattern):
        for match in assignment.finditer(line):
            value=match.group(2).strip(); lower=value.lower()
            if not value or any(marker in lower for marker in PLACEHOLDER_MARKERS) or value in {"true","false","0","1","none","null"}: continue
            candidates.append({"path":name,"line":line_no,"kind":"sensitive_assignment_candidate","name":match.group(1),"value_digest":digest_redacted(value)})
    def unique(rows):
        seen=set(); out=[]
        for row in rows:
            key=(row["path"],row["line"],row["kind"],row.get("name"),row["value_digest"])
            if key in seen: continue
            seen.add(key); out.append(row)
        return out
    confirmed=unique(confirmed); fixtures=unique(fixtures); candidates=unique(candidates)
    return {"schema_version":"opk-rag.task0256.secret-audit.v1","scan_scope":"git_tracked_text_files_via_git_grep","tracked_secret_count":len(confirmed),"test_fixture_secret_pattern_count":len(fixtures),"secret_candidate_count":len(candidates),"confirmed_findings":confirmed[:100],"test_fixture_findings":fixtures[:100],"candidate_findings":candidates[:150],"raw_secret_values_persisted":False,"passed":len(confirmed)==0}


def scan_local_identity(root: Path = ROOT) -> dict[str, Any]:
    findings=[]; counts=Counter()
    grep_pattern=r"/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/"
    for name,line_no,line in _git_grep(root,grep_pattern):
        for kind,pattern in LOCAL_IDENTITY_PATTERNS:
            for match in pattern.finditer(line):
                counts[kind]+=1
                if len(findings)<180:
                    findings.append({"path":name,"line":line_no,"kind":kind,"identity_digest":digest_redacted(match.group(1))})
    return {"schema_version":"opk-rag.task0256.local-machine-identity-audit.v1","local_identity_leak_count":sum(counts.values()),"counts":dict(counts),"findings":findings,"raw_identity_values_persisted":False,"public_release_blocking_severity":"P1" if counts else None}


def personal_data_audit(root: Path = ROOT) -> dict[str, Any]:
    tracked=[rel(p,root) for p in tracked_files(root)]
    private_tracked=[p for p in tracked if p.startswith(".private/") or p.startswith("evaluation-data/dogfooding/private/") or p==".env"]
    source_docs=[p for p in tracked if p.startswith("source-documents/")]
    return {
        "schema_version":"opk-rag.task0256.personal-data-audit.v1",
        "private_namespace_tracked_count":len(private_tracked),
        "private_namespace_tracked_paths":private_tracked[:100],
        "private_knowledge_content_count":len(private_tracked),
        "tracked_demo_source_document_count":len(source_docs),
        "demo_source_publication_authority":"requires_explicit_redistribution/provenance_review",
        "personal_knowledge_base_content_publication_allowed":False,
        "passed_high_confidence_private_namespace_check":len(private_tracked)==0,
    }


def runtime_artifact_audit(root: Path = ROOT) -> dict[str, Any]:
    tracked=[rel(p,root) for p in tracked_files(root)]
    runtime_tracked=[p for p in tracked if p.startswith("runtime/") or p.startswith(".private/") or p==".env"]
    required=[".env",".private/evaluation/phase2_corpus_manifest.json","runtime/qdrant/x","runtime/selective-agent-live-shadow/x","runtime/selective-agent-canary/x","runtime/selective-agent-production/x","runtime/selective-agent-production-monitoring/x"]
    checks={}
    for item in required:
        rc=subprocess.run(["git","check-ignore","-q",item],cwd=root).returncode
        checks[item]=rc==0
    return {
        "schema_version":"opk-rag.task0256.runtime-artifact-audit.v1",
        "private_runtime_artifact_tracked_count":len(runtime_tracked),
        "tracked_private_runtime_paths":runtime_tracked[:100],
        "ignore_checks":checks,
        "all_required_private_paths_ignored":all(checks.values()),
        "passed":len(runtime_tracked)==0 and all(checks.values()),
    }


def external_redistribution_audit(root: Path = ROOT) -> dict[str, Any]:
    tracked=[rel(p,root) for p in tracked_files(root)]
    raw_external=[p for p in tracked if p.startswith("evaluation-data/results/task0103-pdf-parser-bakeoff/candidates/") and "/raw/" in p]
    pdfqa_material=[p for p in tracked if p.startswith("evaluation-data/external/pdfqa/")]
    return {
        "schema_version":"opk-rag.task0256.dataset-redistribution-audit.v1",
        "pdfqa_dataset_revision":"90f787ebee0ba278bd6b8b750a69ed90386bfdf5",
        "pdfqa_annotation_revision":"81a4cba8d049b3fed9faf4a6830747895cdccebd",
        "tracked_pdfqa_governance_file_count":len(pdfqa_material),
        "tracked_parser_raw_external_material_count":len(raw_external),
        "tracked_parser_raw_external_material_examples":raw_external[:30],
        "raw_external_redistribution_authority_proven":False if raw_external else None,
        "dataset_redistribution_blocker_count":1 if raw_external else 0,
        "status":"hold_for_redistribution_review" if raw_external else "reference_only_review_required",
    }


def model_license_audit(root: Path = ROOT) -> dict[str, Any]:
    tracked=[rel(p,root) for p in tracked_files(root)]
    weights=[p for p in tracked if Path(p).suffix.lower() in {".safetensors",".pt",".pth",".gguf",".bin"} and "node_modules" not in p]
    return {
        "schema_version":"opk-rag.task0256.model-license-audit.v1",
        "models":[
            {"name":"Qwen/Qwen3-Embedding-0.6B","role":"embedding","license_verification":"external_metadata_review_required"},
            {"name":"BAAI/bge-reranker-v2-m3","role":"reranker","license_verification":"external_metadata_review_required"},
            {"name":"deepseek-v4-flash","role":"configured_external_provider_model","license_verification":"provider_terms_review_required"},
        ],
        "tracked_model_weight_count":len(weights),
        "tracked_model_weight_paths":weights[:50],
        "model_weight_redistribution":bool(weights),
        "passed_no_weight_redistribution":len(weights)==0,
    }


def dependency_license_inventory(root: Path = ROOT) -> dict[str, Any]:
    text=(root/"pyproject.toml").read_text(encoding="utf-8")
    def array_for(section: str, key: str) -> list[str]:
        section_match=re.search(rf"(?ms)^\[{re.escape(section)}\]\s*(.*?)(?=^\[|\Z)", text)
        if not section_match: return []
        body=section_match.group(1)
        key_match=re.search(rf"(?ms)^{re.escape(key)}\s*=\s*\[(.*?)\]", body)
        if not key_match: return []
        return re.findall(r'"([^"\n]+)"', key_match.group(1))
    deps=array_for("project","dependencies")
    dev=array_for("dependency-groups","dev")
    def name(spec:str)->str: return re.split(r"[<>=!~\[; ]",spec,1)[0]
    rows=[{"package":name(spec),"spec":spec,"scope":"runtime","license_status":"metadata_review_required"} for spec in deps]
    rows += [{"package":name(spec),"spec":spec,"scope":"dev","license_status":"metadata_review_required"} for spec in dev]
    return {"schema_version":"opk-rag.task0256.dependency-license-inventory.v1","dependency_count":len(rows),"dependencies":rows,"license_metadata_externally_verified":False,"status":"pre_entry_inventory_only"}


def project_license_readiness(root: Path = ROOT) -> dict[str, Any]:
    candidates=[root/n for n in ("LICENSE","LICENSE.md","COPYING","NOTICE")]
    existing=[rel(p,root) for p in candidates if p.is_file()]
    return {"schema_version":"opk-rag.task0256.project-license-readiness.v1","project_license_files":existing,"project_license_missing":not existing,"license_blocker_count":0 if existing else 1,"license_selection_automatically_applied":False,"status":"ready" if existing else "hold_for_explicit_license_governance"}


def environment_template_audit(root: Path = ROOT) -> dict[str, Any]:
    path=root/".env.example"; text=path.read_text(encoding="utf-8") if path.is_file() else ""
    values={}
    for line in text.splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line: continue
        k,v=line.split("=",1); values[k.strip()]=v.strip()
    required_safe={
        "OPK_RAG_SELECTIVE_AGENT_LIVE_SHADOW_ENABLED":"0",
        "OPK_RAG_SELECTIVE_AGENT_CANARY_ENABLED":"0",
        "OPK_RAG_SELECTIVE_AGENT_CANARY_MODE":"off",
        "OPK_RAG_SELECTIVE_AGENT_CANARY_EXPOSURE":"0",
        "OPK_RAG_SELECTIVE_AGENT_CANARY_KILL_SWITCH":"1",
        "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_ENABLED":"0",
        "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_KILL_SWITCH":"1",
        "OPK_RAG_SELECTIVE_AGENT_PRODUCTION_MONITORING_ENABLED":"0",
        "OPK_RAG_LLM_ALLOW_REMOTE":"false",
    }
    safe={k:values.get(k)==v for k,v in required_safe.items()}
    sensitive_nonempty=[]
    for key,value in values.items():
        if re.search(r"(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET|PASSWORD|SERVICE_ROLE_KEY)$", key, re.I) and value and not any(m in value.lower() for m in PLACEHOLDER_MARKERS):
            sensitive_nonempty.append(key)
    return {"schema_version":"opk-rag.task0256.environment-template-audit.v1","env_example_exists":path.is_file(),"safe_authority_controls":safe,"safe_authority_controls_valid":all(safe.values()),"nonempty_sensitive_placeholder_violation_count":len(sensitive_nonempty),"violating_variable_names":sensitive_nonempty,"passed":path.is_file() and all(safe.values()) and not sensitive_nonempty}


def clean_clone_reproducibility(root: Path = ROOT) -> dict[str, Any]:
    readme=(root/"README.md").read_text(encoding="utf-8") if (root/"README.md").is_file() else ""
    checks={
        "readme_exists":bool(readme),
        "uv_sync_documented":"uv sync --dev" in readme,
        "env_copy_documented":"cp .env.example .env" in readme,
        "doctor_documented":"opk-rag doctor" in readme,
        "index_public_source_documented":"opk-rag index source-documents" in readme,
        "search_documented":"opk-rag search" in readme,
        "ask_documented":"opk-rag ask" in readme,
    }
    # Runtime dependencies still require local services/config; this task audits docs, not a destructive clean-clone execution.
    return {"schema_version":"opk-rag.task0256.clean-clone-reproducibility.v1","checks":checks,"documentation_path_complete":all(checks.values()),"clean_clone_reproducible":False,"reason":"full_clean_clone_execution_not_authorized_before_stage_entry"}


def public_demo_corpus_readiness(root: Path = ROOT) -> dict[str, Any]:
    files=[rel(p,root) for p in tracked_files(root) if rel(p,root).startswith("source-documents/")]
    manifest=root/"evaluation-data/showcase/project-showcase-demo-manifest.json"
    return {"schema_version":"opk-rag.task0256.public-demo-corpus-readiness.v1","tracked_source_document_count":len(files),"showcase_manifest_exists":manifest.is_file(),"redistribution_authority_explicit":False,"public_demo_corpus_ready":False,"reason":"tracked_demo_content_exists_but_explicit_redistribution_provenance_not_frozen"}


def audit_claim_text(text: str) -> dict[str, Any]:
    lower=text.lower()
    violations=[]
    if ("production" in lower or "生产" in text) and "96.67%" in text:
        violations.append("unscoped_production_accuracy_claim")
    if "100% accuracy" in lower or "100% 准确" in text:
        violations.append("universal_accuracy_claim")
    if "unrestricted autonomous agent" in lower or "完全自主 agent" in lower:
        violations.append("unrestricted_autonomous_agent_claim")
    scoped_9667 = "96.67%" in text and ("30" in text or "frozen" in lower or "benchmark" in lower or "基准" in text) and not violations
    return {"violations":violations,"public_safe":not violations,"scoped_96_67_claim_recognized":scoped_9667}


def readme_and_claim_audit(root: Path = ROOT) -> tuple[dict[str,Any],dict[str,Any]]:
    text=(root/"README.md").read_text(encoding="utf-8") if (root/"README.md").is_file() else ""
    claim=audit_claim_text(text)
    readme_checks={
        "project_scope_present":"OPK-RAG" in text,
        "local_first_present":"本地优先" in text or "local-first" in text.lower(),
        "qdrant_present":"Qdrant" in text,
        "graph_one_hop_scope_present":"一跳" in text or "hop_depth=1" in text or "graph_runtime_hop_depth=1" in text,
        "not_multitenant_scope_present":"多租户" in text or "multi_tenant_support=false" in text,
        "guarded_agent_not_open_planner":"不是开放式自主 Planner" in text or "unrestricted" not in text.lower(),
    }
    return (
        {"schema_version":"opk-rag.task0256.readme-authority-audit.v1","checks":readme_checks,"readme_authority_valid":all(readme_checks.values()) and claim["public_safe"],"claim_violations":claim["violations"]},
        {"schema_version":"opk-rag.task0256.public-claim-authority-audit.v1",**claim,"quantitative_claim_scope_required":True,"production_accuracy_claim_authorized":False},
    )


def public_private_classification(root: Path, inventory: Mapping[str,Any], redistribution: Mapping[str,Any], local_identity: Mapping[str,Any]) -> dict[str,Any]:
    categories={
        "opk_rag/":"public_safe_candidate",
        "tests/":"public_safe_candidate",
        "docs/":"public_safe_after_local_identity_review",
        "tasks/":"public_safe_after_local_identity_review",
        "source-documents/":"public_safe_after_explicit_redistribution_review",
        "evaluation-data/results/task0103-pdf-parser-bakeoff/candidates/":"reference_only_not_redistributable_until_review",
        "runtime/":"runtime_only_gitignored",
        ".private/":"private_gitignored",
        ".env":"private_gitignored",
        ".env.example":"public_safe_template",
    }
    return {"schema_version":"opk-rag.task0256.public-private-file-classification.v1","classification_rules":categories,"tracked_file_count":inventory.get("tracked_file_count"),"unknown_requires_review":True,"pre_entry_only":True,"local_identity_review_required":int(local_identity.get("local_identity_leak_count") or 0)>0,"external_raw_material_review_required":int(redistribution.get("dataset_redistribution_blocker_count") or 0)>0}


def safe_default_audit(env_audit: Mapping[str,Any]) -> dict[str,Any]:
    return {"schema_version":"opk-rag.task0256.safe-default-audit.v1","authority_changing_agent_controls_safe":env_audit.get("safe_authority_controls_valid") is True,"external_provider_requires_configuration":True,"knowledge_base_mutation_by_agent_default":False,"public_network_exposure_default":False,"public_clone_safe_defaults":env_audit.get("passed") is True}


def historical_artifact_policy() -> dict[str,Any]:
    return {"schema_version":"opk-rag.task0256.historical-artifact-policy.v1","keep":["canonical architecture docs","key frozen contracts","benchmark methodology","release authority summaries"],"archive_candidates":["intermediate task reports","superseded experiment results","large generated traces"],"exclude":["runtime stores","private corpus manifests","credentials","model caches"],"automatic_deletion_allowed":False,"status":"policy_defined_no_deletion"}


def ci_readiness(root: Path=ROOT) -> dict[str,Any]:
    workflow_dir=root/".github/workflows"
    workflows=list(workflow_dir.glob("*.yml"))+list(workflow_dir.glob("*.yaml")) if workflow_dir.is_dir() else []
    return {"schema_version":"opk-rag.task0256.ci-readiness.v1","tracked_workflow_count":len(workflows),"default_ci_defined":bool(workflows),"recommended_lanes":["fast","core-no-private-services","optional-integration"],"ci_readiness_valid":bool(workflows),"paid_provider_required_for_default_ci":False}


def security_surface_audit(root: Path=ROOT) -> dict[str,Any]:
    return {"schema_version":"opk-rag.task0256.security-surface-audit.v1","public_network_service_authorized":False,"cloudflare_or_agentdock_public_service_part_of_opk_rag_release":False,"showcase_api_requires_deployment_hardening_review":True,"external_provider_privacy_disclosure_required":True,"local_first_means_always_offline":False,"status":"pre_release_hardening_review_required"}


def test_suite_public_readiness() -> dict[str,Any]:
    reg=read_json(REG,{})
    return {"schema_version":"opk-rag.task0256.test-suite-public-readiness.v1","current_full_suite_passed":reg.get("full_suite_passed"),"current_full_suite_skipped":reg.get("full_suite_skipped"),"current_full_suite_failed":reg.get("full_suite_failed"),"known_historical_failure_families":["CLI JSON output","TASK-0212 execution-scope historical expectation","TASK-0223/0224 historical showcase governance","TASK-0251 task-start HEAD assertion"],"public_ci_green":False,"status":"historical_test_debt_requires_reconciliation_before_public_release"}


def build_blockers(*, secret:Mapping[str,Any], personal:Mapping[str,Any], runtime:Mapping[str,Any], redistribution:Mapping[str,Any], license_readiness:Mapping[str,Any], local_identity:Mapping[str,Any], env_audit:Mapping[str,Any], ci:Mapping[str,Any], demo:Mapping[str,Any], entry_passed:bool) -> dict[str,Any]:
    blockers=[]
    def add(severity,code,reason): blockers.append({"severity":severity,"code":code,"reason":reason})
    if int(secret.get("tracked_secret_count") or 0)>0: add("P0","tracked_secret","confirmed secret-like material exists in tracked content")
    if int(personal.get("private_knowledge_content_count") or 0)>0: add("P0","private_kb_tracked","private namespace/KB content is tracked")
    if int(runtime.get("private_runtime_artifact_tracked_count") or 0)>0 or not runtime.get("all_required_private_paths_ignored"): add("P0","private_runtime_tracking","private/runtime artifact isolation is invalid")
    if int(redistribution.get("dataset_redistribution_blocker_count") or 0)>0: add("P0","external_raw_redistribution_unproven","tracked raw external parser material lacks frozen redistribution authority")
    if license_readiness.get("project_license_missing"): add("P1","project_license_missing","no project-level LICENSE/COPYING authority is present")
    if int(local_identity.get("local_identity_leak_count") or 0)>0: add("P1","local_machine_identity_leak","tracked docs/evaluation artifacts contain developer-specific absolute user paths")
    if not env_audit.get("passed"): add("P1","environment_template_not_public_safe",".env.example authority-changing defaults/template are not fully public-safe")
    if not ci.get("ci_readiness_valid"): add("P1","public_ci_missing","no tracked default public CI workflow is defined")
    if not demo.get("public_demo_corpus_ready"): add("P1","demo_corpus_authority_unproven","tracked demo corpus lacks frozen explicit redistribution/provenance authority")
    if not entry_passed: add("P1","stage_entry_blocked","TASK-0255 has not frozen Agentic RAG V2 production stability")
    counts=Counter(x["severity"] for x in blockers)
    return {"schema_version":"opk-rag.task0256.public-release-blockers.v1","blockers":blockers,"p0_blocker_count":counts.get("P0",0),"p1_blocker_count":counts.get("P1",0),"p2_known_limitation_count":4,"known_p2_limitations":["single-node Qdrant authority","not multi-tenant SaaS","Graph hop depth limited to one","optional external Provider can send configured content off-device"]}


def release_readiness_gate(entry:Mapping[str,Any], secret:Mapping[str,Any], personal:Mapping[str,Any], runtime:Mapping[str,Any], redistribution:Mapping[str,Any], license_readiness:Mapping[str,Any], clean:Mapping[str,Any], demo:Mapping[str,Any], safe:Mapping[str,Any], readme:Mapping[str,Any], claims:Mapping[str,Any], ci:Mapping[str,Any], blockers:Mapping[str,Any]) -> dict[str,Any]:
    gates={
        "stage_entry":entry.get("passed") is True,
        "tracked_secret_zero":int(secret.get("tracked_secret_count") or 0)==0,
        "private_kb_zero":int(personal.get("private_knowledge_content_count") or 0)==0,
        "runtime_isolation":runtime.get("passed") is True,
        "redistribution_resolved":int(redistribution.get("dataset_redistribution_blocker_count") or 0)==0,
        "project_license_authority":not license_readiness.get("project_license_missing"),
        "clean_clone_reproducible":clean.get("clean_clone_reproducible") is True,
        "demo_corpus_ready":demo.get("public_demo_corpus_ready") is True,
        "safe_defaults":safe.get("public_clone_safe_defaults") is True,
        "readme_authority":readme.get("readme_authority_valid") is True,
        "claims_authority":claims.get("public_safe") is True,
        "ci_ready":ci.get("ci_readiness_valid") is True,
        "p0_zero":int(blockers.get("p0_blocker_count") or 0)==0,
    }
    return {"schema_version":"opk-rag.task0256.public-release-readiness-gate.v1","gates":gates,"public_release_readiness":all(gates.values()),"decision":"advance_to_public_release_packaging" if all(gates.values()) else "hold"}


def _contract() -> dict[str,Any]:
    return {"schema_version":"opk-rag.task0256.contract.v1","task_id":TASK_ID,"task_start_head":TASK_START_HEAD,"previous_stage":"llm_agentic_rag_development","target_stage":"project_open_source_release_governance","entry_requires_task0255_decision":"freeze_agentic_rag_v2_production_stability","evaluator_may_publish":False,"evaluator_may_push":False,"evaluator_may_change_visibility":False,"evaluator_may_rewrite_history":False,"project_license_selection_automatic":False,"public_release_execution_allowed":False,"secret_values_may_be_persisted":False}


def build_summary(*, write:bool=True, root:Path=ROOT) -> dict[str,Any]:
    t255=read_json(root / TASK0255.relative_to(ROOT),{})
    entry=evaluate_entry_gate(t255)
    inventory=repository_inventory(root)
    secret=scan_secrets(root)
    personal=personal_data_audit(root)
    local_identity=scan_local_identity(root)
    runtime=runtime_artifact_audit(root)
    redistribution=external_redistribution_audit(root)
    model=model_license_audit(root)
    deps=dependency_license_inventory(root)
    license_readiness=project_license_readiness(root)
    env_audit=environment_template_audit(root)
    safe=safe_default_audit(env_audit)
    clean=clean_clone_reproducibility(root)
    demo=public_demo_corpus_readiness(root)
    readme,claims=readme_and_claim_audit(root)
    ci=ci_readiness(root)
    security=security_surface_audit(root)
    historical=historical_artifact_policy()
    classification=public_private_classification(root,inventory,redistribution,local_identity)
    blockers=build_blockers(secret=secret,personal=personal,runtime=runtime,redistribution=redistribution,license_readiness=license_readiness,local_identity=local_identity,env_audit=env_audit,ci=ci,demo=demo,entry_passed=entry["passed"])
    gate=release_readiness_gate(entry,secret,personal,runtime,redistribution,license_readiness,clean,demo,safe,readme,claims,ci,blockers)
    reg=read_json(root / REG.relative_to(ROOT),{})
    decision="blocked" if not entry["passed"] else ("advance_to_public_release_packaging" if gate["public_release_readiness"] else "hold_for_sanitization")
    summary={
        "schema_version":SCHEMA,"task_id":TASK_ID,"task_status":"blocked" if not entry["passed"] else "partial","implementation_complete":True,
        "previous_stage":"llm_agentic_rag_development","current_stage":"llm_agentic_rag_development" if not entry["passed"] else "project_open_source_release_governance",
        "entry_gate_passed":entry["passed"],"llm_agentic_rag_development_stage_frozen":t255.get("llm_agentic_rag_development_stage_frozen") is True,
        "project_open_source_release_governance_stage_active":entry["passed"],"pre_entry_read_only_audit_executed":True,
        "repository_inventory_valid":inventory["inventory_valid"],"tracked_secret_count":secret["tracked_secret_count"],"secret_candidate_count":secret["secret_candidate_count"],
        "private_knowledge_content_count":personal["private_knowledge_content_count"],"private_runtime_artifact_tracked_count":runtime["private_runtime_artifact_tracked_count"],
        "local_identity_leak_count":local_identity["local_identity_leak_count"],"dataset_redistribution_blocker_count":redistribution["dataset_redistribution_blocker_count"],
        "license_blocker_count":license_readiness["license_blocker_count"],"clean_clone_reproducible":clean["clean_clone_reproducible"],"public_demo_corpus_ready":demo["public_demo_corpus_ready"],
        "public_clone_safe_defaults":safe["public_clone_safe_defaults"],"readme_authority_valid":readme["readme_authority_valid"],"public_claim_authority_valid":claims["public_safe"],
        "ci_readiness_valid":ci["ci_readiness_valid"],"p0_blocker_count":blockers["p0_blocker_count"],"p1_blocker_count":blockers["p1_blocker_count"],"p2_known_limitation_count":blockers["p2_known_limitation_count"],
        "public_release_readiness":False if not entry["passed"] else gate["public_release_readiness"],"public_release_executed":False,"git_push_executed":False,"repository_visibility_changed":False,"git_history_rewrite_executed":False,
        "candidate_decision":decision,"blocking_failures":entry["blockers"] if not entry["passed"] else [b["code"] for b in blockers["blockers"]],
        "next_task":"TASK-0251_resume_after_minimum_live_shadow_traffic" if not entry["passed"] else ("TASK-0257_public_release_packaging" if gate["public_release_readiness"] else "TASK-0256_resolve_public_release_blockers"),
        "task_start_head":TASK_START_HEAD,"current_head":head(root),"git_head_unchanged_since_task_start":head(root)==TASK_START_HEAD,"git_commit_created":head(root)!=TASK_START_HEAD,
        "focused_tests_passed":reg.get("focused_tests_passed"),"full_suite_passed":reg.get("full_suite_passed"),"full_suite_skipped":reg.get("full_suite_skipped"),"full_suite_failed":reg.get("full_suite_failed"),"new_task0256_full_suite_failure_count":reg.get("new_task0256_full_suite_failure_count"),"changed_paths":changed_paths(root),
    }
    if write:
        result=root / RESULT.relative_to(ROOT); contract=root / CONTRACT.relative_to(ROOT)
        write_json(contract,_contract()); result.mkdir(parents=True,exist_ok=True)
        artifacts={
            "entry_gate.json":entry,"repository_inventory.json":inventory,"public_private_file_classification.json":classification,"secret_audit.json":secret,"personal_data_audit.json":personal,
            "local_machine_identity_audit.json":local_identity,"runtime_artifact_audit.json":runtime,"dataset_redistribution_audit.json":redistribution,"model_license_audit.json":model,"dependency_license_inventory.json":deps,
            "project_license_readiness.json":license_readiness,"clean_clone_reproducibility.json":clean,"public_demo_corpus_readiness.json":demo,"environment_template_audit.json":env_audit,"safe_default_audit.json":safe,
            "readme_authority_audit.json":readme,"public_claim_authority_audit.json":claims,"historical_artifact_policy.json":historical,"test_suite_public_readiness.json":test_suite_public_readiness(),"ci_readiness.json":ci,
            "security_surface_audit.json":security,"public_release_blockers.json":blockers,"public_release_readiness_gate.json":gate,"summary.json":summary,
        }
        for name,payload in artifacts.items(): write_json(result/name,payload)
        regression_path=result/"regression.json"
        if not regression_path.is_file():
            write_json(regression_path,{"schema_version":"opk-rag.task0256.regression.v1","task_id":TASK_ID,"focused_tests_passed":None,"full_suite_passed":None,"full_suite_skipped":None,"full_suite_failed":None,"new_task0256_full_suite_failure_count":None,"task0256_verifier_passed":None,"git_diff_check_passed":None,"full_suite_side_effect_artifacts_restored":None})
    return summary


def verify(root:Path=ROOT) -> dict[str,Any]:
    summary=build_summary(write=False,root=root)
    result=root / RESULT.relative_to(ROOT)
    required=[root/CONTRACT.relative_to(ROOT),root/"tasks/TASK-0256_project_open_source_release_governance_stage_entry_and_public_release_readiness_baseline.md",root/"docs/TASK0256_PROJECT_OPEN_SOURCE_RELEASE_GOVERNANCE_STAGE_ENTRY_AND_PUBLIC_RELEASE_READINESS_BASELINE_REPORT.md",root/"scripts/run_task0256_project_open_source_release_governance_stage_entry_and_public_release_readiness_baseline.py",root/"scripts/verify_task0256_project_open_source_release_governance_stage_entry_and_public_release_readiness_baseline.py",result/"summary.json",result/"secret_audit.json",result/"public_release_blockers.json",result/"public_release_readiness_gate.json",result/"regression.json"]
    missing=[rel(p,root) for p in required if not p.is_file()]
    mismatches={}
    fixed={"implementation_complete":True,"public_release_executed":False,"git_push_executed":False,"repository_visibility_changed":False,"git_history_rewrite_executed":False,"git_head_unchanged_since_task_start":True,"git_commit_created":False}
    for k,v in fixed.items():
        if summary.get(k)!=v: mismatches[k]={"expected":v,"actual":summary.get(k)}
    if not summary["entry_gate_passed"]:
        expected={"task_status":"blocked","current_stage":"llm_agentic_rag_development","project_open_source_release_governance_stage_active":False,"public_release_readiness":False,"candidate_decision":"blocked"}
        for k,v in expected.items():
            if summary.get(k)!=v: mismatches[k]={"expected":v,"actual":summary.get(k)}
    stored=read_json(result/"summary.json",{})
    for k in ("task_status","entry_gate_passed","tracked_secret_count","p0_blocker_count","public_release_readiness","candidate_decision"):
        if stored.get(k)!=summary.get(k): mismatches[f"stored_{k}"]={"expected":summary.get(k),"actual":stored.get(k)}
    return {"schema_version":SCHEMA,"task_id":TASK_ID,"verification_passed":not missing and not mismatches,"task_status":summary["task_status"],"entry_gate_passed":summary["entry_gate_passed"],"public_release_readiness":summary["public_release_readiness"],"candidate_decision":summary["candidate_decision"],"missing_files":missing,"mismatches":mismatches}
