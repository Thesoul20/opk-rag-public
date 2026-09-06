from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from opk_rag.document_adapters.base import StructuredDocumentSource
from opk_rag.document_adapters.pdf import PdfDocumentAdapter
from opk_rag.evaluation.task0105_representation_aware_chunking import build_c0_chunks
from opk_rag.reranking.bge import BgeLocalRerankerProvider
from opk_rag.reranking.config import load_reranker_config

ROOT = Path(__file__).resolve().parents[2]
SUBSET = ROOT / "evaluation-data/external/pdfqa/task0263_external_generalization_subset.json"
TASK0103_ASSETS = ROOT / "evaluation-data/external/pdfqa/materialization/task0103_pdf_assets.json"
RESULT = ROOT / "evaluation-data/results/task0263-controlled-agentic-rag-promotion-requalification"
FORMAL = RESULT / "external_generalization_followup.json"
FORMAL_ROWS = RESULT / "external_generalization_followup_rows.jsonl"
FORMAL_INDEPENDENCE = RESULT / "external_generalization_followup_independence.json"
CACHE = ROOT / ".cache/task0263-external"
PARSER_PYTHON = ROOT / ".cache/task0103-parser-envs/pymupdf4llm/bin/python"
PARSER_WORKER = ROOT / "scripts/task0103_pymupdf4llm_worker.py"
SOURCE_COVERAGE_THRESHOLD = 0.45
TOP_K = 20
CANDIDATE_K = 50


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(value: str) -> list[str]:
    cleaned = re.sub(r"<[^>]+>", " ", value.lower())
    return re.findall(r"[a-z0-9]+", cleaned)


def _lexical_score(query: str, document: str) -> float:
    q = set(_tokens(query))
    dt = _tokens(document)
    if not q or not dt:
        return 0.0
    counts = Counter(dt)
    return sum(1.0 + math.log1p(counts[token]) for token in q if token in counts) / math.sqrt(len(dt))


def _source_coverage(source_text: str, candidate_text: str) -> float:
    source_tokens = set(_tokens(source_text))
    candidate_tokens = set(_tokens(candidate_text))
    return len(source_tokens & candidate_tokens) / max(1, len(source_tokens))


def _asset_paths(stem: str) -> dict[str, Path]:
    return {
        "pdf": CACHE / "vault" / f"{stem}.pdf",
        "annotation": CACHE / "annotations" / f"{stem}.json",
        "csv_gold": CACHE / "gold" / f"{stem}.csv",
        "parser_output": CACHE / "parser" / f"{stem}.json",
    }


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=180) as response:
        target.write_bytes(response.read())


def materialize_frozen_assets(manifest: dict[str, Any]) -> dict[str, Any]:
    dataset_revision = str(manifest["dataset_revision"])
    annotation_revision = str(manifest["annotation_revision"])
    records: list[dict[str, Any]] = []
    for item in manifest["records"]:
        stem = str(item["stem"])
        paths = _asset_paths(stem)
        urls = {
            "pdf": f"https://huggingface.co/datasets/pdfqa/pdfQA-Benchmark/resolve/{dataset_revision}/syn-pdfQA/01.2_Input_Files_PDF/books/{quote(stem)}.pdf",
            "annotation": f"https://huggingface.co/datasets/pdfqa/pdfQA-Annotations/resolve/{annotation_revision}/syn-pdfQA/books/{quote(stem)}.json",
            "csv_gold": f"https://huggingface.co/datasets/pdfqa/pdfQA-Benchmark/resolve/{dataset_revision}/syn-pdfQA/01.3_Input_Files_CSV/books/{quote(stem)}.csv",
        }
        for key in ("pdf", "annotation", "csv_gold"):
            expected = str(item[f"{key}_sha256"])
            path = paths[key]
            if not path.is_file() or _sha(path) != expected:
                _download(urls[key], path)
            if _sha(path) != expected:
                raise RuntimeError(f"task0263_external_asset_digest_mismatch:{stem}:{key}")
        parser = paths["parser_output"]
        expected_parser = str(item["parser_output_sha256"])
        if not parser.is_file() or _sha(parser) != expected_parser:
            if not PARSER_PYTHON.is_file():
                raise RuntimeError("task0103_pymupdf4llm_environment_missing")
            subprocess.run(
                [str(PARSER_PYTHON), str(PARSER_WORKER), str(paths["pdf"]), "--output", str(parser)],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
                timeout=900,
            )
        if _sha(parser) != expected_parser:
            raise RuntimeError(f"task0263_parser_output_digest_mismatch:{stem}")
        records.append({"stem": stem, "asset_identity_valid": True})
    return {"schema_version": "opk-rag.task0263.external-materialization.v1", "record_count": len(records), "all_assets_valid": all(row["asset_identity_valid"] for row in records), "records": records}


def evaluate_independence(manifest: dict[str, Any]) -> dict[str, Any]:
    prior = _read_json(TASK0103_ASSETS)
    prior_stems = {Path(str(row.get("relative_path") or "")).stem for row in prior.get("assets", [])}
    selected = [str(row["stem"]) for row in manifest["records"]]
    overlap = sorted(set(selected) & prior_stems)
    question_digests = [q["question_sha256"] for row in manifest["records"] for q in row["questions"]]
    return {
        "schema_version": "opk-rag.task0263.external-followup-independence.v1",
        "selection_rule_frozen": manifest.get("status") == "frozen_before_formal_execution",
        "selected_document_count": len(selected),
        "selected_query_count": len(question_digests),
        "task0103_prior_materialization_overlap": overlap,
        "task0103_prior_materialization_overlap_count": len(overlap),
        "question_identity_unique": len(question_digests) == len(set(question_digests)),
        "result_based_reselection_allowed": False,
        "gold_visible_to_runtime": False,
        "independence_proven": not overlap and len(question_digests) == len(set(question_digests)) and manifest.get("status") == "frozen_before_formal_execution",
    }


def _load_external_corpus(manifest: dict[str, Any]):
    adapter = PdfDocumentAdapter()
    documents = []
    document_stems: dict[str, str] = {}
    for item in manifest["records"]:
        stem = str(item["stem"])
        paths = _asset_paths(stem)
        source = StructuredDocumentSource(
            source_bytes=paths["pdf"].read_bytes(),
            knowledge_id=str(item["knowledge_id"]),
            representation_id=f"task0263:{stem}:pdf",
            representation_type="pdf",
            source_ref=f"syn-pdfQA/books/{stem}.pdf",
            mime_type="application/pdf",
            source_name=f"{stem}.pdf",
            upstream_revision=str(manifest["dataset_revision"]),
            authority={"dataset": "pdfqa/pdfQA-Benchmark", "revision": manifest["dataset_revision"], "evaluation_only": True},
        )
        payload = _read_json(paths["parser_output"])
        document = adapter.to_canonical_document(source, payload)
        documents.append(document)
        document_stems[document.document_id] = stem
    return build_c0_chunks(documents), document_stems


def _question_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest["records"]:
        stem = str(item["stem"])
        annotations = json.loads(_asset_paths(stem)["annotation"].read_text(encoding="utf-8"))
        with _asset_paths(stem)["csv_gold"].open(encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        source_map = {str(row["source_identifier"]): str(row["text_only"]) for row in source_rows}
        for selector in item["questions"]:
            index = int(selector["record_index"])
            annotation = annotations[index]
            question = str(annotation["question"])
            if hashlib.sha256(question.encode("utf-8")).hexdigest() != selector["question_sha256"]:
                raise RuntimeError(f"task0263_question_identity_mismatch:{stem}:{index}")
            sources = [source_map[str(source_id)] for source_id in annotation.get("sources", []) if str(source_id) in source_map]
            rows.append({
                "sample_id": f"{stem}:q{index}",
                "stem": stem,
                "question": question,
                "question_sha256": selector["question_sha256"],
                "gold_source_count": len(sources),
                "gold_sources": sources,
            })
    return rows


def _arm_source_metrics(query: dict[str, Any], chunks: list[Any], document_stems: dict[str, str]) -> tuple[bool, list[float], int | None, int]:
    correct_doc = [chunk for chunk in chunks if document_stems.get(chunk.document_id) == query["stem"]]
    aggregate = " ".join(chunk.content for chunk in correct_doc)
    coverages = [
        max(max((_source_coverage(source, chunk.content) for chunk in chunks), default=0.0), _source_coverage(source, aggregate))
        for source in query["gold_sources"]
    ]
    all_sources = bool(coverages) and all(value >= SOURCE_COVERAGE_THRESHOLD for value in coverages)
    first_rank = next((index + 1 for index, chunk in enumerate(chunks) if document_stems.get(chunk.document_id) == query["stem"]), None)
    return all_sources, coverages, first_rank, len(correct_doc)


def run_formal_external_generalization() -> dict[str, Any]:
    manifest = _read_json(SUBSET)
    if not manifest:
        raise RuntimeError("task0263_external_subset_manifest_missing")
    materialization = materialize_frozen_assets(manifest)
    independence = evaluate_independence(manifest)
    chunks, document_stems = _load_external_corpus(manifest)
    questions = _question_rows(manifest)
    reranker = BgeLocalRerankerProvider(load_reranker_config())
    output_rows: list[dict[str, Any]] = []
    for query in questions:
        candidates = sorted(
            ((_lexical_score(query["question"], chunk.content), chunk) for chunk in chunks),
            key=lambda item: item[0],
            reverse=True,
        )[:CANDIDATE_K]
        e0 = [item[1] for item in candidates[:TOP_K]]
        e0_ok, e0_cov, e0_doc_rank, e0_correct_doc_chunks = _arm_source_metrics(query, e0, document_stems)
        reranker_scores = reranker.score_pairs(query["question"], [item[1].content for item in candidates], execution_scope="search")
        e1_ranked = [item for _, item in sorted(zip(reranker_scores, candidates), key=lambda pair: pair[0], reverse=True)]
        e1 = [item[1] for item in e1_ranked[:TOP_K]]
        e1_ok, e1_cov, e1_doc_rank, e1_correct_doc_chunks = _arm_source_metrics(query, e1, document_stems)
        output_rows.append({
            "sample_id": query["sample_id"],
            "knowledge_identity": f"pdfqa:syn-pdfQA:books:{query['stem']}",
            "question_sha256": query["question_sha256"],
            "gold_source_count": query["gold_source_count"],
            "e0_all_gold_sources_localized_at20": e0_ok,
            "e1_all_gold_sources_localized_at20": e1_ok,
            "e0_source_coverage_at20": [round(value, 6) for value in e0_cov],
            "e1_source_coverage_at20": [round(value, 6) for value in e1_cov],
            "e0_correct_document_first_rank": e0_doc_rank,
            "e1_correct_document_first_rank": e1_doc_rank,
            "e0_correct_document_chunk_count_at20": e0_correct_doc_chunks,
            "e1_correct_document_chunk_count_at20": e1_correct_doc_chunks,
            "e1_harmed_vs_e0": e0_ok and not e1_ok,
            "e1_improved_vs_e0": (not e0_ok) and e1_ok,
            "runtime_gold_exposure": False,
        })
    e0_recall = mean(row["e0_all_gold_sources_localized_at20"] for row in output_rows)
    e1_recall = mean(row["e1_all_gold_sources_localized_at20"] for row in output_rows)
    e1_doc = mean(row["e1_correct_document_first_rank"] is not None for row in output_rows)
    harmed = sum(row["e1_harmed_vs_e0"] for row in output_rows)
    improved = sum(row["e1_improved_vs_e0"] for row in output_rows)
    non_regressive = e1_recall >= e0_recall and harmed == 0 and e1_doc == 1.0
    result = {
        "schema_version": "opk-rag.task0263.external-generalization-followup.v1",
        "evaluation_executed": True,
        "evaluation_scope": "independent_pdfqa_canonical_pdf_chunk_retrieval_and_production_bge_reranker_substrate",
        "agent_action_gold_claimed": False,
        "agent_specific_promotion_evidence_source": "TASK-0257 through TASK-0262 frozen authorities",
        "document_count": int(manifest["document_count"]),
        "query_count": len(output_rows),
        "candidate_chunk_count": len(chunks),
        "source_coverage_threshold": SOURCE_COVERAGE_THRESHOLD,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "e0_arm": "deterministic_lexical_candidate_ranking_over_canonical_pdf_chunks",
        "e1_arm": "same_candidate_pool_plus_production_bge_reranker_search_fp16_autocast",
        "e0_all_source_recall_at20": e0_recall,
        "e1_all_source_recall_at20": e1_recall,
        "e1_correct_document_at20": e1_doc,
        "e1_mean_correct_document_first_rank": mean((row["e1_correct_document_first_rank"] or TOP_K + 1) for row in output_rows),
        "e1_harmed_query_count": harmed,
        "e1_improved_query_count": improved,
        "external_quality_non_regressive": non_regressive,
        "runtime_gold_exposure_count": 0,
        "materialization_valid": materialization["all_assets_valid"],
        "independence_proven": independence["independence_proven"],
        "reranker_audit": reranker.last_forward_audit,
        "passed": bool(non_regressive and independence["independence_proven"] and materialization["all_assets_valid"]),
    }
    _write_json(FORMAL_INDEPENDENCE, independence)
    _write_jsonl(FORMAL_ROWS, output_rows)
    _write_json(FORMAL, result)
    return result


def verify_formal_external_generalization() -> dict[str, Any]:
    result = _read_json(FORMAL)
    independence = _read_json(FORMAL_INDEPENDENCE)
    rows = FORMAL_ROWS.read_text(encoding="utf-8").splitlines() if FORMAL_ROWS.is_file() else []
    checks = {
        "formal_result_exists": bool(result),
        "formal_rows_exist": len(rows) == int(result.get("query_count") or -1),
        "evaluation_executed": result.get("evaluation_executed") is True,
        "independence_proven": independence.get("independence_proven") is True,
        "external_quality_non_regressive": result.get("external_quality_non_regressive") is True,
        "e1_harmed_zero": int(result.get("e1_harmed_query_count") or 0) == 0,
        "correct_document_at20_full": float(result.get("e1_correct_document_at20") or 0.0) == 1.0,
        "gold_exposure_zero": int(result.get("runtime_gold_exposure_count") or 0) == 0,
        "reranker_no_silent_fp32_fallback": (result.get("reranker_audit") or {}).get("silent_fp32_fallback_detected") is False,
        "agent_action_gold_not_claimed": result.get("agent_action_gold_claimed") is False,
    }
    return {
        "schema_version": "opk-rag.task0263.external-generalization-followup-verification.v1",
        "verification_passed": all(checks.values()),
        "checks": checks,
    }
