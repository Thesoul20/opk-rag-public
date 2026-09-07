# CLI

The project exposes one console script:

```bash
uv run opk-rag
```

## `opk-rag doctor`

`doctor` is the safe, read-only runtime health check for the Phase 2 reference runtime. By default it does not touch the network. Remote checks are opt-in and `--remote-deepseek` may incur a small API cost.

```bash
uv run opk-rag doctor
uv run opk-rag doctor --format json
uv run opk-rag doctor --remote-supabase
uv run opk-rag doctor --remote-deepseek
uv run opk-rag doctor --remote
```

Options:

* `--remote`: enable both remote Supabase and DeepSeek checks.
* `--remote-supabase`: enable only remote Supabase checks.
* `--remote-deepseek`: enable only remote DeepSeek checks.
* `--format`: `text` or `json`.

Exit codes:

* `0`: the requested checks passed or only produced warnings.
* `1`: at least one requested check failed.
* `2`: invalid CLI input or an internal doctor error.

See [DOCTOR.md](DOCTOR.md) for the full status model, layer breakdown, and safety boundary.

## `opk-rag index`

`index` scans a Markdown / Obsidian Vault and runs the complete indexing pipeline: document-state sync, Markdown chunking, local embedding persistence, deleted-document cleanup, and lexical BM25 index refresh.

```bash
uv run opk-rag index /path/to/obsidian-vault
uv run opk-rag index /path/to/obsidian-vault --format json
```

Options:

* `vault_path`: Vault directory. If omitted, `OPK_RAG_VAULT_PATH` must be set.
* `--knowledge-base-name`: optional name used when creating the Knowledge Base.
* `--knowledge-base-description`: optional description used when creating the Knowledge Base.
* `--device`: embedding device, one of `auto`, `cpu`, `mps`, or `cuda`.
* `--embedding-local-files-only`: load the pinned embedding model from local Hugging Face cache only.
* `--embedding-cache-dir`: optional Hugging Face cache folder for the embedding model.
* `--lexical-batch-size`: chunks per lexical indexing batch; default `100`.
* `--format`: `text` or `json`.

Exit codes:

* `0`: indexing completed successfully, including the all-unchanged case.
* `1`: indexing completed with file-level failures, or a runtime/database/model error occurred.
* `2`: invalid CLI input or configuration, such as missing `DATABASE_URL` or an invalid Vault path.

The command does not print database URLs, credentials, API keys, vectors, or authorization headers.

After a successful run, use the reported `knowledge_base_id` with `search`, `ask`, or `session create`.

## Search

Default vector search:

```bash
uv run opk-rag search \
  --knowledge-base-id <uuid> \
  --query "增量索引是如何工作的？"
```

Options:

* `--knowledge-base-id`: required Knowledge Base UUID.
* `--query`: required user query.
* `--top-k`: final number of chunks to return.
* `--candidate-k`: vector candidate count.
* `--rerank-top-n`: pre-rerank candidate window; default `10`.
* `--bm25-candidate-k`: BM25 candidate count.
* `--mode`: `vector`, `bm25`, or `hybrid`; default is `vector`.
* `--min-similarity`: optional minimum similarity in `[-1.0, 1.0]`.
* `--bm25-k1`: BM25 saturation parameter; default `1.2`.
* `--bm25-b`: BM25 length normalization parameter; default `0.75`.
* `--rrf-k`: Reciprocal Rank Fusion rank constant; default `60`.
* `--device`: `auto`, `cpu`, `mps`, or `cuda`.
* `--rerank`: explicitly enable the default local BGE CrossEncoder Rank-Fusion reranker.
* `--no-rerank`: explicitly disable reranking and use original retrieval ranking.
* `--reranker-device`: reranker device, one of `auto`, `cpu`, `mps`, or `cuda`.
* `--reranker-local-files-only`: load the pinned reranker revision from local Hugging Face cache only.
* `--reranker-cache-dir`: optional Hugging Face cache folder for the reranker.
* `--reranker-max-pair-tokens`: fixed Query-Document pair token limit; default `1024`.
* `--search-reranker-precision`: Search reranker precision, `fp16_autocast` or `fp32`; use `fp32` for rollback.
* `--ask-reranker-precision`: Ask reranker precision, `fp32` or `fp16_autocast`; production default remains `fp32`.
* `--context-token-budget`: Evidence Bundle context token budget; default `4096`.
* `--context-max-chunks`: maximum selected Evidence chunks; default `5`.
* `--context-max-chunks-per-document`: maximum selected Evidence chunks per document; default `2`.

## `opk-rag ask`

`ask` reuses the `search` retrieval flags, builds the current-turn Evidence Bundle, runs the deterministic Answerability MVP, generates an answer only when the current evidence is sufficient, and then runs Citation Grounding Validation before returning the answer.

For Phase 2 dogfooding, use the reference runtime settings from [REFERENCE_RUNTIME.md](REFERENCE_RUNTIME.md). The code default still points at the local loopback-compatible path, but that path is a fallback. The frozen dogfooding baseline uses remote DeepSeek.

```bash
opk-rag ask --knowledge-base-id <uuid> --query "问题" --format text
opk-rag ask --knowledge-base-id <uuid> --query "问题" --format json
```

Additional answer flags:

* `--llm-base-url`: OpenAI-compatible chat endpoint; default `http://127.0.0.1:11434/v1`.
* `--llm-model-id`: local fallback model ID; the formal reference runtime uses `OPK_RAG_LLM_MODEL=deepseek-v4-flash`.
* `--llm-timeout-seconds`: provider timeout.
* `--allow-remote-llm`: explicitly allow non-loopback LLM endpoints.
* `--format`: `text` or `json`.

Text output example:

```text
通过比较上次记录的 checksum 和当前 checksum，可以判断文件是否变化。[C1]

[C1] facts/checksum.md:10-12
```

When the current retrieval does not provide enough evidence, `ask` does not call the Answer LLM. It prints stable refusal text with `status: refused`, `refusal_reason_code`, and `answerability_reason_code`, and returns exit code 0. System/runtime failures such as provider connection errors, database failures, invalid runtime, missing model, timeout, or invalid model JSON after retry exhaustion return a non-zero exit code and do not emit `answerable=false`.

When the model generates an answer that violates the current-turn citation contract, `ask` returns a normal grounding refusal with exit code 0. The original untrusted answer is not returned. JSON includes `refusal_reason_code: ungrounded_answer` and a nested `grounding.reason_code` such as `missing_citations`, `invalid_citation_format`, `citation_out_of_range`, or `citation_set_mismatch`.

The model-facing contract uses strict JSON Schema response_format with `decision`, `answer`, `citations`, and `reason`. Public CLI JSON keeps the user-facing `answerable` field.

Public Answer JSON includes `status` (`answered` or `refused`), `answerability.status` (`answerable`, `partially_answerable`, or `unanswerable`), `answerability.answerable`, `answerability.reason_code`, `answerability.confidence`, `answerability.evidence_chunk_ids`, `answerability.evidence_score`, `answerability.evidence_count`, `answerability.considered_evidence_count`, `grounding.valid`, `grounding.reason_code`, `grounding.cited_ids`, `grounding.invalid_cited_ids`, `citations`, and the nested SearchResponse.

Default Answerability settings:

* Enabled by default via `OPK_RAG_ANSWERABILITY_ENABLED=true`.
* `OPK_RAG_ANSWERABILITY_MIN_EVIDENCE=1`.
* `OPK_RAG_ANSWERABILITY_MIN_DISTINCT_SOURCES=1`.
* `OPK_RAG_ANSWERABILITY_MIN_SCORE` is empty by default. When set, it is only applied to comparable 0..1 scores: vector similarity, or selected reranker scores when reranking is enabled. BM25 and hybrid RRF scores are not treated as probabilities.

Default Grounding settings:

* `OPK_RAG_GROUNDING_VALIDATION_ENABLED=true`.
* `OPK_RAG_GROUNDING_REQUIRE_CITATIONS=true`.
* `OPK_RAG_GROUNDING_MIN_CITATIONS=1`.
* `OPK_RAG_GROUNDING_MIN_COVERAGE` is empty by default. When set, it is a paragraph-level heuristic threshold, not semantic entailment.
* `OPK_RAG_GROUNDING_MAX_REPAIR_ATTEMPTS=0`; automatic repair is not enabled in this MVP.

JSON output shape:

```json
{
  "query": "增量索引是如何工作的？",
  "normalized_query": "增量索引是如何工作的？",
  "knowledge_base_id": "00000000-0000-0000-0000-000000000000",
  "model_id": "Qwen/Qwen3-Embedding-0.6B",
  "retrieval_mode": "hybrid",
  "query_template_version": "query-input-v1",
  "requested_top_k": 5,
  "candidate_k": 20,
  "rerank_top_n": 10,
  "candidate_count": 2,
  "vector_candidate_count": 1,
  "bm25_candidate_count": 1,
  "threshold_filtered_count": 0,
  "deduplicated_count": 0,
  "result_count": 2,
  "query_token_count": 12,
  "query_input_token_count": 31,
  "lexical_query_terms": ["增量", "索引"],
  "lexical_ready": true,
  "retrieval_degraded": false,
  "reranker_enabled": true,
  "reranker_score_semantics": "raw_cross_encoder_logit_higher_is_more_relevant_not_probability",
  "rerank_candidate_count": 10,
  "context_token_count": 120,
  "evidence_signals": {
    "top_reranker_score": 0.77,
    "second_reranker_score": 0.01,
    "top1_top2_margin": 0.76,
    "selected_context_token_count": 120
  },
  "results": []
}
```

By default, `opk-rag search`, `opk-rag ask`, and `opk-rag session ask` use the TASK-0095 Rank-Fusion reranker policy: `rank_fusion`, k `60`, lambda `0.75`. After TASK-0212, Search uses `fp16_autocast` for the BGE forward pass and Ask remains `fp32`. Set `OPK_RAG_RERANK_ENABLED=false` or pass `--no-rerank` to disable reranking; set `OPK_RAG_SEARCH_RERANKER_PRECISION=fp32` or pass `--search-reranker-precision fp32` to roll Search precision back without changing retrieval candidates or embeddings.

The CLI does not print vectors, absolute paths, database URLs, credentials, or full reranker Query-Document pairs. Reranker scores are raw logits, not percentages or probabilities.

## Conversation Sessions

Conversation commands persist turns for citation-aware follow-up questions. A session is bound to one Knowledge Base and each turn still runs fresh retrieval, Answerability, and answer validation.

```bash
opk-rag session create --knowledge-base-id <uuid> --title "增量索引"
opk-rag session ask --session-id <session-uuid> --query "增量索引是怎么判断文件发生变化的？"
opk-rag session ask --session-id <session-uuid> --query "那变化以后哪些内容会重新计算？"
opk-rag session show --session-id <session-uuid>
opk-rag session list --knowledge-base-id <uuid>
opk-rag session delete --session-id <session-uuid>
```

Additional `session ask` flags:

* `--client-request-id`: optional idempotency key scoped to the session.
* `--resolver`: `local-llm` or `heuristic`; default `local-llm`.
* `--context-max-turns`: default `6`.
* `--conversation-context-token-budget`: default `1200`.
* `--format`: `text` or `json`.

Text output example:

```text
session: 00000000-0000-0000-0000-000000000000
turn: 2
standalone_query: 增量索引如何判断文件变化？；追问：那变化以后哪些内容会重新计算？
is_followup: true

变化后会重新生成受影响文档的 Chunk，并重新计算对应 Embedding 和词法索引。[C1]

[C1] docs/incremental.md:20-32
```

JSON output contains `session`, `turn`, `resolution`, `answer`, and `idempotent_replay`. Historical citations in `session show --format json` include `source_status`.

Conversation history can help the resolver rewrite follow-up questions, but it is not current-turn evidence. A refused turn is saved with `turn_status=abstained`, an abstention reason, empty citations, and answerability metadata. The next turn can continue normally.

## Lexical Index

```bash
uv run opk-rag lexical-index \
  --knowledge-base-id <uuid> \
  --format json
```

Options:

* `--knowledge-base-id`: required Knowledge Base UUID.
* `--batch-size`: chunks per database batch; default `100`.
* `--dry-run`: report readiness without modifying the database.
* `--force-rebuild`: delete and rebuild the current lexical index for the Knowledge Base.
* `--format`: `text` or `json`.

BM25 and hybrid search require this lexical index to be ready. `opk-rag index` runs this refresh automatically; this command remains available for manual checks and rebuilds. Vector search does not require lexical readiness.
