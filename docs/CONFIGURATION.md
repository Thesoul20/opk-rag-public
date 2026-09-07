# Configuration Contract

> Phase 2 reference runtime contract for `opk-rag`.

```text
.env
→ project configuration loader
→ CLI / scripts / tests
→ Supabase + DeepSeek + Vault + Embedding
```

## 1. Overview

The current reference runtime depends on a local untracked `.env` file in the repository root. That file stores the real Vault, Supabase, and DeepSeek settings for dogfooding.

`.env.example` is the committed field template. It must stay free of real secrets and real private hostnames.

## 2. Initialize

```bash
cp .env.example .env
```

Then fill the local Vault path, PostgreSQL connection, Supabase API values, and DeepSeek-compatible provider settings in your own environment.

## 3. Loading Rules

The shared loader is `opk_rag.runtime.dotenv.load_project_env()`.

Current supported entry points call it explicitly:

* `opk-rag` CLI startup;
* `tests/conftest.py` session initialization;
* evaluation and operational helper scripts that need project runtime configuration.

Precedence is consistent:

* existing shell variables win;
* repository-root `.env` only fills missing keys;
* code defaults still apply when both are absent;
* no code path in this repository uses `override=True`.

`.env.example` is not loaded automatically.

Working-directory behavior:

* the repository root is derived from project code, not the caller's current working directory;
* the loader does not scan arbitrary parent directories for `.env`;
* running from the repository root, a subdirectory, an external directory, or a pytest temp cwd all resolves the same repository-root `.env`.

Missing file behavior:

* the loader returns a structured not-found result when `.env` is absent;
* it does not create a file or print an error;
* CLI and scripts can still run from shell-only configuration.

The new `opk-rag doctor` command uses the same shared loader and reports whether the repository-root `.env` was found and loaded without printing secret values.

Doctor and the configuration tests classify every key they report using only these sources: `shell`, `dotenv`, `default`, `compatibility_alias`, and `missing`. They never print the values themselves.

## 4. Categories

The contract below is grouped into:

* Vault
* Supabase / PostgreSQL
* Embedding
* Retrieval
* Reranker
* DeepSeek / answer generation
* Answerability / grounding
* Conversation
* Evaluation and test switches
* Local provider helper settings

## 5. Variable Table

### Vault / Database / Supabase

| Variable | Required | Secret | Default | Used by | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `OPK_RAG_VAULT_PATH` | Yes for `index` / vault workflows | No | unset | CLI, docs, tests | Local Obsidian vault root. |
| `DATABASE_URL` | Yes for live DB / retrieval / indexing / evaluation | Yes | unset | CLI, scripts, tests, db helpers | PostgreSQL connection URL. |
| `SUPABASE_URL` | Conditional for remote Supabase API tests | No | unset | `load_supabase_api_config`, remote tests | Supabase project URL. |
| `SUPABASE_PUBLISHABLE_KEY` | Conditional, preferred API key | Yes | unset | `load_supabase_api_config`, docs, tests | Preferred non-secret API key for Supabase API flows. |
| `SUPABASE_ANON_KEY` | Conditional alias | Yes | unset | `load_supabase_api_config`, remote tests | Legacy alias for the publishable key. |
| `SUPABASE_SERVICE_ROLE_KEY` | Conditional for service-role RPC / API tests | Yes | unset | `load_supabase_api_config`, remote tests | Trusted server-side Supabase key. |

### Embedding

| Variable | Required | Secret | Default | Used by | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `OPK_RAG_EMBEDDING_PROVIDER` | No | No | `local_qwen` | embedding config | Embedding provider identifier. |
| `OPK_RAG_EMBEDDING_MODEL` | No | No | `Qwen/Qwen3-Embedding-0.6B` | embedding config | Formal embedding model name. |
| `OPK_RAG_EMBEDDING_MODEL_NAME` | Alias | No | legacy alias for `OPK_RAG_EMBEDDING_MODEL` | embedding config | Backward-compatible embedding model alias. |
| `OPK_RAG_EMBEDDING_MODEL_REVISION` | No | No | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | embedding config | Pinned embedding revision. |
| `OPK_RAG_EMBEDDING_DIMENSION` | No | No | `1024` | embedding config | Embedding vector dimension. |
| `OPK_RAG_EMBEDDING_BATCH_SIZE` | No | No | `8` | embedding config | Embedding batch size. |
| `OPK_RAG_EMBEDDING_MAX_INPUT_TOKENS` | No | No | `8192` | embedding config | Max tokens per embedding input. |
| `OPK_RAG_EMBEDDING_DEVICE` | No | No | `auto` | embedding config, model smoke | Embedding device selector. |
| `OPK_RAG_EMBEDDING_NORMALIZE` | No | No | `true` | embedding config | Normalize embeddings. |
| `OPK_RAG_EMBEDDING_LOCAL_FILES_ONLY` | No | No | `false` | embedding config, model smoke | Disallow downloads. |
| `OPK_RAG_EMBEDDING_CACHE_DIR` | No | No | unset | embedding config | Hugging Face cache override. |
| `OPK_RAG_EMBEDDING_INPUT_TEMPLATE_VERSION` | No | No | `embedding-input-v1` | embedding config | Template version tag. |
| `OPK_RAG_SCHEMA_VERSION` | No | No | `2026-07-17` | embedding config | Index schema version tag. |
| `OPK_RAG_PARSER_VERSION` | No | No | `markdown-parser-v1` | embedding config | Parser version tag. |
| `OPK_RAG_CHUNKING_VERSION` | No | No | `markdown-chunker-v1` | embedding config | Chunking version tag. |
| `OPK_RAG_EMBEDDING_DISTANCE_METRIC` | No | No | `cosine` | embedding config | Distance metric contract. |

### Retrieval

| Variable | Required | Secret | Default | Used by | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `OPK_RAG_SEARCH_MODE` | No | No | `vector` | search config, CLI, scripts | Retrieval mode. |
| `OPK_RAG_SEARCH_TOP_K` | No | No | `5` | search config | Final top-k. |
| `OPK_RAG_SEARCH_CANDIDATE_K` | No | No | `20` | search config | Vector candidate count. |
| `OPK_RAG_SEARCH_BM25_CANDIDATE_K` | No | No | `20` | search config | BM25 candidate count. |
| `OPK_RAG_SEARCH_MIN_SIMILARITY` | No | No | unset | search config | Optional similarity threshold. |
| `OPK_RAG_SEARCH_DEDUPLICATE` | No | No | `true` | search config | Deduplicate overlapping evidence. |
| `OPK_RAG_BM25_K1` | No | No | `1.2` | search config | BM25 parameter. |
| `OPK_RAG_BM25_B` | No | No | `0.75` | search config | BM25 parameter. |
| `OPK_RAG_RRF_K` | No | No | `60` | search config | Reciprocal rank fusion constant. |
| `OPK_RAG_RRF_VECTOR_WEIGHT` | No | No | `1` | search config | Vector weight in RRF. |
| `OPK_RAG_RRF_BM25_WEIGHT` | No | No | `1` | search config | BM25 weight in RRF. |
| `OPK_RAG_QUERY_MAX_TOKENS` | No | No | `512` | search config | Query token cap. |
| `OPK_RAG_QUERY_TEMPLATE_VERSION` | No | No | `query-input-v1` | search config | Query template version tag. |
| `OPK_RAG_RERANK_ENABLED` | No | No | `true` | search config, CLI, eval | Enable the default rank-fusion reranker. Set `false` to restore original retrieval ranking. |
| `OPK_RAG_RERANKER_POLICY` | No | No | `rank_fusion` | search config, runtime v2, eval | Default reranker promotion policy. |
| `OPK_RAG_RANK_FUSION_K` | No | No | `60` | search config, runtime v2, eval | Frozen Rank Fusion k for reranker promotion v1. |
| `OPK_RAG_RANK_FUSION_LAMBDA` | No | No | `0.75` | search config, runtime v2, eval | Frozen Rank Fusion lambda for reranker promotion v1. |
| `OPK_RAG_RERANK_TOP_N` | No | No | `10` | search config | Rerank candidate count. |
| `OPK_RAG_CONTEXT_TOKEN_BUDGET` | No | No | `4096` | search config | Evidence context token budget. |
| `OPK_RAG_CONTEXT_MAX_CHUNKS` | No | No | `5` | search config | Evidence chunk cap. |
| `OPK_RAG_CONTEXT_MAX_CHUNKS_PER_DOCUMENT` | No | No | `2` | search config | Evidence per-document cap. |

### Reranker

| Variable | Required | Secret | Default | Used by | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `OPK_RAG_RERANKER_PROVIDER` | No | No | `local_bge_cross_encoder` | reranker config | Reranker provider identifier. |
| `OPK_RAG_RERANKER_MODEL_NAME` | No | No | `BAAI/bge-reranker-v2-m3` | reranker config | Reranker model name. |
| `OPK_RAG_RERANKER_MODEL_REVISION` | No | No | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | reranker config | Pinned reranker revision. |
| `OPK_RAG_RERANKER_BATCH_SIZE` | No | No | `8` | reranker config | Reranker batch size. |
| `OPK_RAG_RERANKER_MAX_PAIR_TOKENS` | No | No | `1024` | reranker config | Pair token limit. |
| `OPK_RAG_RERANKER_DEVICE` | No | No | `auto` | reranker config | Device selector. |
| `OPK_RAG_RERANKER_LOCAL_FILES_ONLY` | No | No | `false` | reranker config, smoke | Disallow downloads. |
| `OPK_RAG_RERANKER_CACHE_DIR` | No | No | unset | reranker config | Cache override. |
| `OPK_RAG_RERANKER_INPUT_TEMPLATE_VERSION` | No | No | `reranker-pair-v1` | reranker config | Template version tag. |
| `OPK_RAG_SEARCH_RERANKER_PRECISION` | No | No | `fp16_autocast` | reranker config, CLI, search runtime | Search-scope reranker precision. Allowed values: `fp32`, `fp16_autocast`. Set `fp32` for no-code rollback. |
| `OPK_RAG_ASK_RERANKER_PRECISION` | No | No | `fp32` | reranker config, CLI, ask runtime | Ask-scope reranker precision. Must remain `fp32` until local generation/KV-cache authority exists. |

Reranker precision precedence is CLI explicit argument, then environment variable, then safe default. The matching CLI overrides are `--search-reranker-precision` and `--ask-reranker-precision`.

Search FP32 rollback:

```bash
OPK_RAG_SEARCH_RERANKER_PRECISION=fp32 opk-rag search --knowledge-base-id <uuid> --query "<query>" --format json
uv run python scripts/verify_task0212_search_scoped_fp16_autocast_reranker_production_integration.py
```

Rollback does not require reindexing, rebuilding the Qdrant collection, or regenerating embeddings. Success criteria: `rollback_path_available=true`, `rollback_replay_valid=true`, `rollback_output_equivalence=true`, and `search_autocast_enabled=false` under the rollback config.

### DeepSeek / Answer generation

| Variable | Required | Secret | Default | Used by | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `OPK_RAG_ANSWER_PROVIDER` | No | No | `openai_compatible_local_chat` | answer config, CLI, scripts | Answer provider identifier. |
| `OPK_RAG_LLM_BASE_URL` | Conditional | No | `http://127.0.0.1:11434/v1` | answer config, provider, scripts | OpenAI-compatible endpoint URL. |
| `OPK_RAG_LLM_MODEL` | No | No | `deepseek-v4-flash` | answer config, provider, scripts | Formal DeepSeek answer model identifier. |
| `OPK_RAG_LLM_MODEL_ID` | Alias | No | legacy alias for `OPK_RAG_LLM_MODEL` | answer config, provider, scripts | Backward-compatible answer model alias. |
| `OPK_RAG_LLM_MODEL_REVISION` | No | No | unset | answer config, provider, scripts | Model revision metadata. |
| `OPK_RAG_LLM_MODEL_LICENSE` | No | No | `Apache-2.0` | answer config | Model license metadata. |
| `OPK_RAG_LLM_TIMEOUT_SECONDS` | No | No | `60` | answer config, provider | HTTP timeout. |
| `OPK_RAG_LLM_TEMPERATURE` | No | No | `0` | answer config, provider | Sampling temperature. |
| `OPK_RAG_LLM_TOP_P` | No | No | `1` | answer config, provider | Nucleus sampling. |
| `OPK_RAG_LLM_TOP_K` | No | No | unset | answer config, provider | Optional top-k sampling. |
| `OPK_RAG_LLM_REPETITION_PENALTY` | No | No | `1` | answer config, provider | Repetition penalty. |
| `OPK_RAG_LLM_MAX_OUTPUT_TOKENS` | No | No | `1024` | answer config, provider | Completion token cap. |
| `OPK_RAG_LLM_RESPONSE_FORMAT` | No | No | `json_schema` | answer config, provider | Response format contract. |
| `OPK_RAG_LLM_THINKING_MODE` | No | No | unset | answer config, provider | Provider-specific thinking flag. |
| `OPK_RAG_LLM_SEED` | No | No | unset | answer config, provider | Optional sampling seed. |
| `OPK_RAG_LLM_STOP` | No | No | unset | answer config, provider | NUL-delimited stop sequences. |
| `OPK_RAG_LLM_MAX_RETRIES` | No | No | `1` | answer config, provider | Provider retry count. |
| `OPK_RAG_LLM_ALLOW_REMOTE` | No | No | `false` | answer config, provider | Allow non-loopback endpoints. |
| `OPK_RAG_LLM_API_KEY` | Conditional | Yes | unset | answer provider, scripts | Authorization header token. |
| `OPK_RAG_ANSWER_PROMPT_VERSION` | No | No | `answer-prompt-v4` | answer config, scripts | Prompt contract version. |
| `OPK_RAG_ANSWER_OUTPUT_SCHEMA_VERSION` | No | No | `answer-response-v3` | answer config, scripts | Output schema version. |

The DeepSeek / answer generation path treats `OPK_RAG_LLM_MODEL` as the formal field. `OPK_RAG_LLM_MODEL_ID` remains a compatibility alias only. The local Qwen runtime remains available as a fallback / test / historical path, but it is not the formal DeepSeek baseline.

### Answerability / grounding

| Variable | Required | Secret | Default | Used by | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `OPK_RAG_ANSWERABILITY_ENABLED` | No | No | `true` | answer config, CLI, tests | Enable deterministic answerability gate. |
| `OPK_RAG_ANSWERABILITY_MIN_EVIDENCE` | No | No | `1` | answer config, CLI, tests | Minimum evidence count. |
| `OPK_RAG_ANSWERABILITY_MIN_DISTINCT_SOURCES` | No | No | `1` | answer config | Minimum distinct sources. |
| `OPK_RAG_ANSWERABILITY_MIN_SCORE` | No | No | unset | answer config | Optional comparable-score threshold. |
| `OPK_RAG_ANSWER_MIN_EVIDENCE_ITEMS` | Alias | No | falls back to `OPK_RAG_ANSWERABILITY_MIN_EVIDENCE` | answer config | Legacy alias; prefer `OPK_RAG_ANSWERABILITY_MIN_EVIDENCE`. |
| `OPK_RAG_ANSWER_MIN_CONTEXT_TOKENS` | No | No | `1` | answer config | Minimum context token threshold. |
| `OPK_RAG_GROUNDING_VALIDATION_ENABLED` | No | No | `true` | answer config, CLI, tests | Enable grounding validation. |
| `OPK_RAG_GROUNDING_REQUIRE_CITATIONS` | No | No | `true` | answer config, CLI, tests | Require citations for grounded answers. |
| `OPK_RAG_GROUNDING_MIN_CITATIONS` | No | No | `1` | answer config | Minimum citation count. |
| `OPK_RAG_GROUNDING_MIN_COVERAGE` | No | No | unset | answer config | Optional paragraph coverage threshold. |
| `OPK_RAG_GROUNDING_MAX_REPAIR_ATTEMPTS` | No | No | `0` | answer config | Repair budget; current default is no repair. |

### Conversation

| Variable | Required | Secret | Default | Used by | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `OPK_RAG_CONVERSATION_CONTEXT_MAX_TURNS` | No | No | `6` | answer config, CLI | Follow-up context turn cap. |
| `OPK_RAG_CONVERSATION_CONTEXT_TOKEN_BUDGET` | No | No | `1200` | answer config, CLI | Follow-up context token budget. |

### Evaluation / test switches

| Variable | Required | Secret | Default | Used by | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `OPK_RAG_INTEGRATION_TESTS` | No | No | `0` | `tests/integration` | Enable destructive integration tests. |
| `SUPABASE_REMOTE_TESTS` | No | No | `0` | `tests/remote_supabase` | Enable remote Supabase tests. |
| `OPK_RAG_MODEL_TESTS` | No | No | `0` | model smoke tests | Enable embedding smoke tests. |
| `OPK_RAG_RERANKER_MODEL_TESTS` | No | No | `0` | model smoke tests | Enable reranker smoke tests. |
| `OPK_RAG_ANSWER_MODEL_TESTS` | No | No | `0` | model smoke tests, `evaluate_answers.py` | Enable answer-provider smoke / evaluation. |
| `OPK_RAG_CONVERSATION_MODEL_TESTS` | No | No | `0` | model smoke tests | Enable conversation model smoke tests. |
| `HF_HUB_OFFLINE` | No | No | unset | smoke tests, scripts | Force offline Hugging Face usage. |
| `TRANSFORMERS_OFFLINE` | No | No | unset | smoke tests, scripts | Force offline Transformers usage. |
| `OPK_RAG_ANSWERABILITY_DATASET_VERSION` | No | No | `answerability-eval.v1` | evaluation scripts | Evaluation dataset version tag. |
| `OPK_RAG_EVALUATION_DATABASE_SCOPE` | No | No | `source-documents` | evaluation scripts | Evaluation database scope label. |
| `OPK_RAG_EVALUATION_KNOWLEDGE_BASE_ID` | No | No | unset | evaluation scripts | Optional evaluation KB override. |
| `OPK_RAG_LLM_RUNTIME_NOTE` | No | No | `configured OpenAI-compatible chat runtime` | evaluation scripts | Metadata note for reports. |

### Local provider helper server

| Variable | Required | Secret | Default | Used by | Purpose |
| --- | ---: | ---: | --- | --- | --- |
| `OPK_RAG_LOCAL_CHAT_MODEL_ID` | No | No | `HuggingFaceTB/SmolLM2-135M-Instruct` | `scripts/local_openai_compatible_transformers_server.py` | Local chat model name. |
| `OPK_RAG_LOCAL_CHAT_MODEL_REVISION` | No | No | unset | helper server | Optional local chat revision. |
| `OPK_RAG_LOCAL_CHAT_HOST` | No | No | `127.0.0.1` | helper server | Bind host. |
| `OPK_RAG_LOCAL_CHAT_PORT` | No | No | `11434` | helper server | Bind port. |
| `OPK_RAG_LOCAL_CHAT_MAX_NEW_TOKENS` | No | No | `160` | helper server | Output token cap. |
| `OPK_RAG_LOCAL_CHAT_DEVICE` | No | No | `auto` | helper server | Runtime device. |
| `OPK_RAG_LOCAL_CHAT_DTYPE` | No | No | `float32` | helper server | Runtime dtype. |
| `OPK_RAG_LOCAL_CHAT_QUANTIZATION` | No | No | `none` | evaluation scripts | Metadata only. |
| `OPK_RAG_LOCAL_CHAT_MAX_INPUT_LENGTH` | No | No | `0` | helper server, scripts | Optional prompt truncation cap. |
| `OLLAMA_BIN` | No | No | unset | `scripts/serve_target_generation_model.py` | Explicit Ollama binary path. |

## 6. Alias Priority

Only the following alias groups currently exist:

* `OPK_RAG_ANSWERABILITY_MIN_EVIDENCE` is the recommended name. `OPK_RAG_ANSWER_MIN_EVIDENCE_ITEMS` remains as a legacy fallback.
* `OPK_RAG_ANSWER_MIN_CONTEXT_TOKENS` is the actual context-token setting. No separate alias exists in the current code.
* `SUPABASE_PUBLISHABLE_KEY` is the recommended non-secret Supabase API key. `SUPABASE_ANON_KEY` is accepted as a legacy compatibility alias.

## 7. Security Rules

* `.env` must stay untracked.
* `.env.example` may be committed.
* Secrets and passwords must not appear in logs, docs, or test output.
* Database URLs and Vault paths must be redacted in diagnostics.
* Diagnostics may report whether a variable exists, but not its value.
* If the agent cannot read an untracked `.env`, it must not conclude that Supabase or DeepSeek is unconfigured.
* Shell variables and `.env` variables are distinct; current loaders do not overwrite existing shell values.

## 8. Diagnosis Flow

Use this order:

1. Confirm `.env` exists.
2. Confirm the project loader actually runs for the entry point.
3. Confirm required variables are present.
4. Extract a redacted hostname from the URL or connection string.
5. Check DNS resolution.
6. Check TCP or HTTPS connectivity.
7. Check authentication.
8. Check authorization.
9. Check the database contract or provider contract.

Keep the error classes distinct:

* `configuration_missing`
* `hostname_invalid`
* `dns_resolution_failure`
* `network_connection_failure`
* `tls_failure`
* `authentication_failure`
* `authorization_failure`
* `database_contract_failure`
* `provider_failure`

Do not describe a DNS failure as a login failure.

## 9. Agent Rules

When the agent cannot access an untracked `.env`:

* it must not assert that Supabase or DeepSeek is absent;
* it should inspect `.env.example` and the loader code;
* it should ask for a local, redacted existence check if needed;
* it must not request real secret values;
* it must distinguish shell state from `.env` auto-loading.
