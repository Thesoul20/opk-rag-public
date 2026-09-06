# OPK-RAG Quick Start

This guide is the shortest public path from a clean clone to a local OPK-RAG runtime. It does not require private knowledge-base content or a paid LLM Provider.

## 1. Prerequisites

- Python 3.10+
- `uv`
- Docker + Docker Compose for the reference PostgreSQL/Qdrant services
- Node.js/npm only if you want the Modern Control Center
- optional NVIDIA GPU for faster embedding/reranking

## 2. Install

```bash
uv sync --dev
cp .env.public.example .env
```

No API key is required for install/CLI smoke. Remote generation remains disabled by default.

## 3. Start PostgreSQL and Qdrant

```bash
docker compose -f infra/public/docker-compose.yml up -d
```

The reference services bind only to loopback:
- PostgreSQL/pgvector: `127.0.0.1:55432`
- Qdrant REST: `127.0.0.1:6333`
- Qdrant gRPC: `127.0.0.1:6334`

## 4. Apply database migrations

```bash
uv run python - <<'PY'
import os
from opk_rag.runtime.dotenv import load_project_env
from opk_rag.db.migrations import apply_all_migrations
load_project_env()
apply_all_migrations(os.environ["DATABASE_URL"])
print("migrations applied")
PY
```

## 5. Health check

```bash
uv run opk-rag doctor
```

`doctor` is read-only. In the minimal public setup it may still report unconfigured Supabase / reference-LLM fields; that is truthful configuration status, not a reason to fabricate credentials. Remote network checks are not enabled unless explicitly requested.

## 6. Index the public demo corpus

```bash
uv run opk-rag index examples/public-demo --format json
```

Save the returned `knowledge_base_id`. When `OPK_RAG_VECTOR_BACKEND=qdrant`, the same command reconciles the current knowledge base into the configured Qdrant collection and reports `vector_records_written`, `vector_records_deleted`, and `vector_records_final`. A Qdrant materialization failure makes indexing fail closed instead of reporting a misleading successful index.

## 7. Search

```bash
uv run opk-rag search \
  --knowledge-base-id <knowledge-base-uuid> \
  --query "Qdrant 和 PostgreSQL 分别负责什么？" \
  --format json
```

## 8. Ask (optional generation)

The public baseline does not require a remote model. If you configure an OpenAI-compatible Provider, opt in explicitly:

```bash
OPK_RAG_LLM_ALLOW_REMOTE=true \
OPK_RAG_LLM_BASE_URL=<openai-compatible-url> \
OPK_RAG_LLM_MODEL=<model-id> \
OPK_RAG_LLM_API_KEY=<secret> \
uv run opk-rag ask \
  --knowledge-base-id <knowledge-base-uuid> \
  --query "为什么 Candidate 不等于 Evidence？" \
  --format json
```

When a Provider is unavailable or authority is insufficient, a controlled failure/refusal is valid. Do not replace it with fabricated demo output.

## 9. Modern Control Center (optional)

Terminal A:

```bash
uv run opk-rag showcase-api --host 127.0.0.1 --port 8766
```

Terminal B:

```bash
cd showcase-ui
npm ci
npm run dev
```

Then open the Vite URL printed by npm. The Control Center is a presentation/observability surface over OPK-RAG runtime authority; it does not implement a second RAG path.

## 10. Stop local services

```bash
docker compose -f infra/public/docker-compose.yml down
```

Use `down -v` only when you intentionally want to delete the local demo database/vector volumes.
