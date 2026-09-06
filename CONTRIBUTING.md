# Contributing to OPK-RAG

OPK-RAG is a local-first knowledge RAG engineering project. Contributions should preserve its authority boundaries rather than add hidden alternate runtime paths.

## Development setup

```bash
uv sync --dev
uv run opk-rag --help
uv run pytest tests/public -q
```

For the Control Center:

```bash
cd showcase-ui
npm ci
npm run typecheck
npm run build
```

## Pull requests

Keep changes scoped and explain:
- what runtime or presentation surface changes;
- whether Retrieval / Agent / Graph / Evidence / Answer authority changes;
- what tests prove the change;
- whether new network access, Provider calls, data redistribution or credentials are required.

Never commit private knowledge-base content, `.env`, credentials, model weights, Qdrant/PostgreSQL runtime state, raw desktop recordings or hidden reasoning. New public demo material must have explicit redistribution provenance.

For bug reports or security issues, follow `SECURITY.md`.
