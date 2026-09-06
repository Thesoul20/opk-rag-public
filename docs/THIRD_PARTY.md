# Third-Party Components and Model Policy

The project-level MIT license applies to OPK-RAG-authored source and public demo material. It does not relicense third-party packages, models, services or datasets.

## Python / frontend dependencies

Dependencies are installed from their upstream package registries and retain their upstream licenses. OPK-RAG does not vendor their source into the public release snapshot. Review `pyproject.toml`, `uv.lock`, and `showcase-ui/package-lock.json` for the exact dependency graph.

## Models

The public release references model identifiers such as:
- `Qwen/Qwen3-Embedding-0.6B`
- `BAAI/bge-reranker-v2-m3`

Model weights are **not distributed** in this repository or public snapshot. Users must obtain models from the upstream source and comply with the model's current license/terms.

Remote OpenAI-compatible model providers are optional and subject to the provider's own service terms and data-handling policy.

## Datasets / knowledge bases

The public release does not redistribute historical external parser corpora or the author's private knowledge base. `examples/public-demo/` is project-authored synthetic demonstration material distributed under the repository MIT license.
