# OPK-RAG Open-Source Release Policy

## Development repository vs public snapshot

The development repository intentionally contains extensive historical task/evaluation evidence. Some historical evidence records developer-specific paths or third-party parser experiments whose redistribution authority was not frozen. Publishing the complete development Git history is therefore not the release mechanism.

TASK-0282 defines a deterministic **Public Release Snapshot**. Run:

```bash
uv run python scripts/export_public_release.py --output dist/opk-rag-public-release
```

The exporter copies an explicit allowlist, sanitizes known developer-specific path tokens in copied text, rejects secrets/private paths/oversized binaries, and writes a machine-readable manifest. The resulting directory has no `.git` history.

## Included

- OPK-RAG runtime source and required governed helper modules imported by runtime code;
- Modern Control Center source;
- PostgreSQL migrations and public Docker Compose reference;
- public README / Quick Start / architecture assets;
- project-authored synthetic demo corpus;
- selected release evidence summaries;
- open-source policy files and public CI/smoke tests.

## Excluded

- `.env`, `.private/`, runtime storage and model caches;
- private or historical knowledge bases;
- development `source-documents/` corpus;
- historical `tasks/` and bulk `evaluation-data/`;
- unreviewed external parser raw material;
- database dumps / Qdrant storage;
- raw/final MP4 binaries (publish separately as GitHub Release assets);
- raw desktop/window inventories and hidden reasoning.

## Video

TASK-0281 produced a Git-ignored 1080p delivery asset and a 720p preview. Their checksums are recorded in the release evidence. A public GitHub Release URL is intentionally pending until explicitly created; documentation must not invent one.

## Release procedure

1. Finish TASK-0282 verification with zero release blockers.
2. Export the public snapshot.
3. Inspect the generated manifest and file listing.
4. Initialize/push the snapshot as the public repository or release-source tree.
5. Upload the TASK-0281 video as a GitHub Release asset if desired.
6. Create a release tag/version only after explicit user approval.

TASK-0282 itself does not push, rewrite development history, change repository visibility or create a GitHub Release.
