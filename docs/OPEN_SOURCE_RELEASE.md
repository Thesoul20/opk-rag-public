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

The current homepage Showcase is the accepted V2.1 presentation produced after visual-observability review and an owner-approved comprehension-QA waiver. The README uses a Git-tracked **animated WebP** as an inline autoplay preview, following the same presentation pattern as `chatgpt-swaync-inbox`; the 1080p MP4 remains outside Git history and is published as a GitHub Release asset.

- Inline README preview: `docs/assets/showcase-video-v2.1-preview.webp`
- Preview contract: animated WebP, 960×540, 5 FPS sampling, full 73-second story, SHA-256 `d0995f3988508dfc5c6623d504612a09d77724f6a4a6b9131ed00529e1dcb9ac`, 2,040,334 bytes.
- Release: `https://github.com/Thesoul20/opk-rag-public/releases/tag/v0.1.1`
- MP4 asset: `https://github.com/Thesoul20/opk-rag-public/releases/download/v0.1.1/showcase-video-v2.1-visual-delta-candidate.mp4`
- MP4 SHA-256: `b6c287c60b2c73ade7bafd52ad427543aad002bb80644472869b37460acbf48a`
- Master contract: H.264, 1920×1080, 30 FPS, 2190 frames / 73 seconds, no audio stream.

The preview is lightweight enough to remain inside the public snapshot's 5 MiB per-file policy. Historical TASK-0281 assets remain governed evidence, but they are no longer the README showcase authority.

## Release procedure

1. Finish release verification with zero release blockers.
2. Export or update the public snapshot.
3. Inspect the generated manifest and file listing.
4. Initialize/push the snapshot as the public repository or release-source tree.
5. Publish large showcase binaries separately as GitHub Release assets and record their digest/URL.
6. Create or change release tags/versions only after explicit user approval.

TASK-0282 itself does not push, rewrite development history, change repository visibility or create a GitHub Release.
