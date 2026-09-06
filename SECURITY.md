# Security Policy

## Scope

OPK-RAG processes local knowledge-base content and can optionally call remote OpenAI-compatible Providers. Treat Vault content, database contents, Provider credentials, Runtime Trace payloads and local filesystem paths as potentially sensitive.

## Safe defaults

- Local filesystem / PostgreSQL / Qdrant are the normal trust boundary.
- Remote LLM access is opt-in (`OPK_RAG_LLM_ALLOW_REMOTE=true`).
- API keys and database credentials belong in local `.env` files only.
- The public repository must contain the system, not the author's private knowledge base.
- Runtime Trace must not expose Chain-of-Thought, private prompts, Provider scratchpads or secrets.

## Reporting a vulnerability

For a public GitHub repository, use GitHub's **Security Advisories → Report a vulnerability** flow rather than opening a public issue containing exploit details or credentials. If that channel is unavailable, open a minimal issue asking for a private contact channel without including the sensitive details.

## Non-security issues

Normal bugs, documentation problems and feature requests can use regular GitHub Issues.
