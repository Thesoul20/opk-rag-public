# Vector and Relational Authority

OPK-RAG uses Qdrant as the production vector authority for vector candidate retrieval. PostgreSQL remains the relational authority for documents, chunks, provenance, graph lifecycle and conversation state. PostgreSQL with pgvector is retained as an explicit rollback vector backend.

A reachable Qdrant service does not silently change authority. Backend selection is explicit and observable.
