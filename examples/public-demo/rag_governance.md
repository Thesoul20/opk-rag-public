# RAG Governance Notes

A retrieval candidate is not automatically evidence. OPK-RAG first gathers candidates, reranks them, and then composes a bounded evidence set. Answerability is evaluated after evidence composition. If authority is insufficient, the system can refuse rather than fabricate an answer.

Selective recovery is bounded. Structure Recovery can add nearby structured context, while Graph Recovery expands only one hop from an authoritative relationship seed. Recovered items return to the shared reranking and evidence path.
