# Experiments

This document records controlled experiments used to evaluate design choices in the RAG pipeline.

The project currently has working baselines for:

- document ingestion
- token-aware chunking
- local embedding generation

Formal retrieval experiments will begin once persistent vector storage and semantic retrieval are implemented.

Planned comparisons include:

- chunk size
- chunk overlap
- chunking strategy
- embedding model
- retrieval top-k
- dense vs hybrid retrieval
- reranking
- latency and memory use

Results will be recorded with the configuration, dataset, metrics, and observations required to reproduce each experiment.