"""Runtime composition for the local research API."""

from pathlib import Path

from research_assistant.api.indexer import (
    LocalPDFIndexer,
)
from research_assistant.api.service import (
    ResearchService,
)
from research_assistant.chunking.models import (
    ChunkingConfig,
)
from research_assistant.chunking.tokenizer import (
    HuggingFaceTokenizer,
)
from research_assistant.citations.service import (
    CitationService,
)
from research_assistant.embeddings.embedder import (
    SentenceTransformerEmbedder,
)
from research_assistant.embeddings.models import (
    EmbeddingConfig,
)
from research_assistant.generation.models import (
    LLMConfig,
)
from research_assistant.generation.ollama import (
    OllamaProvider,
)
from research_assistant.rag.context_builder import (
    ContextBuilder,
)
from research_assistant.rag.service import (
    RAGService,
)
from research_assistant.rag.token_counter import (
    HuggingFaceChatTokenCounter,
)
from research_assistant.retrieval.hybrid import (
    HybridRetriever,
)
from research_assistant.retrieval.lexical import (
    BM25Retriever,
)
from research_assistant.retrieval.retriever import (
    SemanticRetriever,
)
from research_assistant.retrieval.router import (
    RetrievalRouter,
)
from research_assistant.vector_store.models import (
    VectorStoreConfig,
)
from research_assistant.vector_store.qdrant_store import (
    QdrantVectorStore,
)


def build_local_service() -> ResearchService:
    """Construct the complete local research application."""

    embedding_config = EmbeddingConfig()

    embedder = SentenceTransformerEmbedder(embedding_config)

    tokenizer = HuggingFaceTokenizer(embedder.model_name)

    store = QdrantVectorStore(
        VectorStoreConfig(
            path=Path("data/vector_store/qdrant"),
            collection_name=("research_chunks_bge_small_en_v1_5"),
            embedding_model=(embedder.model_name),
            vector_size=(embedder.dimension),
        )
    )

    store.ensure_collection()

    dense_retriever = SemanticRetriever(
        embedder=embedder,
        vector_store=store,
    )

    lexical_retriever = BM25Retriever(
        corpus=store,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=(dense_retriever),
        lexical_retriever=(lexical_retriever),
    )

    retriever = RetrievalRouter(
        dense_retriever=(dense_retriever),
        lexical_retriever=(lexical_retriever),
        hybrid_retriever=(hybrid_retriever),
    )

    llm = OllamaProvider(LLMConfig())

    token_counter = HuggingFaceChatTokenCounter(model_name=("Qwen/Qwen3.5-9B"))

    context_builder = ContextBuilder(token_counter)

    rag = RAGService(
        retriever=retriever,
        context_builder=(context_builder),
        llm=llm,
    )

    indexer = LocalPDFIndexer(
        tokenizer=tokenizer,
        embedder=embedder,
        store=store,
        chunking_config=(
            ChunkingConfig(
                chunk_size_tokens=384,
                chunk_overlap_tokens=64,
            )
        ),
    )

    return ResearchService(
        indexer=indexer,
        registry=store,
        retriever=retriever,
        rag=rag,
        citation_service=(CitationService()),
        llm=llm,
    )
