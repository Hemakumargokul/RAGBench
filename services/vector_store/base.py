from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def get_retriever(self, score_threshold: float | None = 0.0) -> VectorStoreRetriever:
        """score_threshold=None requests plain top-k similarity search, bypassing
        the relevance-score-threshold filter entirely (see v3 usage in rag_service.py
        for why: the threshold is miscalibrated for OpenAI embeddings and can filter
        out the single most relevant chunk even at moderate values)."""
        pass

    @abstractmethod
    def add_embeddings(self, texts_and_embeddings: list[tuple[str, list[float]]], metadatas: list[dict]) -> None:
        pass

    @abstractmethod
    def get_all_documents(self) -> list[Document]:
        pass

    @property
    @abstractmethod
    def store_name(self) -> str:
        pass
