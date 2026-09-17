from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from services.vector_store.base import BaseVectorStore
import os

class FAISSStore(BaseVectorStore):
    def __init__(self, embeddings: Embeddings, index_path: str = "faiss_index"):
        self.embeddings = embeddings
        self._index_path = index_path
        self._store = self._load()

    def _load(self):
        if os.path.exists(self._index_path):
            return FAISS.load_local(self._index_path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    def add_documents(self, documents: list[Document]) -> None:
        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embeddings)
        else:
            self._store.add_documents(documents)
        self._store.save_local(self._index_path)

    def get_retriever(self, score_threshold: float | None = 0.0) -> VectorStoreRetriever:
        if self._store is None:
            raise RuntimeError("No documents indexed yet")
        if score_threshold is None:
            return self._store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return self._store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": score_threshold})

    def add_embeddings(self, texts_and_embeddings: list[tuple[str, list[float]]], metadatas: list[dict]) -> None:
        if self._store is None:
            self._store = FAISS.from_embeddings(texts_and_embeddings, self.embeddings, metadatas=metadatas)
        else:
            self._store.add_embeddings(texts_and_embeddings, metadatas=metadatas)
        self._store.save_local(self._index_path)

    def get_all_documents(self) -> list[Document]:
        if self._store is None:
            return []
        return list(self._store.docstore._dict.values())

    @property
    def store_name(self):
        return "faiss"
