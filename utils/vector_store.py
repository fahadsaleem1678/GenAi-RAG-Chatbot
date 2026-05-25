from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


@dataclass
class VectorStoreManager:
    """Thin wrapper around FAISS to keep vector-store behavior in one place."""

    vector_store: FAISS

    @classmethod
    def from_documents(cls, documents: list[Document], embeddings) -> "VectorStoreManager":
        vector_store = FAISS.from_documents(documents, embeddings)
        return cls(vector_store=vector_store)

    @classmethod
    def load(cls, path: str | Path, embeddings) -> "VectorStoreManager":
        vector_store = FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return cls(vector_store=vector_store)

    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(path))

    def get_retriever(self, k: int = 4):
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def similarity_search_with_score(self, query: str, k: int = 4):
        return self.vector_store.similarity_search_with_score(query, k=k)

    def all_documents(self) -> list[Document]:
        store = self.vector_store.docstore
        ids = list(self.vector_store.index_to_docstore_id.values())
        return [store.search(i) for i in ids]
