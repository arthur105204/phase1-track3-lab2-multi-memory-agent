from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .base import MemoryBackend, MemoryItem


@dataclass
class ChromaSemanticMemory(MemoryBackend):
    """
    Semantic memory stored in Chroma (persistent).
    Uses ONNX embedding function by default to avoid heavy ML deps.
    """

    name: str = "chroma"
    persist_dir: Path = Path("chroma_db")
    collection_name: str = "semantic_memory"
    embedding_model: str = "all-MiniLM-L6-v2"

    def _collection(self):
        try:
            import chromadb  # type: ignore
            from chromadb.utils import embedding_functions  # type: ignore
        except Exception as e:
            raise RuntimeError("chromadb not installed") from e

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        # Prefer ONNX embedder (no transformers/keras). Fallback to sentence-transformers if available.
        ef = None
        if hasattr(embedding_functions, "ONNXMiniLM_L6_V2"):
            ef = embedding_functions.ONNXMiniLM_L6_V2()
        elif hasattr(embedding_functions, "DefaultEmbeddingFunction"):
            ef = embedding_functions.DefaultEmbeddingFunction()
        elif hasattr(embedding_functions, "SentenceTransformerEmbeddingFunction"):
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        else:
            raise RuntimeError("No supported Chroma embedding function found")
        return client.get_or_create_collection(name=self.collection_name, embedding_function=ef)

    def read(self, user_id: str, query: str, k: int = 5) -> List[MemoryItem]:
        col = self._collection()
        res = col.query(
            query_texts=[query],
            n_results=max(1, k),
            where={"user_id": user_id},
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        items: List[MemoryItem] = []
        for doc, meta in zip(docs, metas):
            if not doc:
                continue
            items.append(MemoryItem(kind="semantic", text=str(doc), metadata=dict(meta or {})))
        return items

    def write(self, user_id: str, items: List[MemoryItem]) -> None:
        col = self._collection()
        to_add = []
        ids = []
        metas = []
        for it in items:
            if it.kind != "semantic":
                continue
            ids.append(f"{user_id}::{uuid4().hex}")
            to_add.append(it.text)
            meta = dict(it.metadata or {})
            meta["user_id"] = user_id
            metas.append(meta)
        if to_add:
            col.add(ids=ids, documents=to_add, metadatas=metas)

