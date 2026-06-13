import time
from typing import Any, Dict, List
import numpy as np

from benchmark.engines.base import BaseEngine

class ChromaEngine(BaseEngine):
    def __init__(self, embed_fn):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("chromadb is not installed. Install it using 'pip install chromadb'") from e
        
        self.embed_fn = embed_fn
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(
            name="blinky_benchmarks",
            metadata={"hnsw:space": "cosine"}
        )

    async def index_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> float:
        start_time = time.perf_counter()
        
        ids = [f"chunk_{i}" for i in range(len(documents))]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        embeddings_list = embeddings.tolist()
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
        
        return time.perf_counter() - start_time

    async def query(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embed_fn(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        formatted = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                doc_text = results["documents"][0][i]
                dist = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - dist
                formatted.append({"text": doc_text, "score": score})
        return formatted
