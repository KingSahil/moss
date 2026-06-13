import time
from typing import Any, Dict, List
import numpy as np

from benchmark.engines.base import BaseEngine

class PineconeEngine(BaseEngine):
    def __init__(self, api_key: str, index_name: str, dimension: int, embed_fn):
        try:
            from pinecone import Pinecone, ServerlessSpec
            self.ServerlessSpec = ServerlessSpec
        except ImportError as e:
            raise ImportError("pinecone-client is not installed. Install it using 'pip install pinecone-client'") from e
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.embed_fn = embed_fn
        
        # Verify/create Pinecone serverless index
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=self.ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Block until index is fully initialized
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
                
        self.index = self.pc.Index(self.index_name)

    async def index_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> float:
        start_time = time.perf_counter()
        
        # Delete existing data to start fresh
        try:
            self.index.delete(delete_all=True)
            time.sleep(1) # Allow deletion to propagate
        except Exception:
            pass
            
        upsert_data = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            upsert_data.append((
                f"chunk_{i}",
                emb.tolist(),
                {"text": doc["text"], "source": doc["metadata"]["source"]}
            ))
            
        # Batched upserts (batches of 100)
        batch_size = 100
        for idx in range(0, len(upsert_data), batch_size):
            batch = upsert_data[idx:idx + batch_size]
            self.index.upsert(vectors=batch)
            
        return time.perf_counter() - start_time

    async def query(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embed_fn(query_text)
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True
        )
        
        formatted = []
        for match in results.matches:
            formatted.append({
                "text": match.metadata.get("text", "") if match.metadata else "",
                "score": match.score
            })
        return formatted
