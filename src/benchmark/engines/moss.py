import asyncio
import time
from typing import Any, Dict, List
import numpy as np

from benchmark.engines.base import BaseEngine

class MossEngine(BaseEngine):
    def __init__(self, project_id: str, project_key: str, index_name: str):
        try:
            from moss import MossClient, DocumentInfo, QueryOptions
            self.DocumentInfo = DocumentInfo
            self.QueryOptions = QueryOptions
        except ImportError as e:
            raise ImportError("moss SDK is not installed. Install it using 'pip install moss'") from e
        
        self.client = MossClient(project_id, project_key)
        self.index_name = index_name

    async def clean_existing_index(self) -> None:
        """Deletes the index if it already exists."""
        try:
            await self.client.delete_index(self.index_name)
            await asyncio.sleep(1)
        except Exception:
            pass

    async def index_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> float:
        # Note: Moss handles embeddings internally (embedded in SDK),
        # so we ignore the precomputed embeddings parameter.
        start_time = time.perf_counter()
        
        docs_to_add = [self.DocumentInfo(id=f"chunk_{i}", text=doc["text"]) for i, doc in enumerate(documents)]
        
        await self.client.create_index(self.index_name, docs_to_add)
        indexing_duration = time.perf_counter() - start_time
        
        # Load the index into local RAM
        await self.client.load_index(self.index_name)
        
        return indexing_duration

    async def query(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        results = await self.client.query(
            self.index_name,
            query_text,
            self.QueryOptions(top_k=k)
        )
        
        formatted = []
        for doc in results.docs:
            formatted.append({"text": doc.text, "score": doc.score})
        return formatted
