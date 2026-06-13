from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class BaseEngine(ABC):
    """Abstract interface defining required methods for all benchmark vector stores."""

    @abstractmethod
    async def index_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> float:
        """Indexes the documents with pre-computed embeddings and returns indexing time (seconds)."""
        pass

    @abstractmethod
    async def query(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """Executes a similarity query. Returns a list of matches with text and score."""
        pass
