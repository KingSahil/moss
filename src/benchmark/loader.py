import glob
import logging
import os
from typing import Any, Dict, List
import numpy as np

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_and_chunk(self, directory: str) -> List[Dict[str, Any]]:
        """Loads markdown files from target directory and splits them with a sliding window overlap."""
        logger.info("Scanning directory: %s for markdown files", directory)
        md_files = glob.glob(os.path.join(directory, "*.md"))
        if not md_files:
            raise FileNotFoundError(f"No markdown files found in directory: {directory}")

        documents: List[Dict[str, Any]] = []
        total_chars = 0

        for path in md_files:
            filename = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                total_chars += len(content)

                start = 0
                while start < len(content):
                    end = start + self.chunk_size
                    chunk = content[start:end]
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "chunk_index": start // (self.chunk_size - self.overlap)
                        }
                    })
                    start += self.chunk_size - self.overlap

        logger.info("Loaded %d documents from %d files (total chars: %d)", len(documents), len(md_files), total_chars)
        return documents


class EmbeddingGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Lazy import to speed up initial CLI startup
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_documents(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Encodes document texts into dense vectors."""
        texts = [doc["text"] for doc in documents]
        logger.info("Generating embeddings for %d documents...", len(texts))
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a dense vector."""
        return self.model.encode(query, show_progress_bar=False, convert_to_numpy=True)
