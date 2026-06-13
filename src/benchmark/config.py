import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    dataset_dir: str = os.getenv("BENCHMARK_DATASET_DIR", "blinky")
    embedding_model: str = os.getenv("BENCHMARK_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # Moss settings
    moss_project_id: str | None = os.getenv("MOSS_PROJECT_ID")
    moss_project_key: str | None = os.getenv("MOSS_PROJECT_KEY")
    moss_index_name: str = os.getenv("MOSS_INDEX_NAME", "blinky-benchmarks")
    
    # Pinecone settings
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "blinky-benchmarks")

    def validate(self) -> None:
        """Validates configuration parameters."""
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory '{self.dataset_dir}' does not exist.")
        
        # We don't crash here if keys are missing; engines should check their availability.

settings = Settings()
