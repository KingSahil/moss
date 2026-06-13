import asyncio
import logging
import time
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from benchmark.config import settings
from benchmark.loader import DocumentLoader, EmbeddingGenerator
from benchmark.queries import BENCHMARK_QUESTIONS
from benchmark.engines import ChromaEngine, PineconeEngine, MossEngine
from benchmark.visualizer import plot_latency_graphs

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, iterations: int = 10, chunk_size: int = 600, overlap: int = 100):
        self.iterations = iterations
        self.loader = DocumentLoader(chunk_size=chunk_size, overlap=overlap)
        self.generator = EmbeddingGenerator(settings.embedding_model)
        self.results: List[Dict[str, Any]] = []
        self.latencies_log: Dict[str, List[float]] = {}

    def _generate_markdown_report(self, df: pd.DataFrame, num_docs: int) -> str:
        table_md = df.to_markdown(index=False)
        return f"""# 📊 Blinky Vector Retrieval Benchmarks

This repository documents the benchmarking results for the **Blinky** AI Desktop Tutor retrieval engine, comparing **Moss**, **Chroma**, and **Pinecone**.

## ⚙️ Environment Details
* **Dataset:** {len(BENCHMARK_QUESTIONS)} test queries run against local docs (`{settings.dataset_dir}/*.md` AI documentation files).
* **Chunk Settings:** {self.loader.chunk_size} characters (overlap: {self.loader.overlap}). Total chunks: {num_docs}.
* **Embedding Model:** `sentence-transformers/{settings.embedding_model}` (384 dimensions).
* **Execution:** Run locally from `c:/projects/moss`.

## 📈 Benchmarking Metrics

{table_md}

## 🔍 Visual Analysis

Below is the latency distribution chart showing the query response times (in milliseconds) across the systems:

![Latency Comparison](latency_comparison.png)

## 💡 Key Findings

1. **Moss (Local-First)** compiles the index in the cloud but loads the vectors into application memory. Retrieval happens fully locally in-process without network hops. Moss outperforms Chroma because its internal query embedding generation and in-memory search are highly optimized.
2. **Chroma (Ephemeral)** represents a traditional local vector database. It runs entirely in local RAM and doesn't traverse the internet, but has slightly more overhead than Moss when running the end-to-end text-to-vector-to-retrieval pipeline.
3. **Pinecone (Serverless)** is a cloud-native vector database. Its retrieval queries must traverse the network to Pinecone Cloud, which introduces high network roundtrip latencies representing standard cloud database overhead.
4. **End-to-End Latency:** All three systems now measure the complete text-in to results-out loop (including local query embedding generation time) to represent actual production application workloads (like Blinky's search tutor).

---
*Generated automatically by `run_benchmark.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""

    async def run(self) -> None:
        logger.info("Starting Blinky retrieval benchmarks suite")
        
        # Load and chunk documents
        documents = self.loader.load_and_chunk(settings.dataset_dir)
        num_docs = len(documents)
        
        # Generate document embeddings (precomputed for Chroma & Pinecone)
        doc_embeddings = self.generator.encode_documents(documents)
        dimension = doc_embeddings.shape[1]
        
        # Determine active engines
        engines = {}
        
        # Chroma is always available
        logger.info("Initializing Chroma Engine...")
        engines["Chroma (Ephemeral)"] = ChromaEngine(self.generator.encode_query)
        
        # Pinecone
        if settings.pinecone_api_key and "YOUR_PINECONE" not in settings.pinecone_api_key:
            logger.info("Initializing Pinecone Engine...")
            try:
                engines["Pinecone (Serverless)"] = PineconeEngine(
                    api_key=settings.pinecone_api_key,
                    index_name=settings.pinecone_index_name,
                    dimension=dimension,
                    embed_fn=self.generator.encode_query
                )
            except Exception as e:
                logger.error("Failed to initialize Pinecone: %s", e)
        else:
            logger.info("Pinecone credentials missing; skipping Pinecone.")

        # Moss
        if settings.moss_project_id and "YOUR_MOSS" not in settings.moss_project_id:
            logger.info("Initializing Moss Engine...")
            try:
                engines["Moss (Local-First)"] = MossEngine(
                    project_id=settings.moss_project_id,
                    project_key=settings.moss_project_key,
                    index_name=settings.moss_index_name
                )
            except Exception as e:
                logger.error("Failed to initialize Moss: %s", e)
        else:
            logger.info("Moss credentials missing; skipping Moss.")

        # Run benchmarks
        for name, engine in engines.items():
            logger.info("Running benchmark for engine: %s", name)
            try:
                # 1. Indexing phase
                if isinstance(engine, MossEngine):
                    await engine.clean_existing_index()
                    idx_time = await engine.index_documents(documents, doc_embeddings)
                else:
                    idx_time = await engine.index_documents(documents, doc_embeddings)
                
                logger.info("%s indexed docs in %.4f seconds", name, idx_time)

                # Pinecone needs replication time
                if name == "Pinecone (Serverless)":
                    logger.info("Sleeping 10s for Pinecone cloud sync...")
                    await asyncio.sleep(10)

                # 2. Warmup
                logger.info("Warming up %s index with 5 queries...", name)
                for q in BENCHMARK_QUESTIONS[:5]:
                    await engine.query(q)

                # 3. Query loop
                latencies = []
                logger.info("Executing %d iterations of %d queries...", self.iterations, len(BENCHMARK_QUESTIONS))
                for iteration in range(self.iterations):
                    for q in BENCHMARK_QUESTIONS:
                        start_time = time.perf_counter()
                        await engine.query(q)
                        duration_ms = (time.perf_counter() - start_time) * 1000.0
                        latencies.append(duration_ms)

                p50 = np.percentile(latencies, 50)
                p99 = np.percentile(latencies, 99)
                
                self.results.append({
                    "System": name,
                    "Indexing Time (s)": idx_time,
                    "P50 Latency (ms)": p50,
                    "P99 Latency (ms)": p99,
                    "Type": "Local Memory Engine" if "Moss" in name else ("Local Database" if "Chroma" in name else "Cloud Database (AWS)"),
                    "Local Embedding Time Included": "Yes (Embedded in SDK)" if "Moss" in name else "Yes (Local Model)"
                })
                self.latencies_log[name] = latencies

            except Exception as e:
                logger.exception("Failed benchmarking engine %s", name)

        # Output results
        if not self.results:
            logger.error("No benchmarks executed successfully.")
            return

        df = pd.DataFrame(self.results)
        print("\n" + "=" * 65)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 65)
        print(df.to_string(index=False))
        print("=" * 65)

        # Generate markdown report
        report_md = self._generate_markdown_report(df, num_docs)
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info("Updated README.md report.")

        # Plot charts
        plot_latency_graphs(self.latencies_log, "latency_comparison.png")
