import os
import time
import glob
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Try importing the benchmarked systems
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("Warning: pinecone-client is not installed. Run 'uv pip install pinecone-client'")
try:
    import chromadb
except ImportError:
    print("Warning: chromadb is not installed. Run 'uv pip install chromadb'")
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers is not installed. Run 'uv pip install sentence-transformers'")
try:
    from moss import MossClient, DocumentInfo, QueryOptions
    MOSS_AVAILABLE = True
except ImportError:
    MOSS_AVAILABLE = False
    print("Warning: moss is not installed. Run 'uv pip install moss'")

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
DATASET_DIR = "blinky"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MOSS_PROJECT_ID = os.getenv("MOSS_PROJECT_ID")
MOSS_PROJECT_KEY = os.getenv("MOSS_PROJECT_KEY")
MOSS_INDEX_NAME = "blinky-benchmarks"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "blinky-benchmarks")

# --- 1. DATASET LOADING & CHUNKING ---
def load_and_chunk_docs(directory, chunk_size=600, overlap=100):
    print(f"\n[1/6] Loading markdown files from: {directory}")
    md_files = glob.glob(os.path.join(directory, "*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {directory}")
        
    documents = []
    total_raw_chars = 0
    
    for path in md_files:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            total_raw_chars += len(content)
            
            # Simple overlap chunking
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk = content[start:end]
                documents.append({
                    "text": chunk,
                    "metadata": {"source": filename, "chunk_index": start // (chunk_size - overlap)}
                })
                start += chunk_size - overlap
                
    print(f"Loaded {len(md_files)} files. Created {len(documents)} text chunks (Total characters: {total_raw_chars})")
    return documents

# --- 2. VECTOR STORAGE IMPLEMENTATIONS ---

# 2a. Pinecone Implementation
class PineconeBenchIndex:
    def __init__(self, api_key, index_name, dimension):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        
        # Check if index exists, else create it
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone serverless index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
                
        self.index = self.pc.Index(self.index_name)

    def index_documents(self, documents, embeddings):
        start_time = time.perf_counter()
        
        # Delete existing data to start fresh
        try:
            self.index.delete(delete_all=True)
            # Wait a moment for deletion to register
            time.sleep(1)
        except Exception:
            pass
            
        # Format upsert payloads
        upsert_data = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            upsert_data.append((
                f"chunk_{i}",
                emb.tolist(),
                {"text": doc["text"], "source": doc["metadata"]["source"]}
            ))
            
        # Upsert in batches of 100 (safe size for Pinecone)
        batch_size = 100
        for idx in range(0, len(upsert_data), batch_size):
            batch = upsert_data[idx:idx + batch_size]
            self.index.upsert(vectors=batch)
            
        return time.perf_counter() - start_time

    def query(self, query_embedding, k=3):
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True
        )
        
        formatted = []
        for match in results.matches:
            formatted.append({"text": match.metadata.get("text", ""), "score": match.score})
        return formatted

# 2b. Chroma Implementation
class ChromaBenchIndex:
    def __init__(self):
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(
            name="blinky_benchmarks",
            metadata={"hnsw:space": "cosine"}
        )

    def index_documents(self, documents, embeddings):
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

    def query(self, query_embedding, k=3):
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

# 2c. Moss Implementation
class MossBenchIndex:
    def __init__(self, project_id, project_key):
        self.client = MossClient(project_id, project_key)
        self.index_name = MOSS_INDEX_NAME

    async def clean_existing_index(self):
        try:
            print("Cleaning existing Moss index...")
            await self.client.delete_index(self.index_name)
            await asyncio.sleep(1)
        except Exception:
            pass

    async def index_documents(self, documents):
        start_time = time.perf_counter()
        
        docs_to_add = [DocumentInfo(id=f"chunk_{i}", text=doc["text"]) for i, doc in enumerate(documents)]
        
        print("Uploading chunks to Moss Cloud for compilation...")
        await self.client.create_index(self.index_name, docs_to_add)
        
        indexing_duration = time.perf_counter() - start_time
        
        print("Loading compiled Moss index into local process memory...")
        await self.client.load_index(self.index_name)
        
        return indexing_duration

    async def query(self, query_text, k=3):
        results = await self.client.query(
            self.index_name,
            query_text,
            QueryOptions(top_k=k)
        )
        
        formatted = []
        for doc in results.docs:
            formatted.append({"text": doc.text, "score": doc.score})
        return formatted

# --- 3. QUESTIONS SET ---
BENCHMARK_QUESTIONS = [
    "How does Blinky's preflight router classify user intent?",
    "What are the five intents classified by the preflight router?",
    "How does coordinate scaling work in Blinky's screen tutor?",
    "What is the stable ref system for screen elements?",
    "How does the UI map cache with stable refs work?",
    "How does Blinky handle Spotify media playback?",
    "What is the role of SearXNG in Spotify URI resolution?",
    "How does Blinky execute keyboard shortcuts?",
    "How does the app context registry auto-generate navigation guides?",
    "What fallback mechanisms does Blinky use to open apps?",
    "How does the bounded autopilot loop work?",
    "What are the safe and blocked actions in autopilot?",
    "How does Blinky exclude itself from screen capture?",
    "What native API does Blinky use to exclude its windows from screenshots?",
    "How does Blinky handle full-width search input highlighting?",
    "What OCR engines does Blinky use for extraction?",
    "How does Blinky's tool sufficiency check work?",
    "What is the local Ollama execution optimization in Blinky?",
    "How does Blinky interact with Sarvam AI for voice input and output?",
    "Where are the dynamic app context guides saved?",
    "What is the purpose of the SetWindowDisplayAffinity API in Blinky?",
    "How does Blinky map Tauri coordinates to physical screen coordinates?",
    "Does Blinky support multi-monitor setups?",
    "How does the Python worker communicate with Tauri?",
    "What model is used by default for local inference in Blinky?",
    "How does Blinky run Playwright with Edge browser?",
    "What are the options for overriding Groq model and URL?",
    "How does Blinky handle the stop run cancellation?",
    "What is the minimum resizable width of Blinky's command window?",
    "What is the role of dxcam in screen capture?"
]

# --- 4. BENCHMARK EXECUTION SUITE ---
async def main():
    print("=" * 60)
    print("🧠 BLINKY RETRIEVAL BENCHMARK: MOSS vs. CHROMA vs. PINECONE")
    print("=" * 60)
    
    # Check dataset
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory {DATASET_DIR} does not exist.")
        return
        
    # Load and chunk docs
    documents = load_and_chunk_docs(DATASET_DIR)
    
    # Load embedding model locally
    print(f"\n[2/6] Loading local embedding model: '{EMBEDDING_MODEL_NAME}'")
    embed_start = time.perf_counter()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Model loaded in {time.perf_counter() - embed_start:.2f} seconds.")
    
    # Generate document embeddings
    print("Generating embeddings for all text chunks...")
    texts = [doc["text"] for doc in documents]
    doc_embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dimension = doc_embeddings.shape[1]
    print(f"Generated {len(doc_embeddings)} embeddings of dimension {dimension}.")
    
    # Prepare results storage
    results_data = []
    
    # ---------------------------------------------
    # PHASE A: BENCHMARK CHROMA
    # ---------------------------------------------
    print("\n[3/6] Benchmarking Chroma (Ephemeral Client)...")
    chroma_index = ChromaBenchIndex()
    chroma_idx_time = chroma_index.index_documents(documents, doc_embeddings)
    print(f"Chroma indexing complete in: {chroma_idx_time:.4f} seconds")
    
    # Warm-up
    for q in BENCHMARK_QUESTIONS[:5]:
        q_emb = model.encode(q, convert_to_numpy=True)
        chroma_index.query(q_emb)
        
    # Run query benchmark (10 iterations = 300 queries)
    chroma_latencies = []
    for iteration in range(10):
        for q in BENCHMARK_QUESTIONS:
            q_emb = model.encode(q, convert_to_numpy=True)
            start_q = time.perf_counter()
            chroma_index.query(q_emb)
            chroma_latencies.append((time.perf_counter() - start_q) * 1000) # ms
            
    # Record results
    chroma_p50 = np.percentile(chroma_latencies, 50)
    chroma_p99 = np.percentile(chroma_latencies, 99)
    results_data.append({
        "System": "Chroma (Ephemeral)",
        "Indexing Time (s)": chroma_idx_time,
        "P50 Latency (ms)": chroma_p50,
        "P99 Latency (ms)": chroma_p99,
        "Type": "Local Database",
        "Local Embedding Time Included": "No"
    })

    # ---------------------------------------------
    # PHASE B: BENCHMARK PINECONE
    # ---------------------------------------------
    pinecone_latencies = []
    pinecone_idx_time = None
    pinecone_configured = False
    
    if not PINECONE_AVAILABLE:
        print("\n[4/6] Pinecone client is not installed. Skipping Pinecone benchmarks.")
    elif not PINECONE_API_KEY or "YOUR_PINECONE" in PINECONE_API_KEY:
        print("\n[4/6] Pinecone API key is missing. Skipping Pinecone benchmarks.")
        print("💡 Set PINECONE_API_KEY in c:/projects/moss/.env to run it.")
    else:
        print(f"\n[4/6] Benchmarking Pinecone (Cloud Serverless - {PINECONE_INDEX_NAME})...")
        try:
            pinecone_index = PineconeBenchIndex(PINECONE_API_KEY, PINECONE_INDEX_NAME, dimension)
            pinecone_idx_time = pinecone_index.index_documents(documents, doc_embeddings)
            print(f"Pinecone indexing complete in: {pinecone_idx_time:.4f} seconds")
            
            pinecone_configured = True
            
            # Wait for vectors to settle/index in cloud
            print("Waiting 10 seconds for cloud indexing replication...")
            time.sleep(10)
            
            # Warm-up
            for q in BENCHMARK_QUESTIONS[:5]:
                q_emb = model.encode(q, convert_to_numpy=True)
                pinecone_index.query(q_emb)
                
            # Run query benchmark (10 iterations = 300 queries)
            for iteration in range(10):
                for q in BENCHMARK_QUESTIONS:
                    q_emb = model.encode(q, convert_to_numpy=True)
                    start_q = time.perf_counter()
                    pinecone_index.query(q_emb)
                    pinecone_latencies.append((time.perf_counter() - start_q) * 1000) # ms
                    
            # Record results
            pinecone_p50 = np.percentile(pinecone_latencies, 50)
            pinecone_p99 = np.percentile(pinecone_latencies, 99)
            results_data.append({
                "System": "Pinecone (Serverless)",
                "Indexing Time (s)": pinecone_idx_time,
                "P50 Latency (ms)": pinecone_p50,
                "P99 Latency (ms)": pinecone_p99,
                "Type": "Cloud Database (AWS)",
                "Local Embedding Time Included": "No"
            })
        except Exception as e:
            print(f"Error executing Pinecone benchmark: {e}")

    # ---------------------------------------------
    # PHASE C: BENCHMARK MOSS
    # ---------------------------------------------
    moss_latencies = []
    moss_idx_time = None
    moss_configured = False
    
    if not MOSS_AVAILABLE:
        print("\n[5/6] Moss SDK is not installed. Skipping Moss benchmarks.")
    elif not MOSS_PROJECT_ID or "YOUR_MOSS" in MOSS_PROJECT_ID:
        print("\n[5/6] Moss credentials missing. Skipping Moss benchmarks.")
        print("💡 Set MOSS_PROJECT_ID and MOSS_PROJECT_KEY in c:/projects/moss/.env")
    else:
        print("\n[5/6] Benchmarking Moss (Local-First Runtime)...")
        try:
            moss_index = MossBenchIndex(MOSS_PROJECT_ID, MOSS_PROJECT_KEY)
            
            # Clean old index if existing
            await moss_index.clean_existing_index()
            
            # Index document texts
            moss_idx_time = await moss_index.index_documents(documents)
            print(f"Moss indexing & sync complete in: {moss_idx_time:.4f} seconds")
            
            moss_configured = True
            
            # Warm-up
            for q in BENCHMARK_QUESTIONS[:5]:
                await moss_index.query(q)
                
            # Run query benchmark (10 iterations = 300 queries)
            for iteration in range(10):
                for q in BENCHMARK_QUESTIONS:
                    start_q = time.perf_counter()
                    await moss_index.query(q)
                    moss_latencies.append((time.perf_counter() - start_q) * 1000) # ms
                    
            # Record results
            moss_p50 = np.percentile(moss_latencies, 50)
            moss_p99 = np.percentile(moss_latencies, 99)
            results_data.append({
                "System": "Moss (Local-First)",
                "Indexing Time (s)": moss_idx_time,
                "P50 Latency (ms)": moss_p50,
                "P99 Latency (ms)": moss_p99,
                "Type": "Local Memory Engine",
                "Local Embedding Time Included": "Yes (Embedded in SDK)"
            })
        except Exception as e:
            print(f"Error executing Moss benchmark: {e}")

    # --- 5. PRINT SUMMARY TABLE ---
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    df = pd.DataFrame(results_data)
    print(df.to_string(index=False))
    print("=" * 60)
    
    # Write to README.md
    write_results_markdown(df)
    
    # --- 6. PLOT GRAPH ---
    print("\n[6/6] Plotting latency graphs...")
    plot_latency_graphs(chroma_latencies, pinecone_latencies, moss_latencies, pinecone_configured, moss_configured)
    print("Graphs saved to: c:/projects/moss/latency_comparison.png")
    print("\nBenchmark completed successfully! Check c:/projects/moss/README.md for the full report.")

# Helper to write README.md
def write_results_markdown(df):
    table_md = df.to_markdown(index=False)
    
    readme_content = f"""# 📊 Blinky Vector Retrieval Benchmarks

This repository documents the benchmarking results for the **Blinky** AI Desktop Tutor retrieval engine, comparing **Moss**, **Chroma**, and **Pinecone**.

## ⚙️ Environment Details
* **Dataset:** {len(BENCHMARK_QUESTIONS)} test queries run against local docs (`blinky/*.md` AI documentation files).
* **Embedding Model:** `sentence-transformers/{EMBEDDING_MODEL_NAME}` (384 dimensions).
* **Execution:** Run locally from `c:/projects/moss`.

## 📈 Benchmarking Metrics

{table_md}

## 🔍 Visual Analysis

Below is the latency distribution chart showing the query response times (in milliseconds) across the systems:

![Latency Comparison](latency_comparison.png)

## 💡 Key Findings

1. **Chroma (Ephemeral)** represents a traditional local vector database. It has extremely fast retrieval because it runs entirely in local RAM and doesn't traverse the internet.
2. **Pinecone (Serverless)** is a cloud-native vector database. Its retrieval queries must traverse the network to Pinecone Cloud, which introduces network roundtrip latencies (typically 30ms - 100ms) representing standard cloud database performance.
3. **Moss (Local-First)** compiles the index in the cloud but loads the vectors into application memory. Retrieval happens fully locally in-process without network hops. Note that Moss's query latency *includes* the text embedding generation inside the SDK, whereas Chroma and Pinecone query times represent pure vector matching.

---
*Generated automatically by `benchmark.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

# Helper to plot graph with Digital Ledger styling colors
def plot_latency_graphs(chroma_lat, pinecone_lat, moss_lat, pinecone_configured, moss_configured):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = '#071e27'
    plt.rcParams['axes.labelcolor'] = '#071e27'
    plt.rcParams['xtick.color'] = '#071e27'
    plt.rcParams['ytick.color'] = '#071e27'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#f3faff')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')
    
    # Color palette
    colors = {
        "Chroma": "#005db7",      # Aquatic Blue
        "Pinecone": "#00838f",    # Muted Teal
        "Moss": "#004f45"         # Botanical Green
    }
    
    # 1. Latency Bar Chart (P50 and P99)
    systems = ["Chroma"]
    p50_vals = [np.percentile(chroma_lat, 50)]
    p99_vals = [np.percentile(chroma_lat, 99)]
    
    if pinecone_configured:
        systems.append("Pinecone")
        p50_vals.append(np.percentile(pinecone_lat, 50))
        p99_vals.append(np.percentile(pinecone_lat, 99))
        
    if moss_configured:
        systems.append("Moss")
        p50_vals.append(np.percentile(moss_lat, 50))
        p99_vals.append(np.percentile(moss_lat, 99))
        
    x = np.arange(len(systems))
    width = 0.35
    
    # Plotting
    rects1 = ax1.bar(x - width/2, p50_vals, width, label='P50 (Median)', color='#005db7', alpha=0.85)
    rects2 = ax1.bar(x + width/2, p99_vals, width, label='P99 (99th %tile)', color='#004f45', alpha=0.9)
    
    ax1.set_title("Query Latency Comparison (Lower is Better)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel("Latency (milliseconds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontweight='semibold')
    ax1.legend(frameon=False)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
        
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
                        
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)

    # 2. Box Plot of Query Latencies
    data_to_plot = [chroma_lat]
    plot_labels = ["Chroma"]
    
    if pinecone_configured:
        data_to_plot.append(pinecone_lat)
        plot_labels.append("Pinecone")
        
    if moss_configured:
        data_to_plot.append(moss_lat)
        plot_labels.append("Moss")
        
    box = ax2.boxplot(data_to_plot, patch_artist=True, showfliers=False)
    ax2.set_xticks(range(1, len(plot_labels) + 1))
    ax2.set_xticklabels(plot_labels)
    
    box_colors = [colors["Chroma"]]
    if pinecone_configured:
        box_colors.append(colors["Pinecone"])
    if moss_configured:
        box_colors.append(colors["Moss"])
        
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#071e27')
        
    for median in box['medians']:
        median.set_color('#ffffff')
        median.set_linewidth(2)
        
    ax2.set_title("Query Latency Distribution (No Outliers)", fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylabel("Latency (milliseconds)")
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("latency_comparison.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()

if __name__ == "__main__":
    asyncio.run(main())
