# 📖 How the Blinky Retrieval Benchmark Works
### A friendly guide for humans, developers, and vibecoders 🚀

If you want to understand how this benchmark measures vector retrieval speed without getting lost in academic papers, you're in the right place! Here is the step-by-step breakdown of how `benchmark.py` pits **Moss**, **Chroma**, and **Pinecone** against each other using Blinky's actual AI documentation.

---

## 🏗️ The 3-Way Showdown: Who are we testing?

Before looking at the code, let's understand the contestants:

1. **Chroma (The Local Database)**: An embedded database. It runs inside your Python app and uses SQLite under the hood to store text and metadata, alongside an HNSW index for vectors. Since it runs in local RAM, it has no network latency.
2. **Pinecone (The Cloud Database)**: A fully managed cloud-native vector database. We set up a **Serverless Index** (hosted on AWS). When we index or query, the data travels across the internet to Pinecone's servers. This represents standard cloud DB retrieval latency.
3. **Moss (The Hybrid Local-First)**: Moss is built specifically for real-time AI agents. It compiles the index in the cloud, but downloads the finished vectors into your application process's local RAM. Queries are run locally in-process (sub-10ms), eliminating network roundtrips during search.

---

## ⚙️ Step-by-Step Code Flow

Here is exactly what happens when you run `python benchmark.py`:

### Step 1: Document Loading & Chunking
The script looks inside the `blinky/` directory, grabs all the `.md` documentation files, and breaks them into smaller, digestible chunks (600 characters each, with 100 characters of overlap).

```python
# We slice and dice the markdown text so the AI can search specific passages
start = 0
while start < len(content):
    end = start + chunk_size
    chunk = content[start:end]
    documents.append({"text": chunk, "metadata": {"source": filename}})
    start += chunk_size - overlap
```

### Step 2: Local Embedding Generation
To do semantic search, we must turn text into numbers (vectors). We load a lightweight model called `all-MiniLM-L6-v2` locally on your CPU. It converts each text chunk into a **384-dimensional vector**.

```python
# Load the model and convert chunks into raw numbers (embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(texts, convert_to_numpy=True)
```

### Step 3: Indexing Phase
We time how long it takes to build the index for each system. 
* **Chroma** builds quickly in local RAM.
* **Pinecone** connects to your cloud index, deletes existing vectors to ensure a clean run, and upserts the embeddings in batches of 100 over HTTP.
* **Moss** uploads the raw text to Moss Cloud for compilation, then syncs the completed index back to your local memory.

### Step 4: The Latency Battle (Querying)
We define a list of **30 realistic technical questions** about Blinky's architecture. 
1. We run a few **"warm-up"** queries first (this ensures any cold starts, JIT compilation, or caching layers are already loaded).
2. We run the full set of 30 queries **10 times sequentially** (totaling 300 queries per system).
3. We record the exact execution time for every single query.

### Step 5: Visualizing P50 vs. P99
Once finished, the script processes the latencies:
* **P50 (Median Latency)**: The "typical" response time. 50% of the queries were faster than this.
* **P99 (Tail Latency)**: The worst-case response time. 99% of the queries were faster than this. Measuring P99 helps identify lag spikes or network spikes.

The script uses `matplotlib` to render a beautiful bar chart and box plot using premium colors (botanical green `#004f45`, aquatic blue `#005db7`, and teal `#00838f`) and saves it as `latency_comparison.png`.

---

## 🎮 How to run it yourself

Make sure your virtual environment is active in your terminal, then run:

```bash
# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Add your moss.dev and pinecone.io project credentials to .env

# 3. Run the benchmark script
python benchmark.py
```

The script will dump a clean markdown results table directly into `README.md` and save the visual charts.
