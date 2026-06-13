# 📊 Blinky Vector Retrieval Benchmarks

This repository documents the benchmarking results for the **Blinky** AI Desktop Tutor retrieval engine, comparing **Moss**, **Chroma**, and **Pinecone**.

## ⚙️ Environment Details
* **Dataset:** 30 test queries run against local docs (`blinky/*.md` AI documentation files).
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
* **Execution:** Run locally from `c:/projects/moss`.

## 📈 Benchmarking Metrics

| System                |   Indexing Time (s) |   P50 Latency (ms) |   P99 Latency (ms) | Type                 | Local Embedding Time Included   |
|:----------------------|--------------------:|-------------------:|-------------------:|:---------------------|:--------------------------------|
| Chroma (Ephemeral)    |            0.057135 |            0.70345 |            1.08871 | Local Database       | No                              |
| Pinecone (Serverless) |            3.66464  |          483.809   |          719.06    | Cloud Database (AWS) | No                              |
| Moss (Local-First)    |           10.9375   |            4.9902  |           10.6741  | Local Memory Engine  | Yes (Embedded in SDK)           |

## 🔍 Visual Analysis

Below is the latency distribution chart showing the query response times (in milliseconds) across the systems:

![Latency Comparison](latency_comparison.png)

## 💡 Key Findings

1. **Chroma (Ephemeral)** represents a traditional local vector database. It has extremely fast retrieval because it runs entirely in local RAM and doesn't traverse the internet.
2. **Pinecone (Serverless)** is a cloud-native vector database. Its retrieval queries must traverse the network to Pinecone Cloud, which introduces network roundtrip latencies (typically 30ms - 100ms) representing standard cloud database performance.
3. **Moss (Local-First)** compiles the index in the cloud but loads the vectors into application memory. Retrieval happens fully locally in-process without network hops. Note that Moss's query latency *includes* the text embedding generation inside the SDK, whereas Chroma and Pinecone query times represent pure vector matching.

---
*Generated automatically by `benchmark.py` on 2026-06-13 13:10:27*
