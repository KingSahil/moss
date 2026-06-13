#!/usr/bin/env python3
"""
CLI entry point to execute the Moss Retrieval Benchmarks.
"""

import argparse
import asyncio
import logging
import sys

# Ensure src directory is in Python path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from benchmark.runner import BenchmarkRunner

def configure_logging() -> None:
    """Configures structured, clean logging outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Blinky retrieval benchmarks comparing Moss, Chroma, and Pinecone."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of query set loop iterations (default: 10)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Document chunk size in characters (default: 600)."
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Document chunk sliding window overlap in characters (default: 100)."
    )

    args = parser.parse_args()
    configure_logging()

    runner = BenchmarkRunner(
        iterations=args.iterations,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logging.warning("Benchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.exception("Fatal error during benchmark execution: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
