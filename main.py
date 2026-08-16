from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic Search Engine")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--query", type=str, default="", help="Single query string for quick retrieval")
    parser.add_argument("--method", type=str, default="hybrid", choices=["bm25", "semantic", "hybrid"], help="Retrieval method")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print(json.dumps({"config": cfg["dataset"]["name"], "method": args.method, "top_k": args.top_k}, indent=2))
    if args.query:
        print(f"Query supplied: {args.query}")
    print(f"Project root: {Path(__file__).resolve().parent}")


if __name__ == "__main__":
    main()
