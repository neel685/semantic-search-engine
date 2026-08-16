from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.loader import load_msmarco_subset, save_processed_data


def main() -> None:
    config = load_config()
    documents, queries, qrels = load_msmarco_subset(config)
    output_dir = save_processed_data(documents, queries, qrels)
    print(f"Saved {len(documents)} documents, {len(queries)} queries, and relevance mappings to {output_dir}")


if __name__ == "__main__":
    main()
