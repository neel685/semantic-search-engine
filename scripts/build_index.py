from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_project_root, load_config
from src.data.loader import load_processed_data
from src.retrieval.dense import DenseRetriever


def main() -> None:
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    embedding_cfg = config.get("embedding", {})
    retrieval_cfg = config.get("retrieval", {})
    index_path = Path(dataset_cfg.get("index_path", "data/index"))
    index_path = index_path if index_path.is_absolute() else get_project_root() / index_path
    index_path.mkdir(parents=True, exist_ok=True)

    docs, _, _ = load_processed_data(get_project_root() / dataset_cfg.get("processed_path", "data/processed"))
    retriever = DenseRetriever(
        docs,
        model_name=embedding_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
        device=embedding_cfg.get("device", "auto"),
        normalize=bool(embedding_cfg.get("normalize", True)),
        batch_size=int(embedding_cfg.get("batch_size", 32)),
        index_type=retrieval_cfg.get("index_type", "IndexFlatIP"),
    )
    save_path = index_path / "dense_index.faiss"
    mapping_path = index_path / "document_mapping.json"
    retriever.save(save_path)
    print(f"Saved dense index to {save_path} and mapping to {mapping_path}")


if __name__ == "__main__":
    main()
