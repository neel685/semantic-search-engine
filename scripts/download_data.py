from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from src.config import get_project_root, load_config


def main() -> None:
    config = load_config()
    dataset_cfg = config.get("dataset", {})
    raw_dir = Path(dataset_cfg.get("raw_path", "data/raw"))
    raw_dir = raw_dir if raw_dir.is_absolute() else get_project_root() / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    max_documents = int(dataset_cfg.get("max_documents", 2000))
    print(f"Downloading MS MARCO subset with up to {max_documents} documents...")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split=f"train[:{max_documents}]")
    output_path = raw_dir / "msmarco_subset.jsonl"
    with output_path.open("w", encoding="utf-8") as file:
        for row in dataset:
            file.write(str(row).replace("'", '"') + "\n")
    print(f"Saved subset to {output_path}")


if __name__ == "__main__":
    main()
