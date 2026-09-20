from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.datasets.daqa_dataset import DAQADataset
from src.datasets.qa_collator import AudioQACollator


def main():
    qa_path = (
        project_root
        / "data"
        / "daqa"
        / "generated"
        / "qa"
        / "daqa_local_train_qa.json"
    )

    audio_dir = (
        project_root
        / "data"
        / "daqa"
        / "generated"
        / "train"
        / "audio"
    )

    dataset = DAQADataset(
        qa_path=qa_path,
        audio_dir=audio_dir,
    )

    collator = AudioQACollator(
        qa_path=qa_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    print("Batch keys:", batch.keys())
    print(
        "AST input shape:",
        batch["input_values"].shape,
    )
    print(
        "BERT input IDs shape:",
        batch["input_ids"].shape,
    )
    print(
        "Attention mask shape:",
        batch["attention_mask"].shape,
    )
    print(
        "Labels shape:",
        batch["labels"].shape,
    )
    print(
        "Labels:",
        batch["labels"].tolist(),
    )

    print(
        "Number of answer classes:",
        len(collator.answers),
    )

    assert batch["input_values"].ndim == 3
    assert batch["input_values"].shape[0] == 4

    assert batch["input_ids"].ndim == 2
    assert batch["input_ids"].shape[0] == 4

    assert batch["attention_mask"].shape == (
        batch["input_ids"].shape
    )

    assert batch["labels"].shape == (4,)

    assert batch["labels"].dtype == torch.long

    assert (
        batch["labels"].min().item()
        >= 0
    )

    assert (
        batch["labels"].max().item()
        < len(collator.answers)
    )

    assert torch.isfinite(
        batch["input_values"]
    ).all()

    print()
    print("Collator batch test passed.")
    print("CHECKPOINT 39 PASSED")


if __name__ == "__main__":
    main()
