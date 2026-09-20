from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.datasets.daqa_dataset import DAQADataset
from src.datasets.qa_collator import AudioQACollator
from src.models.audio_qa_model import AudioQuestionAnsweringModel


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

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
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    model = AudioQuestionAnsweringModel(
        num_answers=len(collator.answers),
    )

    model = model.to(device)
    model.train()

    input_values = batch["input_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_values=input_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs["answer_logits"]

    print("Logits shape:", logits.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels.tolist())

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels,
    )

    print("Loss:", loss.item())

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite.")

    model.zero_grad(set_to_none=True)

    loss.backward()

    gradient_count = 0
    nonzero_gradient_count = 0

    for parameter in model.parameters():
        if parameter.requires_grad:
            gradient_count += 1

            if parameter.grad is not None:
                if torch.isfinite(parameter.grad).all():
                    nonzero_gradient_count += 1

    print(
        "Trainable parameters with gradients:",
        nonzero_gradient_count,
        "/",
        gradient_count,
    )

    if nonzero_gradient_count == 0:
        raise RuntimeError(
            "No trainable parameters received gradients."
        )

    print()
    print("Training loss and backward pass succeeded.")
    print("CHECKPOINT 41 PASSED")


if __name__ == "__main__":
    main()
