from pathlib import Path
import sys

import torch
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
    model.eval()

    input_values = batch["input_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print("Input values:", input_values.shape)
    print("Input IDs:", input_ids.shape)
    print("Attention mask:", attention_mask.shape)
    print("Labels:", labels.shape)

    with torch.no_grad():
        outputs = model(
            input_values=input_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    print("AST tokens:", outputs["audio_tokens"].shape)
    print(
        "Contextualized audio:",
        outputs["contextualized_audio"].shape,
    )
    print(
        "Question tokens:",
        outputs["question_tokens"].shape,
    )
    print(
        "Fused question:",
        outputs["fused_question"].shape,
    )
    print(
        "Representation:",
        outputs["representation"].shape,
    )
    print(
        "Answer logits:",
        outputs["answer_logits"].shape,
    )

    assert outputs["answer_logits"].shape == (
        2,
        len(collator.answers),
    )

    assert outputs["representation"].shape == (
        2,
        768,
    )

    assert torch.isfinite(
        outputs["answer_logits"]
    ).all()

    print()
    print("Full batch forward pass succeeded.")
    print("CHECKPOINT 40 PASSED")


if __name__ == "__main__":
    main()
