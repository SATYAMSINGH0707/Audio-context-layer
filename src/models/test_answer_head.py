import json
from pathlib import Path
import sys

import torch
from transformers import AutoFeatureExtractor, AutoTokenizer

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.datasets.daqa_dataset import DAQADataset
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

    # Load QA data and build answer vocabulary
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    answers = sorted(
        {
            str(item["answer"]).strip()
            for item in qa_data["questions"]
        }
    )

    answer_to_id = {
        answer: index
        for index, answer in enumerate(answers)
    }

    print("Number of answer classes:", len(answers))

    dataset = DAQADataset(
        qa_path=qa_path,
        audio_dir=audio_dir,
    )

    sample = dataset[0]

    print("Question:", sample["question"])
    print("Expected answer:", sample["answer"])
    print(
        "Expected answer ID:",
        answer_to_id[sample["answer"]],
    )

    # AST preprocessing
    ast_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

    processor = AutoFeatureExtractor.from_pretrained(
        ast_name
    )

    waveform = sample["waveform"].squeeze(0).numpy()

    ast_inputs = processor(
        waveform,
        sampling_rate=sample["sample_rate"],
        return_tensors="pt",
    )

    input_values = ast_inputs["input_values"].to(device)

    # Question preprocessing
    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased"
    )

    text_inputs = tokenizer(
        sample["question"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    # Complete model
    model = AudioQuestionAnsweringModel(
        num_answers=len(answers),
        context_layers=2,
        dropout=0.1,
    ).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(
            input_values=input_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs["answer_logits"]

    predicted_id = logits.argmax(dim=-1).item()
    predicted_answer = answers[predicted_id]

    print("Answer logits shape:", logits.shape)
    print("Predicted answer ID:", predicted_id)
    print("Predicted answer:", predicted_answer)
    print(
        "Representation shape:",
        outputs["representation"].shape,
    )

    assert logits.shape == (1, len(answers))
    assert outputs["representation"].shape == (1, 768)
    assert torch.isfinite(logits).all()

    print("Answer classification head: PASSED")
    print("CHECKPOINT 35 PASSED")


if __name__ == "__main__":
    main()
