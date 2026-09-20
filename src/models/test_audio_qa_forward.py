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

    # Paths
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

    # Load one real dataset sample
    dataset = DAQADataset(
        qa_path=qa_path,
        audio_dir=audio_dir,
    )

    sample = dataset[0]

    print("Audio file:", sample["audio_filename"])
    print("Question:", sample["question"])
    print("Expected answer:", sample["answer"])

    # AST preprocessing
    ast_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    processor = AutoFeatureExtractor.from_pretrained(ast_name)

    waveform = sample["waveform"].squeeze(0).numpy()

    ast_inputs = processor(
        waveform,
        sampling_rate=sample["sample_rate"],
        return_tensors="pt",
    )

    input_values = ast_inputs["input_values"].to(device)

    # BERT preprocessing
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

    # Build complete model
    model = AudioQuestionAnsweringModel(
        context_layers=2,
        dropout=0.1,
    ).to(device)

    model.eval()

    # Full forward pass
    with torch.no_grad():
        outputs = model(
            input_values=input_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    print("AST audio tokens:", outputs["audio_tokens"].shape)
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
        "Final representation:",
        outputs["representation"].shape,
    )

    # Device checks
    print(
        "Representation device:",
        outputs["representation"].device,
    )

    assert outputs["audio_tokens"].ndim == 3
    assert outputs["contextualized_audio"].ndim == 3
    assert outputs["question_tokens"].ndim == 3
    assert outputs["fused_question"].ndim == 3
    assert outputs["representation"].ndim == 2

    assert (
        outputs["representation"].shape[0]
        == 1
    )

    assert (
        outputs["representation"].shape[1]
        == 768
    )

    assert (
        outputs["representation"].device.type
        == device.type
    )

    print("CHECKPOINT 32 PASSED")


if __name__ == "__main__":
    main()
