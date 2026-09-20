from pathlib import Path
import sys

import torch
from transformers import AutoFeatureExtractor

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.datasets.daqa_dataset import DAQADataset


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

    processor = AutoFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    print("Testing AST preprocessing")
    print()

    for index in [0, 1, 2]:
        sample = dataset[index]

        waveform = sample["waveform"].squeeze(0).numpy()

        processed = processor(
            waveform,
            sampling_rate=sample["sample_rate"],
            return_tensors="pt",
        )

        input_values = processed["input_values"]

        print(f"Sample {index}:")
        print("  Audio:", sample["audio_filename"])
        print(
            "  Original samples:",
            sample["waveform"].shape[-1],
        )
        print(
            "  Original duration:",
            round(
                sample["waveform"].shape[-1]
                / sample["sample_rate"],
                2,
            ),
            "seconds",
        )
        print(
            "  AST input shape:",
            input_values.shape,
        )
        print(
            "  AST input dtype:",
            input_values.dtype,
        )
        print()

        assert input_values.ndim == 3
        assert input_values.shape[0] == 1
        assert torch.isfinite(input_values).all()

    print("AST preprocessing handles variable-length audio.")
    print("CHECKPOINT 36 PASSED")


if __name__ == "__main__":
    main()
