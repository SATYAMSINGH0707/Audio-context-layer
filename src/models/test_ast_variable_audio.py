from pathlib import Path
import sys

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

    # Find the first QA example for each unique audio file.
    seen = set()
    samples = []

    for index in range(len(dataset)):
        sample = dataset[index]
        filename = sample["audio_filename"]

        if filename not in seen:
            seen.add(filename)
            samples.append(sample)

        if len(samples) >= 5:
            break

    print("Unique audio files tested:", len(samples))
    print()

    ast_shapes = []

    for index, sample in enumerate(samples):
        waveform = sample["waveform"].squeeze(0).numpy()

        processed = processor(
            waveform,
            sampling_rate=sample["sample_rate"],
            return_tensors="pt",
        )

        input_values = processed["input_values"]

        ast_shapes.append(tuple(input_values.shape))

        duration = (
            sample["waveform"].shape[-1]
            / sample["sample_rate"]
        )

        print(f"Audio {index}:")
        print("  File:", sample["audio_filename"])
        print("  Duration:", round(duration, 2), "seconds")
        print("  AST input:", input_values.shape)
        print()

    # All examples should have the same AST input shape.
    assert len(set(ast_shapes)) == 1

    print("All tested audio files produce the same AST input shape.")
    print("AST batch preprocessing is compatible.")
    print("CHECKPOINT 37 PASSED")


if __name__ == "__main__":
    main()
