import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class DAQADataset(Dataset):
    def __init__(
        self,
        qa_path,
        audio_dir,
        target_sample_rate=16000,
    ):
        self.qa_path = Path(qa_path)
        self.audio_dir = Path(audio_dir)
        self.target_sample_rate = target_sample_rate

        with open(self.qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.questions = data["questions"]

        if not self.questions:
            raise RuntimeError(
                f"No questions found in {self.qa_path}"
            )

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        item = self.questions[index]

        audio_path = self.audio_dir / item["audio_filename"]

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono if necessary.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed.
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.target_sample_rate,
            )

        return {
            "waveform": waveform,
            "sample_rate": self.target_sample_rate,
            "question": item["question"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "audio_filename": item["audio_filename"],
            "audio_index": item["audio_index"],
        }


def main():
    project_root = Path(__file__).resolve().parents[2]

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

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("Audio filename:", sample["audio_filename"])
    print("Waveform shape:", sample["waveform"].shape)
    print("Sample rate:", sample["sample_rate"])
    print("Question type:", sample["question_type"])
    print("Question:", sample["question"])
    print("Answer:", sample["answer"])


if __name__ == "__main__":
    main()
