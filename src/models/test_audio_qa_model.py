from pathlib import Path
import sys

import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.audio_qa_model import AudioQuestionAnsweringModel


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = AudioQuestionAnsweringModel(
        context_layers=2,
        dropout=0.1,
    ).to(device)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Model:", type(model).__name__)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)
    print(
        "Model device:",
        next(model.parameters()).device,
    )

    assert next(model.parameters()).device.type == device.type

    print("CHECKPOINT 31 PASSED")


if __name__ == "__main__":
    main()
