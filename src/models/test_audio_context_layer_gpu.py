from pathlib import Path
import sys

import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.audio_context_layer import AudioContextLayer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    model = AudioContextLayer(
        hidden_size=768,
        num_heads=12,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    audio_tokens = torch.randn(
        2,
        100,
        768,
        device=device,
    )

    output = model(
        audio_tokens=audio_tokens,
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model device:", next(model.parameters()).device)
    print("Input device:", audio_tokens.device)
    print("Input shape:", audio_tokens.shape)
    print("Output shape:", output.shape)

    assert output.device.type == "cuda"
    assert output.shape == (2, 100, 768)

    print("CHECKPOINT 25 PASSED")


if __name__ == "__main__":
    main()
