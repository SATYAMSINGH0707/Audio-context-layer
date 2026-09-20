from pathlib import Path
import sys

import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.audio_context_layer import AudioContextLayer


def main():
    model = AudioContextLayer(
        hidden_size=768,
        num_heads=12,
        num_layers=2,
        dropout=0.1,
    )

    print("Audio Context Layer instantiated successfully")
    print("Model type:", type(model).__name__)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Total parameters:", total_params)

    audio_tokens = torch.randn(2, 20, 768)

    output = model(
        audio_tokens=audio_tokens,
    )

    print("Input shape:", audio_tokens.shape)
    print("Output shape:", output.shape)

    assert output.shape == (2, 20, 768)

    print("CHECKPOINT 24 PASSED")


if __name__ == "__main__":
    main()
