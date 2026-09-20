import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.audio_question_fusion import AudioQuestionFusion


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = AudioQuestionFusion(
        hidden_size=768,
        num_heads=12,
        dropout=0.1,
    ).to(device)

    audio_tokens = torch.randn(
        2,
        100,
        768,
        device=device,
    )

    question_tokens = torch.randn(
        2,
        10,
        768,
        device=device,
    )

    question_attention_mask = torch.ones(
        2,
        10,
        dtype=torch.long,
        device=device,
    )

    # Simulate padding in the second question.
    question_attention_mask[1, 8:] = 0

    with torch.no_grad():
        fused_question = model(
            audio_tokens=audio_tokens,
            question_tokens=question_tokens,
            question_attention_mask=question_attention_mask,
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Audio tokens shape:", audio_tokens.shape)
    print("Question tokens shape:", question_tokens.shape)
    print("Attention mask shape:", question_attention_mask.shape)
    print("Fused output shape:", fused_question.shape)
    print("Fused output device:", fused_question.device)

    assert fused_question.shape == (2, 10, 768)
    assert fused_question.device.type == device.type

    # Verify that padded question positions are zero.
    padded_values = fused_question[1, 8:, :]

    assert torch.allclose(
        padded_values,
        torch.zeros_like(padded_values),
    )

    print("Padding mask behavior: PASSED")
    print("CHECKPOINT 29 PASSED")


if __name__ == "__main__":
    main()
