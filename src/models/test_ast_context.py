from pathlib import Path
import sys

import torch
import torchaudio
from transformers import AutoFeatureExtractor, ASTModel

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.audio_context_layer import AudioContextLayer


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    audio_path = (
        project_root
        / "data"
        / "daqa"
        / "generated"
        / "train"
        / "audio"
        / "daqa_local_000000.wav"
    )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    ast_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

    processor = AutoFeatureExtractor.from_pretrained(ast_name)
    ast_model = ASTModel.from_pretrained(ast_name).to(device)

    context_layer = AudioContextLayer(
        hidden_size=768,
        num_heads=12,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            16000,
        )

    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )

    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        ast_output = ast_model(
            input_values=input_values
        )

        audio_tokens = ast_output.last_hidden_state

        contextualized_audio = context_layer(
            audio_tokens=audio_tokens
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Audio shape:", waveform.shape)
    print("AST output shape:", audio_tokens.shape)
    print("Context output shape:", contextualized_audio.shape)
    print("AST device:", audio_tokens.device)
    print("Context device:", contextualized_audio.device)

    assert audio_tokens.shape[-1] == 768
    assert contextualized_audio.shape == audio_tokens.shape
    assert contextualized_audio.device.type == device.type

    print("CHECKPOINT 26 PASSED")


if __name__ == "__main__":
    main()
