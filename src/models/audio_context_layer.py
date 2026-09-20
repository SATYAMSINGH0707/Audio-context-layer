import torch
import torch.nn as nn


class AudioContextLayer(nn.Module):
    """
    Temporal context module for AST audio representations.

    Input:
        audio_tokens: [batch, time, hidden_size]

    Output:
        contextualized_audio: [batch, time, hidden_size]
    """

    def __init__(
        self,
        hidden_size=768,
        num_heads=12,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, audio_tokens):
        contextualized_audio = self.temporal_transformer(
            audio_tokens
        )

        contextualized_audio = self.layer_norm(
            contextualized_audio
        )

        return contextualized_audio
