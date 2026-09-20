import torch
import torch.nn as nn


class AudioQuestionFusion(nn.Module):
    """
    Cross-attention fusion between audio context tokens
    and question tokens.

    Inputs:
        audio_tokens:
            [batch, audio_time, hidden_size]

        question_tokens:
            [batch, question_time, hidden_size]

        question_attention_mask:
            [batch, question_time]

    Output:
        fused_question:
            [batch, question_time, hidden_size]
    """

    def __init__(
        self,
        hidden_size=768,
        num_heads=12,
        dropout=0.1,
    ):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        audio_tokens,
        question_tokens,
        question_attention_mask=None,
    ):
        key_padding_mask = None

        if question_attention_mask is not None:
            key_padding_mask = question_attention_mask == 0

        attended_question, _ = self.cross_attention(
            query=question_tokens,
            key=audio_tokens,
            value=audio_tokens,
        )

        fused_question = self.norm(
            question_tokens + attended_question
        )

        if key_padding_mask is not None:
            fused_question = fused_question.masked_fill(
                key_padding_mask.unsqueeze(-1),
                0.0,
            )

        return fused_question
