import torch
import torch.nn as nn
from transformers import ASTModel, BertModel

from models.audio_context_layer import AudioContextLayer
from models.audio_question_fusion import AudioQuestionFusion


class AudioQuestionAnsweringModel(nn.Module):
    """
    Audio Question Answering model.

    Pipeline:

        Audio
          ↓
        AST
          ↓
        Audio Context Layer
          ↓
        Question-conditioned Cross Attention
          ↓
        Mean pooling over question tokens
          ↓
        Answer representation
          ↓
        Answer classification head
    """

    def __init__(
        self,
        num_answers,
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        bert_model_name="google-bert/bert-base-uncased",
        hidden_size=768,
        num_heads=12,
        context_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.ast = ASTModel.from_pretrained(ast_model_name)

        self.bert = BertModel.from_pretrained(bert_model_name)

        self.audio_context = AudioContextLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=context_layers,
            dropout=dropout,
        )

        self.fusion = AudioQuestionFusion(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.answer_classifier = nn.Linear(
            hidden_size,
            num_answers,
        )

    def forward(
        self,
        input_values,
        input_ids,
        attention_mask,
    ):
        ast_output = self.ast(
            input_values=input_values
        )

        audio_tokens = ast_output.last_hidden_state

        contextualized_audio = self.audio_context(
            audio_tokens=audio_tokens
        )

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        question_tokens = bert_output.last_hidden_state

        fused_question = self.fusion(
            audio_tokens=contextualized_audio,
            question_tokens=question_tokens,
            question_attention_mask=attention_mask,
        )

        mask = attention_mask.unsqueeze(-1).float()

        masked_question = fused_question * mask

        pooled_question = masked_question.sum(dim=1) / mask.sum(
            dim=1
        ).clamp(min=1.0)

        representation = self.dropout(
            pooled_question
        )

        answer_logits = self.answer_classifier(
            representation
        )

        return {
            "audio_tokens": audio_tokens,
            "contextualized_audio": contextualized_audio,
            "question_tokens": question_tokens,
            "fused_question": fused_question,
            "representation": representation,
            "answer_logits": answer_logits,
        }
