import json
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor, AutoTokenizer


class AudioQACollator:
    """
    Converts DAQA samples into tensors suitable for training.

    Output:
        input_values
        input_ids
        attention_mask
        labels
    """

    def __init__(
        self,
        qa_path,
        sample_rate=16000,
        ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
        bert_model_name="google-bert/bert-base-uncased",
    ):
        self.qa_path = Path(qa_path)
        self.sample_rate = sample_rate

        with open(self.qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        answers = sorted(
            {
                str(item["answer"]).strip()
                for item in data["questions"]
            }
        )

        self.answers = answers

        self.answer_to_id = {
            answer: index
            for index, answer in enumerate(self.answers)
        }

        self.processor = AutoFeatureExtractor.from_pretrained(
            ast_model_name
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_model_name
        )

    def __call__(self, batch):
        waveforms = [
            item["waveform"].squeeze(0).numpy()
            for item in batch
        ]

        questions = [
            item["question"]
            for item in batch
        ]

        answers = [
            str(item["answer"]).strip()
            for item in batch
        ]

        question_types = [
            item["question_type"]
            for item in batch
        ]

        question_texts = [
            item["question"]
            for item in batch
        ]
        # AST preprocessing.
        ast_inputs = self.processor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )

        # BERT question tokenization.
        text_inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = torch.tensor(
            [
                self.answer_to_id[answer]
                for answer in answers
            ],
            dtype=torch.long,
        )

        return {
            "input_values": ast_inputs["input_values"],
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
            "question_type": question_types,
            "question": question_texts,

        }
