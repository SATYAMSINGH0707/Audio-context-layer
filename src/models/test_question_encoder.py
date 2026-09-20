from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, BertModel

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_name = "google-bert/bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    question = "What sound occurs first in the audio?"

    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    question_tokens = output.last_hidden_state

    print("GPU:", torch.cuda.get_device_name(0))
    print("Question:", question)
    print("Input IDs shape:", input_ids.shape)
    print("Question token shape:", question_tokens.shape)
    print("Question device:", question_tokens.device)
    print("Hidden size:", question_tokens.shape[-1])

    assert question_tokens.shape[0] == 1
    assert question_tokens.shape[-1] == 768
    assert question_tokens.device.type == device.type

    print("CHECKPOINT 27 PASSED")


if __name__ == "__main__":
    main()
