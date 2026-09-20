import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QA_DIR = PROJECT_ROOT / "data" / "daqa" / "generated" / "qa"


def load_questions(split):
    path = QA_DIR / f"daqa_local_{split}_qa.json"

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


train_questions = load_questions("train")
test_questions = load_questions("test")

train_answers = {
    str(item["answer"]).strip()
    for item in train_questions
}

print("Unseen test answers:")
print()

for item in test_questions:
    answer = str(item["answer"]).strip()

    if answer not in train_answers:
        print(f"Question type: {item['question_type']}")
        print(f"Question: {item['question']}")
        print(f"Answer: {answer}")
        print(f"Audio: {item['audio_filename']}")
        print("-" * 60)
