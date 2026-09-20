import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QA_PATH = (
    PROJECT_ROOT
    / "data"
    / "daqa"
    / "generated"
    / "qa"
    / "daqa_local_train_qa.json"
)


with open(QA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


answers = [
    str(item["answer"]).strip()
    for item in data["questions"]
]

counts = Counter(answers)

print("Training answer distribution:")
print(f"Unique answers: {len(counts)}")
print(f"Total QA pairs: {len(answers)}")
print()

for answer, count in counts.most_common():
    print(f"{count:3d} | {answer}")
