import json
from collections import Counter, defaultdict
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


distribution = defaultdict(Counter)

for item in data["questions"]:
    question_type = item["question_type"]
    answer = str(item["answer"]).strip()
    distribution[question_type][answer] += 1


for question_type in sorted(distribution):

    counts = distribution[question_type]

    print(f"{question_type}:")
    print(f"  QA pairs: {sum(counts.values())}")
    print(f"  Unique answers: {len(counts)}")

    for answer, count in counts.most_common():
        print(f"    {count:3d} | {answer}")

    print()


print("QUESTION TYPE DISTRIBUTION CHECK COMPLETE")
