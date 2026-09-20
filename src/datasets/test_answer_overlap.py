import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QA_DIR = PROJECT_ROOT / "data" / "daqa" / "generated" / "qa"


def load_answers(split):
    path = QA_DIR / f"daqa_local_{split}_qa.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        str(item["answer"]).strip()
        for item in data["questions"]
    }


train_answers = load_answers("train")
val_answers = load_answers("val")
test_answers = load_answers("test")

val_unseen = val_answers - train_answers
test_unseen = test_answers - train_answers

print("Answer vocabulary:")
print("  Train:", len(train_answers))
print("  Validation:", len(val_answers))
print("  Test:", len(test_answers))

print()
print("Validation answers unseen in train:", len(val_unseen))

for answer in sorted(val_unseen):
    print("  ", answer)

print()
print("Test answers unseen in train:", len(test_unseen))

for answer in sorted(test_unseen):
    print("  ", answer)

print()

if not val_unseen and not test_unseen:
    print("ANSWER VOCABULARY CHECK PASSED")
else:
    print("ANSWER VOCABULARY OVERLAP REQUIRES ATTENTION")
