import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS = ["train", "val", "test"]

QA_DIR = (
    PROJECT_ROOT
    / "data"
    / "daqa"
    / "generated"
    / "qa"
)


for split in SPLITS:

    path = QA_DIR / f"daqa_local_{split}_qa.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    causal = [
        item
        for item in data["questions"]
        if item["question_type"] == "causal"
    ]

    templates = Counter(
        item.get("question_template", "MISSING")
        for item in causal
    )

    print(f"{split}:")
    print(f"  Causal QA pairs: {len(causal)}")
    print(f"  Question templates: {len(templates)}")

    for template, count in templates.most_common():
        print(f"    {count:3d} | {template}")

    print()


print("CAUSAL TEMPLATE INSPECTION COMPLETE")
