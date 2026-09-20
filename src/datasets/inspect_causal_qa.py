import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QA_DIR = (
    PROJECT_ROOT
    / "data"
    / "daqa"
    / "generated"
    / "qa"
)


for split in ["train", "val", "test"]:

    path = QA_DIR / f"daqa_local_{split}_qa.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    causal = [
        item
        for item in data["questions"]
        if item["question_type"] == "causal"
    ]

    print("=" * 80)
    print(split.upper())
    print("=" * 80)

    for index, item in enumerate(causal, start=1):

        print(f"[{index}]")
        print(f"Question: {item['question']}")
        print(f"Answer:   {item['answer']}")
        print(f"Audio:    {item['audio_filename']}")
        print()

print("CAUSAL QA INSPECTION COMPLETE")
