import json
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]

    qa_path = (
        project_root
        / "data"
        / "daqa"
        / "generated"
        / "qa"
        / "daqa_local_train_qa.json"
    )

    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    answers = sorted(
        {
            str(item["answer"]).strip()
            for item in questions
        }
    )

    print("Total QA pairs:", len(questions))
    print("Unique answers:", len(answers))
    print()
    print("Answer vocabulary:")

    for index, answer in enumerate(answers):
        print(f"{index:3d}: {answer}")

    assert len(answers) > 0

    print()
    print("CHECKPOINT 33 PASSED")


if __name__ == "__main__":
    main()
