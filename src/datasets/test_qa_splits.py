import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = PROJECT_ROOT / "data" / "daqa" / "generated"

SPLITS = {
    "train": 40,
    "val": 10,
    "test": 10,
}

REQUIRED_TYPES = {
    "what",
    "counting",
    "temporal",
    "causal",
}

REQUIRED_FIELDS = {
    "question_type",
    "question",
    "answer",
    "set",
    "audio_index",
    "audio_filename",
}


for split, expected_audio_count in SPLITS.items():

    qa_path = (
        BASE_DIR
        / "qa"
        / f"daqa_local_{split}_qa.json"
    )

    audio_dir = BASE_DIR / split / "audio"

    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    print(f"{split}:")
    print(f"  Questions: {len(questions)}")

    assert data["info"]["split"] == split
    assert data["info"]["num_audio"] == expected_audio_count
    assert data["info"]["num_questions"] == len(questions)

    question_types = {
        item["question_type"]
        for item in questions
    }

    print(f"  Question types: {sorted(question_types)}")

    assert REQUIRED_TYPES.issubset(question_types)

    missing_fields = []

    for index, item in enumerate(questions):

        missing = REQUIRED_FIELDS - set(item.keys())

        if missing:
            missing_fields.append(
                (index, sorted(missing))
            )

        audio_path = audio_dir / item["audio_filename"]

        assert audio_path.exists(), (
            f"Missing audio: {audio_path}"
        )

        assert item["set"] == split

    assert not missing_fields, (
        f"Missing fields: {missing_fields}"
    )

    referenced_audio = {
        item["audio_filename"]
        for item in questions
    }

    actual_audio = {
        path.name
        for path in audio_dir.glob("*.wav")
    }

    assert referenced_audio.issubset(actual_audio)

    print("  Audio references: PASS")
    print("  Required fields: PASS")
    print("  Split labels: PASS")
    print()


print("QA SPLIT VALIDATION PASSED")
