import json
from pathlib import Path


SPLITS = ["train", "val", "test"]


def load_narratives(narrative_dir):
    narrative_dir = Path(narrative_dir)

    files = sorted(
        f for f in narrative_dir.glob("*.json")
        if not f.name.startswith("._")
    )

    narratives = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            narratives.append(json.load(f))

    return narratives


def event_label(event):
    return f"{event['source']} {event['action']}"


def generate_questions(narrative):
    events = narrative["events"]
    questions = []

    # ---------------------------------------------------------
    # 1. WHAT QUESTIONS
    # ---------------------------------------------------------

    first_event = events[0]

    questions.append({
        "question_type": "what",
        "question": "What sound occurs first in the audio?",
        "answer": event_label(first_event),
    })

    last_event = events[-1]

    questions.append({
        "question_type": "what",
        "question": "What sound occurs last in the audio?",
        "answer": event_label(last_event),
    })

    # ---------------------------------------------------------
    # 2. COUNTING QUESTIONS
    # ---------------------------------------------------------

    source_counts = {}

    for event in events:
        source = event["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in sorted(source_counts.items()):
        questions.append({
            "question_type": "counting",
            "question": f"How many {source} events occur in the audio?",
            "answer": str(count),
        })

    # ---------------------------------------------------------
    # 3. TEMPORAL QUESTIONS
    # ---------------------------------------------------------

    for i in range(len(events) - 1):
        current = events[i]
        following = events[i + 1]

        current_label = event_label(current)
        following_label = event_label(following)

        questions.append({
            "question_type": "temporal",
            "question": (
                f"What happens immediately after the {current_label}?"
            ),
            "answer": following_label,
        })

    # ---------------------------------------------------------
    # 4. CAUSAL / REASONING QUESTIONS
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 4. CAUSAL / REASONING QUESTIONS
    # ---------------------------------------------------------

    background = narrative.get("background", "None")

    if background != "None":
        questions.append({
            "question_type": "causal",
            "question_template": "background_environment",
            "question": (
                "Why does the audio suggest that a background environment "
                "is present?"
            ),
            "answer": "background environment",
            "evidence": (
                f"Background sound layer: {background}."
            ),
        })

    # Crowd reasoning
    crowd_events = [
        event for event in events
        if event["source"] == "crowd"
    ]

    if crowd_events:
        questions.append({
            "question_type": "causal",
            "question_template": "crowd_environment",
            "question": (
                "Why does the audio suggest a crowd environment?"
            ),
            "answer": "crowd environment",
            "evidence": (
                f"The audio contains {len(crowd_events)} "
                "crowd-related events."
            ),
        })
    return questions


def generate_split(project_root, split):
    narrative_dir = (
        project_root
        / "data"
        / "daqa"
        / "generated"
        / split
        / "narratives"
    )

    output_path = (
        project_root
        / "data"
        / "daqa"
        / "generated"
        / "qa"
        / f"daqa_local_{split}_qa.json"
    )

    narratives = load_narratives(narrative_dir)

    if not narratives:
        raise RuntimeError(
            f"No narrative files found in {narrative_dir}"
        )

    all_questions = []

    for narrative in narratives:
        questions = generate_questions(narrative)

        for question in questions:
            question["set"] = narrative["set"]
            question["audio_index"] = narrative["audio_index"]
            question["audio_filename"] = narrative["audio_filename"]

            all_questions.append(question)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "info": {
            "dataset": "DAQA-style local synthetic dataset",
            "version": "1.0",
            "split": split,
            "question_types": [
                "what",
                "counting",
                "temporal",
                "causal",
            ],
            "num_audio": len(narratives),
            "num_questions": len(all_questions),
        },
        "questions": all_questions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"{split} QA generation complete.")
    print("  Audio samples:", len(narratives))
    print("  Questions:", len(all_questions))
    print("  Output:", output_path)


def main():
    project_root = Path(__file__).resolve().parents[2]

    for split in SPLITS:
        generate_split(project_root, split)


if __name__ == "__main__":
    main()
