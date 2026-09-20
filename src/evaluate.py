import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.daqa_dataset import DAQADataset
from datasets.qa_collator import AudioQACollator
from models.audio_qa_model import AudioQuestionAnsweringModel


CHECKPOINT_PATH = Path(
    "checkpoints/audio_qa_best.pt"
)

TEST_QA_PATH = Path(
    "data/daqa/generated/qa/daqa_local_test_qa.json"
)

TEST_AUDIO_DIR = Path(
    "data/daqa/generated/test/audio"
)

RESULTS_PATH = Path(
    "results/test_evaluation.json"
)

BATCH_SIZE = 2


def main():

    print("=" * 60)
    print("HELD-OUT TEST EVALUATION")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not TEST_QA_PATH.exists():
        raise FileNotFoundError(
            f"Test QA file not found: {TEST_QA_PATH}"
        )

    if not TEST_AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Test audio directory not found: "
            f"{TEST_AUDIO_DIR}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    answers = checkpoint["answers"]
    answer_to_id = checkpoint["answer_to_id"]

    print(
        f"Test QA file: {TEST_QA_PATH}"
    )

    print(
        f"Test audio directory: "
        f"{TEST_AUDIO_DIR}"
    )

    print(
        f"Answer classes: {len(answers)}"
    )

    dataset = DAQADataset(
        qa_path=TEST_QA_PATH,
        audio_dir=TEST_AUDIO_DIR,
    )

    collator = AudioQACollator(
        qa_path=TEST_QA_PATH,
    )

    # Use the training checkpoint vocabulary.
    collator.answers = answers
    collator.answer_to_id = answer_to_id

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    if len(dataset) != 116:
        raise RuntimeError(
            f"Expected 116 test QA pairs, "
            f"found {len(dataset)}."
        )

    model = AudioQuestionAnsweringModel(
        num_answers=len(answers),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        f"Test QA pairs: {len(dataset)}"
    )

    print(
        f"Test batches: {len(loader)}"
    )

    correct = 0
    total = 0

    type_correct = {}
    type_total = {}

    errors = []

    with torch.no_grad():

        for batch in loader:

            input_values = batch[
                "input_values"
            ].to(device)

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            outputs = model(
                input_values=input_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs[
                "answer_logits"
            ]

            predictions = logits.argmax(
                dim=-1
            )

            for i in range(
                len(predictions)
            ):

                predicted_id = (
                    predictions[i]
                    .item()
                )

                label_id = (
                    labels[i]
                    .item()
                )

                predicted_answer = (
                    answers[predicted_id]
                )

                true_answer = (
                    answers[label_id]
                )

                question_type = (
                    batch["question_type"][i]
                )

                question = (
                    batch["question"][i]
                )

                audio_filename = (
                    dataset.questions[
                        total
                    ]["audio_filename"]
                )

                total += 1

                if predicted_id == label_id:
                    correct += 1

                type_total[
                    question_type
                ] = (
                    type_total.get(
                        question_type,
                        0,
                    )
                    + 1
                )

                if predicted_id == label_id:
                    type_correct[
                        question_type
                    ] = (
                        type_correct.get(
                            question_type,
                            0,
                        )
                        + 1
                    )

                else:
                    errors.append(
                        {
                            "audio_filename":
                                audio_filename,
                            "question":
                                question,
                            "question_type":
                                question_type,
                            "true_answer":
                                true_answer,
                            "predicted_answer":
                                predicted_answer,
                        }
                    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Overall test accuracy: "
        f"{accuracy:.4f}"
    )

    print()
    print("Accuracy by question type:")

    type_results = {}

    for question_type in sorted(
        type_total.keys()
    ):

        correct_count = type_correct.get(
            question_type,
            0,
        )

        total_count = type_total[
            question_type
        ]

        type_accuracy = (
            correct_count / total_count
        )

        type_results[
            question_type
        ] = {
            "correct":
                correct_count,
            "total":
                total_count,
            "accuracy":
                type_accuracy,
        }

        print(
            f"  {question_type}: "
            f"{correct_count}/"
            f"{total_count} "
            f"({type_accuracy:.4f})"
        )

    print()
    print(
        f"Total errors: {len(errors)}"
    )

    results = {
        "checkpoint":
            str(CHECKPOINT_PATH),
        "test_qa_path":
            str(TEST_QA_PATH),
        "test_audio_dir":
            str(TEST_AUDIO_DIR),
        "total_questions":
            total,
        "correct":
            correct,
        "accuracy":
            accuracy,
        "accuracy_by_question_type":
            type_results,
        "errors":
            errors,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        f"Results saved: {RESULTS_PATH}"
    )

    if total != len(dataset):
        raise RuntimeError(
            "Evaluation count mismatch."
        )

    print()
    print(
        "HELD-OUT TEST EVALUATION PASSED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
