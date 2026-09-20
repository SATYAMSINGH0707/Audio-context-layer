from pathlib import Path

from datasets.daqa_dataset import DAQADataset
from datasets.qa_collator import AudioQACollator


TRAIN_QA_PATH = (
    "data/daqa/generated/qa/daqa_local_train_qa.json"
)

VAL_QA_PATH = (
    "data/daqa/generated/qa/daqa_local_val_qa.json"
)

TRAIN_AUDIO_DIR = (
    "data/daqa/generated/train/audio"
)

VAL_AUDIO_DIR = (
    "data/daqa/generated/val/audio"
)

TENSORBOARD_DIR = Path(
    "checkpoints/tensorboard"
)


def main():

    print("=" * 60)
    print("TRAINING SETUP VERIFICATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Training dataset
    # -----------------------------------------------------

    train_dataset = DAQADataset(
        qa_path=TRAIN_QA_PATH,
        audio_dir=TRAIN_AUDIO_DIR,
    )

    print(
        f"Train QA pairs: {len(train_dataset)}"
    )

    if len(train_dataset) != 475:
        raise RuntimeError(
            "Expected 475 training QA pairs, "
            f"found {len(train_dataset)}."
        )

    # -----------------------------------------------------
    # Validation dataset
    # -----------------------------------------------------

    val_dataset = DAQADataset(
        qa_path=VAL_QA_PATH,
        audio_dir=VAL_AUDIO_DIR,
    )

    print(
        f"Validation QA pairs: {len(val_dataset)}"
    )

    if len(val_dataset) != 123:
        raise RuntimeError(
            "Expected 123 validation QA pairs, "
            f"found {len(val_dataset)}."
        )

    # -----------------------------------------------------
    # Training answer vocabulary
    # -----------------------------------------------------

    train_collator = AudioQACollator(
        qa_path=TRAIN_QA_PATH,
        sample_rate=16000,
    )

    print(
        f"Training answer classes: "
        f"{len(train_collator.answers)}"
    )

    if len(train_collator.answers) != 20:
        raise RuntimeError(
            "Expected 20 training answer classes, "
            f"found {len(train_collator.answers)}."
        )

    # -----------------------------------------------------
    # Validation answer vocabulary
    # -----------------------------------------------------

    val_collator = AudioQACollator(
        qa_path=VAL_QA_PATH,
        sample_rate=16000,
    )

    val_collator.answers = train_collator.answers
    val_collator.answer_to_id = (
        train_collator.answer_to_id
    )

    print(
        "Validation uses training vocabulary: PASS"
    )

    # -----------------------------------------------------
    # Audio directories
    # -----------------------------------------------------

    train_audio_path = Path(
        TRAIN_AUDIO_DIR
    )

    val_audio_path = Path(
        VAL_AUDIO_DIR
    )

    if not train_audio_path.exists():
        raise RuntimeError(
            f"Training audio directory missing: "
            f"{train_audio_path}"
        )

    if not val_audio_path.exists():
        raise RuntimeError(
            f"Validation audio directory missing: "
            f"{val_audio_path}"
        )

    print(
        "Training audio directory: PASS"
    )

    print(
        "Validation audio directory: PASS"
    )

    # -----------------------------------------------------
    # TensorBoard directory
    # -----------------------------------------------------

    TENSORBOARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"TensorBoard directory: "
        f"{TENSORBOARD_DIR}"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("=" * 60)
    print("TRAINING SETUP VERIFICATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
