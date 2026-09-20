from pathlib import Path

import torch


CHECKPOINT_PATH = Path(
    "checkpoints/audio_qa_best.pt"
)


def main():

    print("=" * 60)
    print("BEST CHECKPOINT INTEGRITY TEST")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    required_keys = {
        "model_state_dict",
        "answer_to_id",
        "answers",
        "seed",
        "batch_size",
        "learning_rate",
        "epochs",
    }

    missing_keys = (
        required_keys
        - set(checkpoint.keys())
    )

    if missing_keys:
        raise RuntimeError(
            f"Missing checkpoint keys: {missing_keys}"
        )

    print(
        f"Checkpoint path: "
        f"{CHECKPOINT_PATH.resolve()}"
    )

    print(
        f"Number of answers: "
        f"{len(checkpoint['answers'])}"
    )

    if len(checkpoint["answers"]) != 20:
        raise RuntimeError(
            "Expected 20 answer classes."
        )

    print(
        "Answer vocabulary: "
        f"{checkpoint['answers']}"
    )

    print(
        f"Stored seed: "
        f"{checkpoint['seed']}"
    )

    print(
        f"Stored batch size: "
        f"{checkpoint['batch_size']}"
    )

    print(
        f"Stored learning rate: "
        f"{checkpoint['learning_rate']}"
    )

    print(
        f"Stored epochs: "
        f"{checkpoint['epochs']}"
    )

    if checkpoint["epochs"] != 3:
        raise RuntimeError(
            "Expected 3 training epochs."
        )

    model_state = checkpoint[
        "model_state_dict"
    ]

    if not model_state:
        raise RuntimeError(
            "Model state dictionary is empty."
        )

    print(
        f"Model parameters: "
        f"{len(model_state)} tensors"
    )

    print(
        "Model state dictionary: PASS"
    )

    print("=" * 60)
    print("BEST CHECKPOINT INTEGRITY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
