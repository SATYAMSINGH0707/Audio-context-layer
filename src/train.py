
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.daqa_dataset import DAQADataset
from datasets.qa_collator import AudioQACollator
from models.audio_qa_model import AudioQuestionAnsweringModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42

BATCH_SIZE = 2
LEARNING_RATE = 1e-5
EPOCHS = 3

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

CHECKPOINT_DIR = Path("checkpoints")

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "audio_qa_best.pt"
)

HISTORY_PATH = (
    CHECKPOINT_DIR / "audio_qa_history.json"
)

TENSORBOARD_DIR = (
    CHECKPOINT_DIR / "tensorboard"
)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Build validation collator using training vocabulary
# ---------------------------------------------------------

def build_validation_collator(
    train_collator,
    val_qa_path,
):
    val_collator = AudioQACollator(
        qa_path=val_qa_path,
        sample_rate=16000,
    )

    # IMPORTANT:
    # Validation must use exactly the same answer
    # vocabulary and answer-to-ID mapping as training.
    val_collator.answers = train_collator.answers
    val_collator.answer_to_id = train_collator.answer_to_id

    return val_collator


# ---------------------------------------------------------
# Run one epoch
# ---------------------------------------------------------

def run_epoch(
    model,
    dataloader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:

        input_values = batch["input_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(
                input_values=input_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["answer_logits"]

            loss = criterion(
                logits,
                labels,
            )

            predictions = logits.argmax(dim=1)

            correct = (
                predictions == labels
            ).sum().item()

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += correct
        total_examples += batch_size

    average_loss = (
        total_loss / total_examples
    )

    accuracy = (
        total_correct / total_examples
    )

    return average_loss, accuracy


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    set_seed(SEED)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    # -----------------------------------------------------
    # Create output directories
    # -----------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TENSORBOARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # TensorBoard
    # -----------------------------------------------------

    writer = SummaryWriter(
        log_dir=str(TENSORBOARD_DIR)
    )

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = DAQADataset(
        qa_path=TRAIN_QA_PATH,
        audio_dir=TRAIN_AUDIO_DIR,
    )

    val_dataset = DAQADataset(
        qa_path=VAL_QA_PATH,
        audio_dir=VAL_AUDIO_DIR,
    )

    print()
    print(
        f"Training QA pairs: {len(train_dataset)}"
    )

    print(
        f"Validation QA pairs: {len(val_dataset)}"
    )

    # -----------------------------------------------------
    # Dataset size checks
    # -----------------------------------------------------

    if len(train_dataset) != 475:
        raise RuntimeError(
            "Expected 475 training QA pairs, "
            f"found {len(train_dataset)}."
        )

    if len(val_dataset) != 123:
        raise RuntimeError(
            "Expected 123 validation QA pairs, "
            f"found {len(val_dataset)}."
        )

    # -----------------------------------------------------
    # Collators
    # -----------------------------------------------------

    train_collator = AudioQACollator(
        qa_path=TRAIN_QA_PATH,
        sample_rate=16000,
    )

    val_collator = build_validation_collator(
        train_collator=train_collator,
        val_qa_path=VAL_QA_PATH,
    )

    num_answers = len(
        train_collator.answers
    )

    print(
        f"Answer classes: {num_answers}"
    )

    if num_answers != 20:
        raise RuntimeError(
            "Expected exactly 20 training answer classes, "
            f"found {num_answers}."
        )

    # -----------------------------------------------------
    # Verify validation vocabulary matches training
    # -----------------------------------------------------

    if val_collator.answer_to_id != (
        train_collator.answer_to_id
    ):
        raise RuntimeError(
            "Validation answer vocabulary does not "
            "match the training vocabulary."
        )

    print(
        "Training/validation answer vocabulary: MATCH"
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_collator,
    )

    print(
        f"Training batches per epoch: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches per epoch: "
        f"{len(val_loader)}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print()
    print("Loading model...")

    model = AudioQuestionAnsweringModel(
        num_answers=num_answers,
    )

    model = model.to(device)

    print("Model loaded.")

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -----------------------------------------------------
    # Training history
    # -----------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -1.0
    best_epoch = None

    # -----------------------------------------------------
    # Record configuration in TensorBoard
    # -----------------------------------------------------

    writer.add_text(
        "Configuration/Info",
        (
            f"Train QA pairs: {len(train_dataset)}\n"
            f"Validation QA pairs: {len(val_dataset)}\n"
            f"Answer classes: {num_answers}\n"
            f"Batch size: {BATCH_SIZE}\n"
            f"Learning rate: {LEARNING_RATE}\n"
            f"Epochs: {EPOCHS}\n"
            f"Seed: {SEED}"
        ),
    )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        print()
        print(
            "=" * 60
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            "=" * 60
        )

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        train_loss, train_accuracy = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        val_loss, val_accuracy = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # -------------------------------------------------
        # Store history
        # -------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        # -------------------------------------------------
        # Console output
        # -------------------------------------------------

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{val_accuracy:.4f}"
        )

        # -------------------------------------------------
        # TensorBoard logging
        # -------------------------------------------------

        writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch,
        )

        writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch,
        )

        writer.add_scalar(
            "Accuracy/Train",
            train_accuracy,
            epoch,
        )

        writer.add_scalar(
            "Accuracy/Validation",
            val_accuracy,
            epoch,
        )

        writer.add_scalar(
            "Learning_Rate",
            LEARNING_RATE,
            epoch,
        )

        writer.flush()

        # -------------------------------------------------
        # Save best validation checkpoint
        # -------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            checkpoint = {
                "model_state_dict": (
                    model.state_dict()
                ),
                "answer_to_id": (
                    train_collator.answer_to_id
                ),
                "answers": (
                    train_collator.answers
                ),
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "best_epoch": best_epoch,
                "best_val_accuracy": (
                    best_val_accuracy
                ),
                "history": history,
            }

            torch.save(
                checkpoint,
                BEST_CHECKPOINT_PATH,
            )

            print()
            print(
                "New best validation result."
            )

            print(
                f"Best validation accuracy: "
                f"{best_val_accuracy:.4f}"
            )

            print(
                f"Best epoch: {best_epoch}"
            )

            print(
                "Best checkpoint saved:"
            )

            print(
                f"  {BEST_CHECKPOINT_PATH}"
            )

    # -----------------------------------------------------
    # Save training history
    # -----------------------------------------------------

    history_output = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "num_answers": num_answers,
        "best_epoch": best_epoch,
        "best_val_accuracy": (
            best_val_accuracy
        ),
        "history": history,
    }

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history_output,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Close TensorBoard writer
    # -----------------------------------------------------

    writer.close()

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "Training complete."
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best checkpoint: "
        f"{BEST_CHECKPOINT_PATH}"
    )

    print(
        f"Training history: "
        f"{HISTORY_PATH}"
    )

    print(
        f"TensorBoard logs: "
        f"{TENSORBOARD_DIR}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()

