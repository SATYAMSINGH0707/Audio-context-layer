# Audio Context Layer — Audio Question Answering

A small-scale research prototype for **Audio Question Answering (AudioQA)** using an Audio Context Layer to model temporal audio context and combine it with question representations.

This project is inspired by the **DAQA (Temporal Reasoning via Audio Question Answering)** benchmark and implements a reproducible local/synthetic DAQA-style subset for experimentation.

## Project Overview

The model takes an audio clip and a natural-language question and predicts the answer from a fixed vocabulary of 20 answer classes.

### Architecture

```text
Audio
  ↓
AST (Audio Spectrogram Transformer)
  ↓
Audio Context Layer
(Temporal Transformer)
  ↓
Audio Context Representation
  ↓
Cross-Attention Fusion ← BERT Question Encoder
  ↓
Pooled Representation
  ↓
20-Class Answer Classifier
```

### Components

* **Audio encoder:** `MIT/ast-finetuned-audioset-10-10-0.4593`
* **Question encoder:** `google-bert/bert-base-uncased`
* **Audio Context Layer:** 2-layer Transformer encoder with 12 attention heads
* **Fusion:** Question-to-audio cross-attention
* **Answer head:** 20-class classification
* **Framework:** PyTorch + torchaudio
* **GPU tested:** NVIDIA RTX A5000 24 GB

## Dataset

This repository uses a small DAQA-style local subset rather than the complete official DAQA benchmark.

### Dataset size

| Split      | Audio samples | QA pairs |
| ---------- | ------------: | -------: |
| Train      |            40 |      475 |
| Validation |            10 |      123 |
| Test       |            10 |      116 |
| **Total**  |        **60** |  **714** |

The QA dataset contains four question types:

* What
* Counting
* Temporal
* Causal

There are **20 answer classes**.

The generated dataset uses fixed random seeds for reproducibility:

* Train: seed `0`
* Validation: seed `1`
* Test: seed `2`

Audio files and large model checkpoints are intentionally excluded from this Git repository. The complete packaged dataset is provided separately for submission.

## Repository Structure

```text
audio_context_layer/
├── checkpoints/
├── configs/
├── data/
│   └── daqa/
│       └── generated/
├── results/
│   └── test_evaluation.json
├── src/
│   ├── datasets/
│   ├── models/
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## Installation

Create the project environment and install the required packages.

```bash
conda create -n audioqa python=3.10
conda activate audioqa
pip install -r requirements.txt
```

The experiments were run with CUDA-enabled PyTorch on an NVIDIA RTX A5000 GPU.

## Dataset Generation

The local DAQA-style audio dataset can be generated with:

```bash
python src/datasets/generate_daqa_local.py
```

Then generate the QA pairs:

```bash
python src/datasets/generate_qa.py
```

The generated data is placed under:

```text
data/daqa/generated/
```

with separate train, validation, and test splits.

## Training

Train the model with:

```bash
python src/train.py
```

The training configuration used for the reported experiment was:

* Batch size: `2`
* Learning rate: `1e-5`
* Epochs: `3`
* Random seed: `42`
* Number of answer classes: `20`

The best model is selected using validation accuracy.

Training history is saved to:

```text
checkpoints/audio_qa_history.json
```

TensorBoard logs are written to:

```text
checkpoints/tensorboard/
```

TensorBoard can be launched with:

```bash
tensorboard --logdir checkpoints/tensorboard
```

## Evaluation

Evaluate the held-out test set using:

```bash
python src/evaluate.py
```

The evaluation uses the best validation checkpoint and does not use the test set during training.

Results are saved to:

```text
results/test_evaluation.json
```

## Experimental Results

The reported held-out test evaluation contains **116 QA pairs**.

| Question type | Correct |   Total |   Accuracy |
| ------------- | ------: | ------: | ---------: |
| Causal        |      14 |      14 |    100.00% |
| Counting      |      10 |      25 |     40.00% |
| Temporal      |       9 |      57 |     15.79% |
| What          |       5 |      20 |     25.00% |
| **Overall**   |  **38** | **116** | **32.76%** |

The validation accuracy reached **35.77%** at the best checkpoint (epoch 1).

The training accuracy increased during the three epochs, while validation accuracy did not improve after the first epoch. This provides evidence of limited generalization in this small-scale prototype.

## Limitations

This project is intended as a **small-scale research prototype / proof of concept**, not as a reproduction of the full official DAQA benchmark.

Important limitations include:

* The dataset is substantially smaller than the official DAQA benchmark.
* Only a subset of the available audio material is used locally.
* Temporal question performance is relatively weak in the reported experiment.
* Causal questions are based on metadata/context-grounded relationships rather than causal inference from acoustic signals.
* The answer space is restricted to 20 classes.
* The pretrained AST and BERT encoders are used as components of the prototype rather than being extensively fine-tuned or systematically optimized.
* Results should therefore not be interpreted as directly comparable to results obtained on the full official DAQA benchmark.

## Reproducibility

The experiment uses fixed dataset-generation seeds and a fixed training seed.

The repository contains the code required to:

1. Generate the local dataset.
2. Generate QA pairs.
3. Train the model.
4. Evaluate the held-out test split.
5. Produce the reported evaluation JSON.

Large audio files and model checkpoints are excluded from Git to keep the code repository lightweight.

## Original DAQA Benchmark

This work is inspired by the **DAQA: Temporal Reasoning via Audio Question Answering** benchmark by Fayek and Johnson.

Official DAQA provides large-scale audio question-answering data for temporal reasoning over audio.

Reference:

Fayek, H. M. and Johnson, J. (2019). *Temporal Reasoning via Audio Question Answering.*

The original DAQA resources are used only as the source/inspiration for the small-scale experimental setup. This repository does **not** contain the complete official DAQA dataset.

## License / Data Notice

The original DAQA resources are subject to their original licensing and attribution requirements. Please consult the official DAQA repository and dataset documentation before redistributing any source audio.

The code and generated project materials in this repository are provided for research and educational use.
