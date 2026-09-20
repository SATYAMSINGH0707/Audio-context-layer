import json
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000
MIN_EVENTS = 5
MAX_EVENTS = 8
BACKGROUND_PROBABILITY = 0.5

SPLITS = {
    "train": {
        "num_samples": 40,
        "seed": 0,
    },
    "val": {
        "num_samples": 10,
        "seed": 1,
    },
    "test": {
        "num_samples": 10,
        "seed": 2,
    },
}


def load_metadata(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_local_events(raw_dir, dataset):
    raw_dir = Path(raw_dir)

    files = sorted(
        f for f in raw_dir.glob("*.wav")
        if not f.name.startswith("._")
    )

    local_events = []

    for path in files:
        recording_id = path.stem
        event_id = recording_id.rsplit("_", 1)[0]

        if recording_id not in dataset["origins"]:
            raise RuntimeError(
                f"Local file {path.name} refers to unknown "
                f"DAQA recording {recording_id}"
            )

        if event_id not in dataset["sources"]:
            raise RuntimeError(
                f"Local file {path.name} refers to unknown "
                f"DAQA event {event_id} in sources"
            )

        if event_id not in dataset["actions"]:
            raise RuntimeError(
                f"Local file {path.name} refers to unknown "
                f"DAQA event {event_id} in actions"
            )

        local_events.append(path)

    if not local_events:
        raise RuntimeError(
            f"No WAV files found in {raw_dir}"
        )

    return local_events


def load_audio(path):
    audio, sample_rate = sf.read(
        path,
        dtype="int16",
    )

    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"{path} has sample rate {sample_rate}, "
            f"expected {SAMPLE_RATE}"
        )

    if audio.ndim != 1:
        raise RuntimeError(
            f"{path} is not mono."
        )

    return audio


def add_background(audio, background_path, rng):
    background = load_audio(
        background_path
    )

    if len(background) < len(audio):
        raise RuntimeError(
            f"Background {background_path} is shorter "
            f"than generated audio."
        )

    start = rng.integers(
        0,
        len(background) - len(audio) + 1,
    )

    background = background[
        start:start + len(audio)
    ]

    mixed = (
        audio.astype(np.int32)
        + background.astype(np.int32)
    )

    mixed = np.clip(
        mixed,
        -32768,
        32767,
    )

    return mixed.astype(np.int16)


def generate_sample(
    local_events,
    background_files,
    dataset,
    output_audio,
    output_narrative,
    sample_index,
    split,
    rng,
):
    num_events = int(
        rng.integers(
            MIN_EVENTS,
            MAX_EVENTS + 1,
        )
    )

    if split == "train" and sample_index == 39:
        crowd_events = [
            i
            for i, event_path in enumerate(local_events)
            if event_path.stem.rsplit("_", 1)[0] in {
                "c000",
                "c001",
                "c002",
            }
        ]

        if len(crowd_events) < 1:
            raise RuntimeError(
                "No local crowd recordings available."
            )

        crowd_indices = rng.choice(
            crowd_events,
            size=6,
            replace=True,
        )

        remaining_count = max(
            num_events - 6,
            0,
        )

        if remaining_count > 0:
            other_indices = rng.choice(
                len(local_events),
                size=remaining_count,
                replace=True,
            )
            selected_indices = np.concatenate(
                [crowd_indices, other_indices]
            )
        else:
            selected_indices = crowd_indices
    else:
        selected_indices = rng.choice(
            len(local_events),
            size=num_events,
            replace=True,
        )

    selected = [
        local_events[int(i)]
        for i in selected_indices
    ]

    audio_parts = []
    events = []
    current_sample = 0

    for order, event_path in enumerate(selected):

        event_audio = load_audio(
            event_path
        )

        event_id = (
            event_path.stem.rsplit(
                "_",
                1,
            )[0]
        )

        duration = (
            len(event_audio)
            / SAMPLE_RATE
        )

        audio_parts.append(
            event_audio
        )

        events.append(
            {
                "order": order,
                "event": event_id,
                "audio": event_path.name,
                "start_sample": current_sample,
                "end_sample": (
                    current_sample
                    + len(event_audio)
                ),
                "start_time": (
                    current_sample
                    / SAMPLE_RATE
                ),
                "end_time": (
                    current_sample
                    + len(event_audio)
                ) / SAMPLE_RATE,
                "duration": duration,
                "source": dataset[
                    "sources"
                ][event_id][
                    int(
                        rng.integers(
                            0,
                            len(
                                dataset[
                                    "sources"
                                ][event_id]
                            ),
                        )
                    )
                ],
                "action": dataset[
                    "actions"
                ][event_id][
                    int(
                        rng.integers(
                            0,
                            len(
                                dataset[
                                    "actions"
                                ][event_id]
                            ),
                        )
                    )
                ],
                "loudness": dataset[
                    "origins"
                ][event_path.stem][
                    "loudness"
                ],
            }
        )

        current_sample += len(
            event_audio
        )

    audio = np.concatenate(
        audio_parts
    )

    background_name = "None"

    if (
        background_files
        and rng.random()
        < BACKGROUND_PROBABILITY
    ):
        background_path = (
            background_files[
                int(
                    rng.integers(
                        0,
                        len(background_files),
                    )
                )
            ]
        )

        audio = add_background(
            audio,
            background_path,
            rng,
        )

        background_name = (
            background_path.stem
        )

    audio_path = Path(
        output_audio
    )

    narrative_path = Path(
        output_narrative
    )

    audio_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    narrative_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        audio_path,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    narrative = {
        "set": split,
        "audio_index": sample_index,
        "audio_filename": audio_path.name,
        "sample_rate": SAMPLE_RATE,
        "duration": (
            len(audio)
            / SAMPLE_RATE
        ),
        "background": background_name,
        "events": events,
    }

    with open(
        narrative_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            narrative,
            f,
            indent=2,
        )

    return narrative


def generate_split(
    split,
    config,
    local_events,
    background_files,
    dataset,
    project_root,
):
    num_samples = config[
        "num_samples"
    ]

    seed = config["seed"]

    rng = np.random.default_rng(
        seed
    )

    output_audio_dir = (
        project_root
        / "data"
        / "daqa"
        / "generated"
        / split
        / "audio"
    )

    output_narrative_dir = (
        project_root
        / "data"
        / "daqa"
        / "generated"
        / split
        / "narratives"
    )

    print()
    print(
        f"Generating {split}: "
        f"{num_samples} samples "
        f"(seed={seed})"
    )

    for i in range(num_samples):

        audio_path = (
            output_audio_dir
            / f"daqa_local_{i:06d}.wav"
        )

        narrative_path = (
            output_narrative_dir
            / f"daqa_local_{i:06d}.json"
        )

        narrative = generate_sample(
            local_events=local_events,
            background_files=background_files,
            dataset=dataset,
            output_audio=audio_path,
            output_narrative=narrative_path,
            sample_index=i,
            split=split,
            rng=rng,
        )

        print(
            f"[{i + 1}/{num_samples}] "
            f"{audio_path.name} | "
            f"{narrative['duration']:.2f}s | "
            f"{len(narrative['events'])} events | "
            f"background="
            f"{narrative['background']}"
        )

    print(
        f"{split} generation complete."
    )


def main():
    project_root = (
        Path(__file__).resolve().parents[2]
    )

    dataset_path = (
        project_root
        / "daqa"
        / "daqa-gen"
        / "daqa.json"
    )

    raw_dir = (
        project_root
        / "data"
        / "daqa"
        / "daqa-audio"
        / "raws"
    )

    background_dir = (
        project_root
        / "data"
        / "daqa"
        / "daqa-audio"
        / "backgrounds"
    )

    dataset = load_metadata(
        dataset_path
    )

    local_events = get_local_events(
        raw_dir,
        dataset,
    )

    background_files = sorted(
        f
        for f in background_dir.glob(
            "*.wav"
        )
        if not f.name.startswith("._")
    )

    print(
        "Local event recordings:",
        len(local_events),
    )

    print(
        "Background recordings:",
        len(background_files),
    )

    for split, config in SPLITS.items():
        generate_split(
            split=split,
            config=config,
            local_events=local_events,
            background_files=background_files,
            dataset=dataset,
            project_root=project_root,
        )

    print()
    print(
        "All dataset splits generated."
    )


if __name__ == "__main__":
    main()
