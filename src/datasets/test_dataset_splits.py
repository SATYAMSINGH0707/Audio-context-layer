from pathlib import Path


BASE_DIR = Path("data/daqa/generated")

EXPECTED = {
    "train": 40,
    "val": 10,
    "test": 10,
}


for split, expected_count in EXPECTED.items():
    audio_dir = BASE_DIR / split / "audio"
    narrative_dir = BASE_DIR / split / "narratives"

    audio_files = sorted(audio_dir.glob("*.wav"))
    narrative_files = sorted(narrative_dir.glob("*.json"))

    print(f"{split}:")
    print(f"  Audio files: {len(audio_files)}")
    print(f"  Narrative files: {len(narrative_files)}")
    print(f"  Expected: {expected_count}")

    assert len(audio_files) == expected_count
    assert len(narrative_files) == expected_count

    audio_names = {p.stem for p in audio_files}
    narrative_names = {p.stem for p in narrative_files}

    assert audio_names == narrative_names

    print("  Alignment: PASS")


print("\nDATASET SPLIT VALIDATION PASSED")
