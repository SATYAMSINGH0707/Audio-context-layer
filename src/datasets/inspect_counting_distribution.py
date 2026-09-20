import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("data/daqa/generated")


def inspect_split(split):
    narrative_dir = ROOT / split / "narratives"

    counts = defaultdict(Counter)

    for path in sorted(narrative_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            narrative = json.load(f)

        event_counts = Counter(
            event["source"]
            for event in narrative["events"]
        )

        for event_type, count in event_counts.items():
            counts[event_type][count] += 1

    print(f"\n{split.upper()}")

    for event_type in sorted(counts):
        distribution = dict(sorted(counts[event_type].items()))
        print(f"  {event_type}: {distribution}")


def main():
    for split in ["train", "val", "test"]:
        inspect_split(split)


if __name__ == "__main__":
    main()
