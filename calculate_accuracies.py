import csv
import os
import re
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"
OUTPUT_FILE = Path(__file__).parent / "scores.csv"


def extract_model_name(dirname: str) -> str:
    """Extract model name by removing the trailing _YYYYMMDD_HHMMSS."""
    return re.sub(r"_\d{8}_\d{6}$", "", dirname)


def parse_summary(summary_path: Path) -> dict | None:
    """Parse a summary.csv and return accuracy and macro_average."""
    text = summary_path.read_text()
    lines = text.strip().splitlines()

    accuracy = None
    macro_avg = None

    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if parts[0] == "ACCEPTED" and len(parts) >= 3:
            accuracy = float(parts[2])
        if parts[0] == "macro_average_percentage" and len(parts) >= 2:
            macro_avg = float(parts[1])

    if accuracy is None or macro_avg is None:
        return None
    return {"accuracy": accuracy, "macro_average": macro_avg}


def main():
    results = []
    for entry in sorted(LOGS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        summary = entry / "summary.csv"
        if not summary.exists():
            continue
        model_name = extract_model_name(entry.name)
        scores = parse_summary(summary)
        if scores is None:
            print(f"WARNING: could not parse {summary}")
            continue
        results.append({
            "name": model_name,
            "accuracy": scores["accuracy"],
            "macro_average": scores["macro_average"],
        })

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "accuracy", "macro_average"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} rows to {OUTPUT_FILE}")
    for r in results:
        print(f"  {r['name']}: accuracy={r['accuracy']}%, macro_avg={r['macro_average']}%")


if __name__ == "__main__":
    main()