import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

LOGS_DIR = Path(__file__).parent / "logs"
SCORES_DIR = Path(__file__).parent / "scores"
PLOT_DIR = SCORES_DIR / "plot_scores"

FIELDNAMES = ["name", "accuracy", "macro_average"]
REVISIONS_PREFIX = "revisions_"


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


def write_scores(output_path: Path, results: list[dict]):
    """Write a list of result dicts to a scores CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {output_path}")
    for r in results:
        print(f"  {r['name']}: accuracy={r['accuracy']}%, macro_avg={r['macro_average']}%")


def collect_seed_results(directory: Path, name: str) -> dict | None:
    """Collect per-seed accuracy and macro_average from seed-* subdirectories.

    Returns a dict with name, per-seed values, or None if no seeds found.
    e.g. {"name": "model-x", "seed_42_accuracy": 9.77, "seed_42_macro_average": 13.81, ...}
    """
    result = {"name": name}
    found_any = False
    for seed_dir in sorted(directory.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed-"):
            continue
        summary = seed_dir / "summary.csv"
        if not summary.exists():
            continue
        scores = parse_summary(summary)
        if scores is None:
            print(f"WARNING: could not parse {summary}")
            continue
        seed = seed_dir.name.removeprefix("seed-")
        result[f"seed_{seed}_accuracy"] = scores["accuracy"]
        result[f"seed_{seed}_macro_average"] = scores["macro_average"]
        found_any = True
    return result if found_any else None


def write_seed_scores(output_path: Path, results: list[dict]):
    """Write seed-aggregated scores CSV with per-seed columns and aggregates."""
    # Collect all unique seed numbers
    all_seeds: set[str] = set()
    for r in results:
        for k in r:
            if k.startswith("seed_") and k.endswith("_accuracy"):
                all_seeds.add(k.removeprefix("seed_").removesuffix("_accuracy"))
    seeds = sorted(all_seeds, key=int)

    # Build per-seed column pairs and compute aggregates
    seed_cols = []
    for s in seeds:
        seed_cols.extend([f"seed_{s}_accuracy", f"seed_{s}_macro_average"])

    for r in results:
        acc_vals = [r[f"seed_{s}_accuracy"] for s in seeds if f"seed_{s}_accuracy" in r]
        macro_vals = [r[f"seed_{s}_macro_average"] for s in seeds if f"seed_{s}_macro_average" in r]
        n = len(acc_vals)
        r["num_seeds"] = n
        if n > 0:
            acc_mean = sum(acc_vals) / n
            macro_mean = sum(macro_vals) / n
            r["mean_accuracy"] = round(acc_mean, 2)
            r["mean_macro_average"] = round(macro_mean, 2)
            if n > 1:
                acc_var = sum((v - acc_mean) ** 2 for v in acc_vals) / (n - 1)
                macro_var = sum((v - macro_mean) ** 2 for v in macro_vals) / (n - 1)
                r["std_error_accuracy"] = round(math.sqrt(acc_var / n), 2)
                r["std_error_macro_average"] = round(math.sqrt(macro_var / n), 2)
            else:
                r["std_error_accuracy"] = 0.0
                r["std_error_macro_average"] = 0.0
        else:
            r["mean_accuracy"] = ""
            r["mean_macro_average"] = ""
            r["std_error_accuracy"] = ""
            r["std_error_macro_average"] = ""

    fieldnames = (
        ["name"]
        + seed_cols
        + ["mean_accuracy", "std_error_accuracy", "mean_macro_average", "std_error_macro_average", "num_seeds"]
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {output_path}")
    for r in results:
        print(
            f"  {r['name']}: acc={r['mean_accuracy']}% (se={r['std_error_accuracy']}%), "
            f"macro={r['mean_macro_average']}% (se={r['std_error_macro_average']}%), n={r['num_seeds']}"
        )


def plot_seed_scores(output_path: Path, results: list[dict], metric: str):
    """Save a horizontal bar chart of mean ± std_error for each model.

    metric should be 'accuracy' or 'macro_average'.
    """
    mean_key = f"mean_{metric}"
    se_key = f"std_error_{metric}"
    rows = [r for r in results if isinstance(r.get(mean_key), (int, float))]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r[mean_key])

    names = [r["name"] for r in rows]
    means = [r[mean_key] for r in rows]
    errors = [r.get(se_key) or 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(names))))
    ax.barh(names, means, xerr=errors, capsize=3, color="steelblue", ecolor="black")
    ax.set_xlabel(f"{metric.replace('_', ' ').title()} (%)")
    ax.set_title(f"USACO — {metric.replace('_', ' ').title()}")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_revision_scores(output_path: Path, results: list[dict], metric: str, model_name: str):
    """Save a scatter plot of mean ± std_error vs. step for revisions.

    metric should be 'accuracy' or 'macro_average'.
    """
    mean_key = f"mean_{metric}"
    se_key = f"std_error_{metric}"
    rows = [r for r in results if isinstance(r.get(mean_key), (int, float))]
    if not rows:
        return

    parsed = []
    for r in rows:
        m = re.search(r"(\d+)", r["name"])
        if m:
            parsed.append((int(m.group(1)), r[mean_key], r.get(se_key) or 0.0))
    if not parsed:
        return
    parsed.sort(key=lambda t: t[0])

    steps = [p[0] for p in parsed]
    means = [p[1] for p in parsed]
    errors = [p[2] for p in parsed]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(steps, means, yerr=errors, fmt="o", capsize=4,
                color="steelblue", ecolor="black", markersize=6)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
    ax.set_title(f"{model_name} — {metric.replace('_', ' ').title()}")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    SCORES_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)

    # --- Process top-level model directories ---
    results = []
    for entry in sorted(LOGS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(REVISIONS_PREFIX):
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

    write_scores(SCORES_DIR / "scores.csv", results)

    # --- Process top-level model directories (seed-aggregated) ---
    seed_results = []
    for entry in sorted(LOGS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(REVISIONS_PREFIX):
            continue
        model_name = extract_model_name(entry.name)
        row = collect_seed_results(entry, model_name)
        if row is not None:
            seed_results.append(row)

    if seed_results:
        write_seed_scores(SCORES_DIR / "scores_seeds.csv", seed_results)
        plot_seed_scores(PLOT_DIR / "accuracy_scores.png", seed_results, "accuracy")
        plot_seed_scores(PLOT_DIR / "macro_average_scores.png", seed_results, "macro_average")

    # --- Process revision directories ---
    for entry in sorted(LOGS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if not entry.name.startswith(REVISIONS_PREFIX):
            continue
        model_name = entry.name[len(REVISIONS_PREFIX):]
        revision_results = []
        for step_dir in sorted(entry.iterdir()):
            if not step_dir.is_dir():
                continue
            summary = step_dir / "summary.csv"
            if not summary.exists():
                continue
            scores = parse_summary(summary)
            if scores is None:
                print(f"WARNING: could not parse {summary}")
                continue
            revision_results.append({
                "name": step_dir.name,
                "accuracy": scores["accuracy"],
                "macro_average": scores["macro_average"],
            })

        output_path = SCORES_DIR / f"{model_name}_scores.csv"
        write_scores(output_path, revision_results)

        # Seed-aggregated revision scores
        revision_seed_results = []
        for step_dir in sorted(entry.iterdir()):
            if not step_dir.is_dir():
                continue
            row = collect_seed_results(step_dir, step_dir.name)
            if row is not None:
                revision_seed_results.append(row)

        if revision_seed_results:
            output_path = SCORES_DIR / f"{model_name}_scores_seeds.csv"
            write_seed_scores(output_path, revision_seed_results)
            plot_revision_scores(
                PLOT_DIR / f"{model_name}_accuracy_scores.png",
                revision_seed_results, "accuracy", model_name,
            )
            plot_revision_scores(
                PLOT_DIR / f"{model_name}_macro_average_scores.png",
                revision_seed_results, "macro_average", model_name,
            )


if __name__ == "__main__":
    main()
