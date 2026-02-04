#!/usr/bin/env python3
"""
Convert results.csv to JSONL format with Problem and Solution fields.

Usage:
    python csv_to_jsonl.py path/to/results.csv

Output:
    Creates a .jsonl file at the same location as the input CSV.
"""

import argparse
import csv
import json
from pathlib import Path


def csv_to_jsonl(csv_path: str) -> str:
    """
    Convert results.csv to JSONL format (only ACCEPTED results).

    Args:
        csv_path: Path to the results.csv file

    Returns:
        Path to the output JSONL file
    """
    csv_path = Path(csv_path)
    jsonl_path = csv_path.with_suffix('.jsonl')

    # Increase CSV field size limit to handle large model responses
    csv.field_size_limit(10 * 1024 * 1024)  # 10MB limit

    with open(csv_path, 'r', encoding='utf-8') as f_in, \
         open(jsonl_path, 'w', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        count = 0
        skipped = 0

        for row in reader:
            # Only include ACCEPTED results
            if row.get('result_type', '') != 'ACCEPTED':
                skipped += 1
                continue

            entry = {
                "Problem": row.get('prompt', ''),
                "Solution": row.get('response', ''),
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1

    print(f"Converted {count} ACCEPTED examples to {jsonl_path} (skipped {skipped} non-ACCEPTED)")
    return str(jsonl_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert results.csv to JSONL format with Problem and Solution fields'
    )
    parser.add_argument('csv_path', help='Path to the results.csv file')
    args = parser.parse_args()

    csv_to_jsonl(args.csv_path)


if __name__ == '__main__':
    main()
