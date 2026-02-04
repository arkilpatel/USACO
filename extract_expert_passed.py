#!/usr/bin/env python3
"""
Extract problem IDs with ACCEPTED results from results.csv and save to JSON.

Usage:
    python extract_expert_passed.py path/to/results.csv

Output:
    Creates data/expert_passed_problems.json with the list of accepted problem IDs.
"""

import argparse
import csv
import json
import os
from pathlib import Path


def extract_expert_passed(csv_path: str, output_path: str = None) -> dict:
    """
    Extract problem IDs with ACCEPTED results from results.csv.

    Args:
        csv_path: Path to the results.csv file
        output_path: Path to output JSON file (default: data/expert_passed_problems.json)

    Returns:
        Dictionary mapping problem_id to result info
    """
    if output_path is None:
        # Default to data/expert_passed_problems.json relative to repo root
        repo_root = Path(__file__).parent
        output_path = repo_root / 'data' / 'expert_passed_problems.json'
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Increase CSV field size limit to handle large model responses
    csv.field_size_limit(10 * 1024 * 1024)  # 10MB limit

    accepted_problems = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            result_type = row.get('result_type', '')
            problem_id = row.get('problem_id', '')

            if result_type == 'ACCEPTED' and problem_id:
                # Store problem info (can extend with more fields if needed)
                accepted_problems[problem_id] = {
                    'num_passed': int(row.get('num_passed', 0)),
                    'num_tests': int(row.get('num_tests', 0)),
                    'percentage_passed': float(row.get('percentage_passed', 0)),
                }

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(accepted_problems, f, indent=2)

    print(f"Extracted {len(accepted_problems)} ACCEPTED problems")
    print(f"Saved to: {output_path}")

    return accepted_problems


def main():
    parser = argparse.ArgumentParser(
        description='Extract ACCEPTED problem IDs from results.csv'
    )
    parser.add_argument('csv_path', help='Path to the results.csv file')
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output JSON path (default: data/expert_passed_problems.json)'
    )
    args = parser.parse_args()

    extract_expert_passed(args.csv_path, args.output)


if __name__ == '__main__':
    main()
