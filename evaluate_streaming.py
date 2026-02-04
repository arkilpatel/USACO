"""
Streaming evaluation module that judges solutions as soon as generation completes.
Provides real-time logging to a file for monitoring progress.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Callable
import pickle
import os
import csv
from datetime import datetime
from functools import partial
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from USACOBench.utils import get_code_from_solution
from USACOBench.data_utils import load_problem_dict
from USACOBench.evaluation.judges.usaco_batch_judge import usaco_judge
from USACOBench.evaluation.result_type import ResultType

Problem = Dict[Any, Any]
Solution = Dict[str, Union[str, None]]
SolutionSet = List[Solution]
SolutionDict = Dict[str, SolutionSet]
Result = Dict[str, str]
ResultSet = List[Result]
ResultDict = Dict[str, ResultSet]
Query = Dict[str, str]

# Ensure logs directory exists
LOGS_DIR = Path(__file__).parent / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class StreamingLogger:
    """Thread-safe logger that writes to a file and CSV in real-time."""

    def __init__(self, log_dir: Path, model_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = log_dir / "evaluation.log"
        self.csv_path = log_dir / "results.csv"
        self.lock = threading.Lock()

        # Write log header
        with open(self.log_path, 'w') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"USACO Evaluation Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"=" * 80 + "\n\n")

        # Write CSV header
        self.csv_columns = [
            'problem_idx', 'problem_id', 'result_type', 'num_passed', 'num_tests',
            'fraction_passed', 'percentage_passed', 'status', 'timestamp', 'prompt', 'response', 'extracted_code'
        ]
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_columns)
            writer.writeheader()

    def log(self, message: str):
        """Thread-safe logging to file."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        with self.lock:
            with open(self.log_path, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
                f.flush()

    def log_problem_result(self, problem_id: str, prompt: str, response: str,
                           extracted_code: str, result: Dict, problem_idx: int, total: int):
        """Log a complete problem evaluation result to both log file and CSV."""
        result_type = result.get('result_type', ResultType.UNKNOWN)
        result_type_str = result_type.name if hasattr(result_type, 'name') else str(result_type)
        status = result.get('status', 'Unknown')
        num_passed = result.get('num_passed', 0)
        num_tests = result.get('num_tests', 0)
        fraction_passed = result.get('fraction_passed', 0)

        # Calculate percentage
        if num_tests > 0:
            percentage_passed = (num_passed / num_tests) * 100
        else:
            percentage_passed = 0.0

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with self.lock:
            # Write to log file
            with open(self.log_path, 'a') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"PROBLEM {problem_idx + 1}/{total}: {problem_id}\n")
                f.write(f"Time: {timestamp}\n")
                f.write("=" * 80 + "\n\n")

                # Result summary at top for quick scanning
                f.write(f">>> RESULT: {result_type_str} ({num_passed}/{num_tests} tests passed, {percentage_passed:.1f}%)\n")
                f.write(f">>> Status: {status}\n\n")

                # Problem prompt (full content)
                f.write("-" * 40 + " PROMPT " + "-" * 40 + "\n")
                f.write(prompt + "\n\n")

                # Model response (full content)
                f.write("-" * 40 + " MODEL RESPONSE " + "-" * 40 + "\n")
                f.write(response + "\n\n")

                # Extracted code
                f.write("-" * 40 + " EXTRACTED CODE " + "-" * 40 + "\n")
                if extracted_code:
                    f.write(extracted_code + "\n\n")
                else:
                    f.write("(No code extracted)\n\n")

                # Detailed result
                f.write("-" * 40 + " JUDGE RESULT " + "-" * 40 + "\n")
                f.write(f"Result Type: {result_type_str}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Tests Passed: {num_passed}/{num_tests}\n")

                # Show per-test results if available
                if 'result_list' in result and result['result_list']:
                    f.write("\nPer-test results:\n")
                    for i, test_result in enumerate(result['result_list'][:10]):  # Show first 10
                        test_type = test_result.get('result_type', ResultType.UNKNOWN)
                        test_type_str = test_type.name if hasattr(test_type, 'name') else str(test_type)
                        test_status = test_result.get('status', '')[:100]  # Truncate long status
                        f.write(f"  Test {i+1}: {test_type_str} - {test_status}\n")
                    if len(result['result_list']) > 10:
                        f.write(f"  ... and {len(result['result_list']) - 10} more tests\n")

                f.write("\n")
                f.flush()

            # Write to CSV
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writerow({
                    'problem_idx': problem_idx + 1,
                    'problem_id': problem_id,
                    'result_type': result_type_str,
                    'num_passed': num_passed,
                    'num_tests': num_tests,
                    'fraction_passed': fraction_passed,
                    'percentage_passed': f"{percentage_passed:.2f}",
                    'status': status,
                    'timestamp': timestamp,
                    'prompt': prompt,
                    'response': response,
                    'extracted_code': extracted_code or '',
                })

    def log_summary(self, results: List[Result], total_time: float, model_name: str):
        """Log final summary statistics to log file and summary CSV."""
        result_types = [r.get('result_type', ResultType.UNKNOWN) for r in results]
        type_counts = Counter(result_types)

        # Calculate per-problem percentages and macro average
        percentages = []
        for r in results:
            num_passed = r.get('num_passed', 0)
            num_tests = r.get('num_tests', 0)
            if num_tests > 0:
                pct = (num_passed / num_tests) * 100
                percentages.append(pct)
            else:
                percentages.append(0.0)

        macro_avg = sum(percentages) / len(percentages) if percentages else 0.0

        # Calculate additional metrics
        num_perfect = sum(1 for pct in percentages if pct == 100.0)
        num_partial = sum(1 for pct in percentages if 0 < pct < 100.0)
        num_zero = sum(1 for pct in percentages if pct == 0.0)

        with self.lock:
            # Write to log file
            with open(self.log_path, 'a') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write("FINAL SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total problems: {len(results)}\n")
                f.write(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)\n\n")

                f.write("=" * 40 + "\n")
                f.write("SCORING METRICS\n")
                f.write("=" * 40 + "\n")
                f.write(f"Macro Average (per-problem %): {macro_avg:.2f}%\n")
                f.write(f"Perfect scores (100%): {num_perfect}/{len(results)} ({100*num_perfect/len(results):.1f}%)\n")
                f.write(f"Partial scores (0-100%): {num_partial}/{len(results)} ({100*num_partial/len(results):.1f}%)\n")
                f.write(f"Zero scores (0%): {num_zero}/{len(results)} ({100*num_zero/len(results):.1f}%)\n\n")

                f.write("Results breakdown by type:\n")
                for result_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    type_str = result_type.name if hasattr(result_type, 'name') else str(result_type)
                    pct = 100 * count / len(results) if results else 0
                    f.write(f"  {type_str}: {count} ({pct:.1f}%)\n")
                f.write("\n")
                f.flush()

            # Write summary CSV
            summary_path = self.log_dir / "summary.csv"
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])
                writer.writerow(['model_name', model_name])
                writer.writerow(['total_problems', len(results)])
                writer.writerow(['total_time_seconds', f"{total_time:.2f}"])
                writer.writerow(['total_time_minutes', f"{total_time/60:.2f}"])
                writer.writerow(['timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow([])  # Empty row separator
                writer.writerow(['scoring_metric', 'value', 'description'])
                writer.writerow(['macro_average_percentage', f"{macro_avg:.2f}", 'Average of per-problem test pass percentages'])
                writer.writerow(['perfect_scores', num_perfect, 'Problems with 100% tests passed'])
                writer.writerow(['partial_scores', num_partial, 'Problems with 0-100% tests passed'])
                writer.writerow(['zero_scores', num_zero, 'Problems with 0% tests passed'])
                writer.writerow([])  # Empty row separator
                writer.writerow(['result_type', 'count', 'percentage'])
                for result_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    type_str = result_type.name if hasattr(result_type, 'name') else str(result_type)
                    pct = 100 * count / len(results) if results else 0
                    writer.writerow([type_str, count, f"{pct:.1f}"])


def evaluate_model_streaming(
    model_fn: Callable,
    prompt_fn: Callable,
    queries: List[Query],
    problem_dict: Dict[str, Problem],
    attempts: int = 1,
    problem_ids: List[str] = None,
    n_judge_workers: int = 8,
    model_name: str = "model",
    verbose: bool = True,
    log_dir: str = None,
) -> Tuple[ResultDict, SolutionDict, List[ResultSet], List[SolutionSet]]:
    """
    Streaming evaluation that judges solutions as soon as generation completes.

    Args:
        model_fn: Function that takes list of prompts and returns list of responses
        prompt_fn: Function that creates prompt from query
        queries: List of query dicts with problem_id and problem_description
        problem_dict: Dict mapping problem_id to problem metadata
        attempts: Number of attempts per problem
        problem_ids: Optional list to filter which problems to evaluate
        n_judge_workers: Number of parallel judging threads
        model_name: Name for logging purposes
        verbose: Whether to print progress
        log_dir: Directory for logging (optional, will create if not provided)

    Returns:
        Tuple of (rdict, sdict, rs, ss) same as evaluate_model
    """
    # Setup logging - use provided directory or create a new one
    if log_dir is not None:
        log_dir = Path(log_dir)
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        log_dir = LOGS_DIR / f"{safe_model_name}_{timestamp_str}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = StreamingLogger(log_dir, model_name)

    if verbose:
        print(f"Logging to: {log_dir}/")
        print(f"  - evaluation.log (human-readable)")
        print(f"  - results.csv (structured data)")

    # Filter queries if needed
    if problem_ids is not None:
        problem_ids_set = set(problem_ids)
        queries = [q for q in queries if q['problem_id'] in problem_ids_set]

    if verbose:
        print(f"Evaluating {len(queries)} problems with {attempts} attempt(s) each...")

    logger.log(f"Starting evaluation: {len(queries)} problems, {attempts} attempts each")
    logger.log(f"Model: {model_name}")
    logger.log(f"Judge workers: {n_judge_workers}")

    start_time = time.time()

    # Prepare prompts
    prompts = [prompt_fn(query) for query in queries]

    # Results storage
    all_results = []
    all_responses = []
    all_prompts = []
    all_queries = []

    # For attempts > 1, we repeat queries
    expanded_queries = queries * attempts
    expanded_prompts = prompts * attempts

    # Queue for passing generated responses to judge workers
    judge_queue = queue.Queue()
    results_dict = {}  # idx -> result
    results_lock = threading.Lock()

    def judge_worker():
        """Worker thread that judges solutions from the queue."""
        while True:
            item = judge_queue.get()
            if item is None:  # Poison pill
                judge_queue.task_done()
                break

            idx, query, prompt, response = item
            problem_id = query['problem_id']

            try:
                # Extract code from response
                extracted_code = get_code_from_solution(response)

                # Judge the solution
                problem = problem_dict[problem_id]
                result = usaco_judge(
                    problem=problem,
                    solution_code=extracted_code,
                    language='Python3',
                    mode='eval_all'
                )
                result['problem_id'] = problem_id

                # Log the result
                logger.log_problem_result(
                    problem_id=problem_id,
                    prompt=prompt,
                    response=response,
                    extracted_code=extracted_code,
                    result=result,
                    problem_idx=idx,
                    total=len(expanded_queries)
                )

                # Store result
                with results_lock:
                    results_dict[idx] = {
                        'result': result,
                        'response': response,
                        'prompt': prompt,
                        'extracted_code': extracted_code,
                        'query': query,
                    }

                # Print progress
                result_type = result.get('result_type', ResultType.UNKNOWN)
                result_type_str = result_type.name if hasattr(result_type, 'name') else str(result_type)
                if verbose:
                    print(f"[{idx+1}/{len(expanded_queries)}] {problem_id}: {result_type_str}")

            except Exception as e:
                logger.log(f"ERROR judging {problem_id}: {str(e)}")
                if verbose:
                    print(f"[{idx+1}/{len(expanded_queries)}] {problem_id}: ERROR - {str(e)}")

                with results_lock:
                    results_dict[idx] = {
                        'result': {
                            'result_type': ResultType.UNKNOWN,
                            'status': f'Error during judging: {str(e)}',
                            'problem_id': problem_id,
                            'num_passed': 0,
                            'num_tests': 10,
                            'fraction_passed': 0,
                            'result_list': None,
                        },
                        'response': response,
                        'prompt': prompt,
                        'extracted_code': get_code_from_solution(response) if response else None,
                        'query': query,
                    }

            judge_queue.task_done()

    # Start judge worker threads
    judge_threads = []
    for _ in range(n_judge_workers):
        t = threading.Thread(target=judge_worker, daemon=True)
        t.start()
        judge_threads.append(t)

    # Generate responses ONE AT A TIME and immediately queue for judging
    if verbose:
        print("Starting generation + judging (one problem at a time)...")
    logger.log("Starting generation + judging (streaming mode)...")

    from tqdm import tqdm

    for idx, (query, prompt) in enumerate(tqdm(zip(expanded_queries, expanded_prompts),
                                                total=len(expanded_queries),
                                                desc="Generating",
                                                disable=not verbose)):
        # Generate single response
        responses = model_fn([prompt], verbose=False)
        response = responses[0] if responses else ""

        # Immediately queue for judging
        judge_queue.put((idx, query, prompt, response))

        # Log generation progress
        if (idx + 1) % 10 == 0:
            logger.log(f"Generated {idx + 1}/{len(expanded_queries)} responses")

    logger.log(f"All {len(expanded_queries)} generations complete, waiting for judging to finish...")

    # Wait for all judging to complete
    judge_queue.join()

    # Send poison pills to stop workers
    for _ in judge_threads:
        judge_queue.put(None)
    for t in judge_threads:
        t.join()

    # Collect results in order
    ordered_results = [results_dict[i] for i in range(len(expanded_queries))]

    total_time = time.time() - start_time

    # Log summary
    all_judge_results = [r['result'] for r in ordered_results]
    logger.log_summary(all_judge_results, total_time, model_name)

    # Calculate and display macro average
    percentages = []
    for r in all_judge_results:
        num_passed = r.get('num_passed', 0)
        num_tests = r.get('num_tests', 0)
        if num_tests > 0:
            percentages.append((num_passed / num_tests) * 100)
        else:
            percentages.append(0.0)
    macro_avg = sum(percentages) / len(percentages) if percentages else 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"Macro Average Score: {macro_avg:.2f}% (per-problem test pass rate)")
        print(f"Results saved to: {log_dir}/")
        print(f"{'='*60}")

    # Format results same as original evaluate_model
    rdict = {}
    for item in ordered_results:
        problem_id = item['query']['problem_id']
        if problem_id not in rdict:
            rdict[problem_id] = []
        rdict[problem_id].append(item['result'])
    rs = list(rdict.values())

    sdict = {}
    for item in ordered_results:
        problem_id = item['query']['problem_id']
        if problem_id not in sdict:
            sdict[problem_id] = []
        sdict[problem_id].append({
            'solution': item['response'],
            'solution_code': item['extracted_code'],
            'result': item['result'],
            'problem_id': problem_id,
            'prompt': item['prompt'],
        })
    ss = list(sdict.values())

    return rdict, sdict, rs, ss


def evaluate_model_streaming_parallel(
    model: str,
    prompt_fn: Callable,
    queries: List[Query],
    problem_dict: Dict[str, Problem],
    attempts: int = 1,
    problem_ids: List[str] = None,
    n_judge_workers: int = 8,
    n_llm_workers: int = 8,
    model_name: str = "model",
    temperature: float = 0.6,
    max_tokens: int = 28000,
    verbose: bool = True,
    log_dir: str = None,
) -> Tuple[ResultDict, SolutionDict, List[ResultSet], List[SolutionSet]]:
    """
    Streaming evaluation with PARALLEL generation using Together AI.
    Sends multiple requests concurrently and judges solutions as they complete.

    Args:
        model: Model name for Together AI
        prompt_fn: Function that creates prompt from query
        queries: List of query dicts with problem_id and problem_description
        problem_dict: Dict mapping problem_id to problem metadata
        attempts: Number of attempts per problem
        problem_ids: Optional list to filter which problems to evaluate
        n_judge_workers: Number of parallel judging threads
        n_llm_workers: Number of parallel LLM generation workers
        model_name: Name for logging purposes
        temperature: Temperature for generation
        max_tokens: Max tokens for generation
        verbose: Whether to print progress
        log_dir: Directory for logging (optional, will create if not provided)

    Returns:
        Tuple of (rdict, sdict, rs, ss) same as evaluate_model
    """
    from models import together_streaming

    # Setup logging - use provided directory or create a new one
    if log_dir is not None:
        log_dir = Path(log_dir)
    else:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        log_dir = LOGS_DIR / f"{safe_model_name}_{timestamp_str}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = StreamingLogger(log_dir, model_name)

    if verbose:
        print(f"Logging to: {log_dir}/")
        print(f"  - evaluation.log (human-readable)")
        print(f"  - results.csv (structured data)")

    # Filter queries if needed
    if problem_ids is not None:
        problem_ids_set = set(problem_ids)
        queries = [q for q in queries if q['problem_id'] in problem_ids_set]

    if verbose:
        print(f"Evaluating {len(queries)} problems with {attempts} attempt(s) each...")
        print(f"Using {n_llm_workers} parallel LLM workers")

    logger.log(f"Starting evaluation: {len(queries)} problems, {attempts} attempts each")
    logger.log(f"Model: {model_name}")
    logger.log(f"LLM workers: {n_llm_workers}")
    logger.log(f"Judge workers: {n_judge_workers}")

    start_time = time.time()

    # Prepare prompts
    prompts = [prompt_fn(query) for query in queries]

    # For attempts > 1, we repeat queries
    expanded_queries = queries * attempts
    expanded_prompts = prompts * attempts

    # Queue for passing generated responses to judge workers
    judge_queue = queue.Queue()
    results_dict = {}  # idx -> result
    results_lock = threading.Lock()
    generated_count = [0]  # Use list to allow modification in nested function

    def judge_worker():
        """Worker thread that judges solutions from the queue."""
        while True:
            item = judge_queue.get()
            if item is None:  # Poison pill
                judge_queue.task_done()
                break

            idx, query, prompt, response = item
            problem_id = query['problem_id']

            try:
                # Extract code from response
                extracted_code = get_code_from_solution(response)

                # Judge the solution
                problem = problem_dict[problem_id]
                result = usaco_judge(
                    problem=problem,
                    solution_code=extracted_code,
                    language='Python3',
                    mode='eval_all'
                )
                result['problem_id'] = problem_id

                # Log the result
                logger.log_problem_result(
                    problem_id=problem_id,
                    prompt=prompt,
                    response=response,
                    extracted_code=extracted_code,
                    result=result,
                    problem_idx=idx,
                    total=len(expanded_queries)
                )

                # Store result
                with results_lock:
                    results_dict[idx] = {
                        'result': result,
                        'response': response,
                        'prompt': prompt,
                        'extracted_code': extracted_code,
                        'query': query,
                    }

                # Print progress
                result_type = result.get('result_type', ResultType.UNKNOWN)
                result_type_str = result_type.name if hasattr(result_type, 'name') else str(result_type)
                if verbose:
                    print(f"[Judged {idx+1}/{len(expanded_queries)}] {problem_id}: {result_type_str}")

            except Exception as e:
                logger.log(f"ERROR judging {problem_id}: {str(e)}")
                if verbose:
                    print(f"[Judged {idx+1}/{len(expanded_queries)}] {problem_id}: ERROR - {str(e)}")

                with results_lock:
                    results_dict[idx] = {
                        'result': {
                            'result_type': ResultType.UNKNOWN,
                            'status': f'Error during judging: {str(e)}',
                            'problem_id': problem_id,
                            'num_passed': 0,
                            'num_tests': 10,
                            'fraction_passed': 0,
                            'result_list': None,
                        },
                        'response': response,
                        'prompt': prompt,
                        'extracted_code': get_code_from_solution(response) if response else None,
                        'query': query,
                    }

            judge_queue.task_done()

    # Start judge worker threads
    judge_threads = []
    for _ in range(n_judge_workers):
        t = threading.Thread(target=judge_worker, daemon=True)
        t.start()
        judge_threads.append(t)

    # Callback function called when each generation completes
    def on_generation_complete(idx: int, prompt: str, response: str):
        """Called when a single generation completes - queues for judging immediately."""
        query = expanded_queries[idx]
        judge_queue.put((idx, query, prompt, response))

        generated_count[0] += 1
        if generated_count[0] % 10 == 0:
            logger.log(f"Generated {generated_count[0]}/{len(expanded_queries)} responses")

    # Generate responses IN PARALLEL and queue for judging as they complete
    if verbose:
        print("Starting parallel generation + judging...")
    logger.log("Starting parallel generation + judging...")

    # Use together_streaming for parallel generation with callbacks
    together_streaming(
        prompts=expanded_prompts,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_workers=n_llm_workers,
        on_result=on_generation_complete,
        verbose=verbose,
    )

    logger.log(f"All {len(expanded_queries)} generations complete, waiting for judging to finish...")

    # Wait for all judging to complete
    judge_queue.join()

    # Send poison pills to stop workers
    for _ in judge_threads:
        judge_queue.put(None)
    for t in judge_threads:
        t.join()

    # Collect results in order
    ordered_results = [results_dict[i] for i in range(len(expanded_queries))]

    total_time = time.time() - start_time

    # Log summary
    all_judge_results = [r['result'] for r in ordered_results]
    logger.log_summary(all_judge_results, total_time, model_name)

    # Calculate and display macro average
    percentages = []
    for r in all_judge_results:
        num_passed = r.get('num_passed', 0)
        num_tests = r.get('num_tests', 0)
        if num_tests > 0:
            percentages.append((num_passed / num_tests) * 100)
        else:
            percentages.append(0.0)
    macro_avg = sum(percentages) / len(percentages) if percentages else 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"Macro Average Score: {macro_avg:.2f}% (per-problem test pass rate)")
        print(f"Results saved to: {log_dir}/")
        print(f"{'='*60}")

    # Format results same as original evaluate_model
    rdict = {}
    for item in ordered_results:
        problem_id = item['query']['problem_id']
        if problem_id not in rdict:
            rdict[problem_id] = []
        rdict[problem_id].append(item['result'])
    rs = list(rdict.values())

    sdict = {}
    for item in ordered_results:
        problem_id = item['query']['problem_id']
        if problem_id not in sdict:
            sdict[problem_id] = []
        sdict[problem_id].append({
            'solution': item['response'],
            'solution_code': item['extracted_code'],
            'result': item['result'],
            'problem_id': problem_id,
            'prompt': item['prompt'],
        })
    ss = list(sdict.values())

    return rdict, sdict, rs, ss


def run_solve_streaming(
    model_fn,
    model_name,
    problem_dict,
    attempts,
    n_judge_workers=8,
    n_llm_workers=1,
    backend=None,
    temperature=0.6,
    max_tokens=28000,
    log_dir=None,
):
    """
    Streaming version of run_solve that judges as soon as generation completes.

    Args:
        model_fn: Model function for non-parallel backends
        model_name: Name of the model
        problem_dict: Dictionary of problems
        attempts: Number of attempts per problem
        n_judge_workers: Number of parallel judge workers
        n_llm_workers: Number of parallel LLM workers (only for 'together' backend)
        backend: Backend type ('together' enables parallel generation)
        temperature: Temperature for generation (only for parallel mode)
        max_tokens: Max tokens for generation (only for parallel mode)
        log_dir: Directory for logging (optional, will create if not provided)
    """
    from USACOBench.prompts import solve_prompt_fn
    from utils import save_json

    queries = []
    for problem_id in problem_dict.keys():
        queries.append({
            'problem_id': problem_id,
            'problem_description': problem_dict[problem_id]['description']
        })

    # Use parallel streaming for Together backend with multiple workers
    if backend == 'together' and n_llm_workers > 1:
        print(f"Using parallel streaming mode with {n_llm_workers} LLM workers")
        rdict, sdict, rs, ss = evaluate_model_streaming_parallel(
            model=model_name,
            prompt_fn=solve_prompt_fn,
            queries=queries,
            problem_dict=problem_dict,
            attempts=int(attempts),
            problem_ids=list(problem_dict.keys()),
            n_judge_workers=n_judge_workers,
            n_llm_workers=n_llm_workers,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=True,
            log_dir=log_dir,
        )
    else:
        # Use sequential streaming for other backends
        rdict, sdict, rs, ss = evaluate_model_streaming(
            model_fn=model_fn,
            prompt_fn=solve_prompt_fn,
            queries=queries,
            problem_dict=problem_dict,
            attempts=int(attempts),
            problem_ids=list(problem_dict.keys()),
            n_judge_workers=n_judge_workers,
            model_name=model_name,
            verbose=True,
            log_dir=log_dir,
        )

    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    save_json([rdict, sdict, rs, ss], f'results/results_{safe_model_name}_solve_{attempts}attempts')
    return rdict, sdict, rs, ss
