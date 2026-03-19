#!/usr/bin/env python3
"""
Generate expert model trajectories with top-k logprobs for USACO problems.

For each USACO problem, this script:
1. Generates a full solution trajectory using an expert model
2. Captures the top-k (default 20) logprobs at each generated token
3. Saves trajectories as Parquet (with msgpack-encoded logprobs)
4. Also emits backward-compatible JSONL

Supports two backends:
  - vLLM:     Local batched generation with logprobs
  - Together:  Together AI API via OpenAI-compatible SDK (async, high parallelism)

Example usage (Together AI):
    python scripts/get_expert_trajectories.py \
        --model moonshotai/Kimi-K2.5 \
        --model-short-name kimi-k2.5 \
        --backend together \
        --num-workers 32 \
        --temperature 1.0 --top-p 0.95 --max-new-tokens 32000 \
        --grade

Example usage (vLLM):
    python scripts/get_expert_trajectories.py \
        --model Qwen/Qwen3-Next-80B-A3B-Thinking \
        --model-short-name qwen3next \
        --backend vllm \
        --batch-size 32 \
        --max-new-tokens 32000 \
        --grade
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed as futures_as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Add USACO root to path so USACOBench imports work regardless of cwd
_USACO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_USACO_ROOT))

from USACOBench.data_utils import load_problem_dict
from USACOBench.prompts import solve_prompt_fn
from USACOBench.utils import get_code_from_solution


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate expert trajectories with top-k logprobs for USACO.")
    p.add_argument("--model", required=True, help="HF model ID / path (vLLM) or Together model name.")
    p.add_argument("--model-short-name", required=True, help="Short name for output naming (e.g. 'kimi-k2.5').")
    p.add_argument("--backend", choices=("vllm", "together"), required=True)
    p.add_argument("--top-k", type=int, default=20, help="Number of logprobs per token (default: 20).")
    p.add_argument("--max-new-tokens", type=int, default=32000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--sampling-top-k", type=int, default=20, help="Top-k sampling (vocab filter). -1 to disable.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for vLLM.")
    p.add_argument("--concurrency", "--num-workers", type=int, default=64, dest="concurrency",
                   help="Max concurrent requests for Together API (default: 64).")
    p.add_argument("--out-dir", default="expert_outputs", help="Parent output directory.")
    p.add_argument("--grade", action="store_true",
                   help="Execute generated code against test cases and store results.")
    p.add_argument("--grade-workers", type=int, default=8,
                   help="Number of parallel workers for code execution grading (default: 8).")
    p.add_argument("--force-finish", action="store_true",
                   help="Skip generation; finalize from existing progress.jsonl (grade + save parquet/jsonl/summary).")
    p.add_argument("--debug", type=int, default=None, metavar="N",
                   help="Only run first N problems (for quick testing).")
    p.add_argument("--timeout", type=int, default=8000,
                   help="HTTP timeout in seconds for API requests (default: 8000).")
    p.add_argument("--together-api-key", default=None,
                   help="Together API key (or set TOGETHER_API_KEY env var).")
    args = p.parse_args(argv)
    if args.backend == "together" and not args.together_api_key:
        args.together_api_key = os.environ.get("TOGETHER_API_KEY")
        if not args.together_api_key:
            p.error("Together backend requires --together-api-key or TOGETHER_API_KEY env var.")
    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        # NOTE: do NOT call torch.cuda.manual_seed_all here — doing so before
        # vLLM forks its worker processes initialises CUDA in the parent
        # process, which causes "Cannot re-initialize CUDA in forked
        # subprocess". vLLM manages GPU seeding itself via SamplingParams.seed.
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------
def load_task_data(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load USACO problems. Returns (problem_list, problem_dict_by_usaco_id)."""
    problem_dict = load_problem_dict("usaco_subset307")  # dict: str -> problem

    items: List[Dict[str, Any]] = []
    for i, (pid, prob) in enumerate(problem_dict.items()):
        query = {"problem_description": prob["description"]}
        prompt_text = solve_prompt_fn(query)
        messages = [{"role": "user", "content": prompt_text}]
        items.append({
            "problem_id": i,
            "usaco_problem_id": pid,
            "problem_text": prob["description"],
            "messages": messages,
            "domain": prob.get("problem_level", "unknown"),  # bronze/silver/gold/platinum
            "answer": "",  # no single string answer for USACO
            "task": "usaco",
            "extra": {
                "runtime_limit": prob.get("runtime_limit", 2.0),
                "memory_limit": prob.get("memory_limit", 256),
                "num_tests": prob.get("num_tests", 10),
            },
        })

    print(f"Loaded {len(items)} USACO problems")
    return items, problem_dict


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------
def init_vllm(args: argparse.Namespace):
    """Initialize vLLM engine once. Returns LLM instance."""
    import torch
    from vllm import LLM

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model_lower = args.model.lower()

    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.model,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "tensor_parallel_size": max(gpu_count, 1),
        "max_model_len": args.max_new_tokens + 2048,
    }
    if "nvidia" in model_lower:
        llm_kwargs["mamba_ssm_cache_dtype"] = "float32"
    if "minimax" in model_lower:
        llm_kwargs["enable_expert_parallel"] = True
    if "ministral" in model_lower:
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["config_format"] = "mistral"
        llm_kwargs["load_format"] = "mistral"

    print(f"Initializing vLLM with {gpu_count} GPU(s)...")
    return LLM(**llm_kwargs)


def generate_with_vllm(problems: List[Dict], args: argparse.Namespace, out_dir: Path, llm=None) -> List[Dict]:
    from vllm import SamplingParams

    model_lower = args.model.lower()
    enable_thinking = "nemotron" in model_lower or "qwen3" in model_lower

    results: List[Dict] = []
    total = len(problems)
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, total, batch_size), desc="vLLM batches"):
        batch = problems[batch_start: batch_start + batch_size]
        batch_messages = [p["messages"] for p in batch]
        sampling_params = [
            SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.sampling_top_k,
                seed=args.seed + p["problem_id"],
                logprobs=args.top_k,
            )
            for p in batch
        ]

        chat_kwargs = {}
        if enable_thinking:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}

        outputs = llm.chat(batch_messages, sampling_params=sampling_params, **chat_kwargs)

        for p, request_output in zip(batch, outputs):
            completion = request_output.outputs[0]
            text = completion.text.strip()
            token_ids = list(completion.token_ids)
            finish_reason = completion.finish_reason or ""

            sampled_tokens: List[Tuple[int, str, float]] = []
            per_position_logprobs: List[List[Tuple[int, str, float]]] = []
            if completion.logprobs:
                for pos_idx, position_lps in enumerate(completion.logprobs):
                    top_k_entries = sorted(
                        [(tid, lp.decoded_token, lp.logprob) for tid, lp in position_lps.items()],
                        key=lambda x: x[2],
                        reverse=True,
                    )
                    per_position_logprobs.append(top_k_entries)
                    sampled_tid = token_ids[pos_idx] if pos_idx < len(token_ids) else None
                    if sampled_tid is not None and sampled_tid in position_lps:
                        lp = position_lps[sampled_tid]
                        sampled_tokens.append((sampled_tid, lp.decoded_token, lp.logprob))
                    else:
                        sampled_tokens.append(tuple(top_k_entries[0]))

            result = {
                **p,
                "solution": text,
                "num_tokens": len(token_ids),
                "finish_reason": finish_reason,
                "token_ids": token_ids,
                "sampled_tokens": sampled_tokens,
                "logprobs": per_position_logprobs,
                "model": args.model,
                # grading fields (populated later)
                "extracted_code": None,
                "fraction_passed": None,
                "result_type": None,
                "correct": None,
            }
            save_result_incremental(result, out_dir)
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Together AI backend (async via openai SDK)
# ---------------------------------------------------------------------------
async def _together_generate_one(
    client,
    problem: Dict,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> Dict:
    """Generate a single completion with retries."""
    async with semaphore:
        last_exc = None
        for attempt in range(max_retries):
            try:
                create_kwargs = {
                    "model": args.model,
                    "messages": problem["messages"],
                    "max_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed + problem["problem_id"],
                    "logprobs": True,
                    "top_logprobs": args.top_k,
                }
                if args.sampling_top_k > 0:
                    create_kwargs["extra_body"] = {"top_k": args.sampling_top_k}
                response = await client.chat.completions.create(**create_kwargs)
                choice = response.choices[0]
                msg = choice.message
                content = msg.content or ""
                finish_reason = choice.finish_reason or ""

                # Thinking models return reasoning in a separate field
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
                if reasoning:
                    text = f"<think>\n{reasoning}\n</think>\n{content}"
                else:
                    text = content

                sampled_tokens: List[Tuple[str, float]] = []
                per_position_logprobs: List[List[Tuple[str, float]]] = []
                if choice.logprobs and choice.logprobs.content:
                    for token_info in choice.logprobs.content:
                        sampled_tokens.append((token_info.token, token_info.logprob))
                        top_k_entries: List[Tuple[str, float]] = []
                        if token_info.top_logprobs:
                            for tlp in token_info.top_logprobs:
                                top_k_entries.append((tlp.token, tlp.logprob))
                        per_position_logprobs.append(top_k_entries)

                if not text.strip() and sampled_tokens:
                    text = "".join(tok for tok, _ in sampled_tokens)

                return {
                    **problem,
                    "solution": text.strip(),
                    "num_tokens": len(per_position_logprobs),
                    "finish_reason": finish_reason,
                    "token_ids": [],  # not available via OpenAI-compatible API
                    "sampled_tokens": sampled_tokens,
                    "logprobs": per_position_logprobs,
                    "model": args.model,
                    # grading fields (populated later)
                    "extracted_code": None,
                    "fraction_passed": None,
                    "result_type": None,
                    "correct": None,
                }

            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                retryable = any(k in err_str for k in ("rate", "429", "500", "502", "503", "timeout", "timed out"))
                if retryable and attempt < max_retries - 1:
                    wait = min(2 ** attempt * 1.0, 60.0) + random.random()
                    tqdm.write(f"  [RETRY] problem {problem['problem_id']} in {wait:.1f}s "
                               f"(attempt {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(wait)
                elif not retryable:
                    tqdm.write(f"  [ERROR] problem {problem['problem_id']}: {e}")
                    raise

        tqdm.write(f"  [FAILED] problem {problem['problem_id']} after {max_retries} retries: {last_exc}")
        raise RuntimeError(f"Failed after {max_retries} retries for problem {problem['problem_id']}: {last_exc}")


async def _together_generate_all(problems: List[Dict], args: argparse.Namespace, out_dir: Path) -> List[Dict]:
    import httpx
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=args.together_api_key,
        base_url="https://api.together.xyz/v1",
        timeout=httpx.Timeout(args.timeout, connect=30.0),
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [_together_generate_one(client, p, args, semaphore) for p in problems]

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Together API"):
        result = await coro
        save_result_incremental(result, out_dir)
        results.append(result)

    results.sort(key=lambda r: r["problem_id"])
    return results


def generate_with_together(problems: List[Dict], args: argparse.Namespace, out_dir: Path) -> List[Dict]:
    return asyncio.run(_together_generate_all(problems, args, out_dir))


# ---------------------------------------------------------------------------
# Grading: execute code against USACO test cases
# ---------------------------------------------------------------------------
def grade_results(results: List[Dict], problem_dict: Dict[str, Any], n_workers: int = 8) -> List[Dict]:
    """
    Grade each result by executing extracted Python code against USACO test cases.
    Populates: extracted_code, fraction_passed, result_type, correct.
    """
    from USACOBench.evaluation.judges.usaco_batch_judge import usaco_judge
    from USACOBench.evaluation.judges.sandbox_config import initialize_sandbox

    initialize_sandbox()

    def grade_one(r: Dict) -> Dict:
        solution = r.get("solution", "")
        code = get_code_from_solution(solution)
        pid = r["usaco_problem_id"]
        problem = problem_dict[pid]
        try:
            result = usaco_judge(problem, code, mode="eval_all")
            r["extracted_code"] = code
            r["fraction_passed"] = result.get("fraction_passed", 0.0)
            r["result_type"] = result["result_type"].name
            r["correct"] = r["result_type"] == "ACCEPTED"
        except Exception as e:
            tqdm.write(f"  [GRADE ERROR] problem {r['problem_id']} ({pid}): {e}")
            r["extracted_code"] = code
            r["fraction_passed"] = 0.0
            r["result_type"] = "UNKNOWN"
            r["correct"] = False
        return r

    print(f"Grading {len(results)} solutions ({n_workers} workers)...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futs = {executor.submit(grade_one, r): r for r in results}
        for fut in tqdm(futures_as_completed(futs), total=len(futs), desc="Grading"):
            fut.result()  # raises on exception; grade_one catches internally

    n_accepted = sum(1 for r in results if r.get("correct"))
    avg_frac = sum(r.get("fraction_passed", 0.0) for r in results) / len(results) if results else 0.0
    print(f"Grading complete: {n_accepted}/{len(results)} fully accepted, "
          f"avg fraction passed = {avg_frac:.4f}")
    return results


# ---------------------------------------------------------------------------
# Incremental progress: save/load/resume
# ---------------------------------------------------------------------------
_progress_lock = threading.Lock()


def _progress_path(out_dir: Path) -> Path:
    return out_dir / "progress.jsonl"


def _serialize_result(r: Dict) -> str:
    """Serialize a single result to a JSON line (binary fields as base64)."""
    row = {
        "problem_id": r["problem_id"],
        "usaco_problem_id": r["usaco_problem_id"],
        "problem_text": r["problem_text"],
        "solution": r["solution"],
        "domain": r.get("domain", ""),
        "num_tokens": r["num_tokens"],
        "finish_reason": r["finish_reason"],
        "token_ids": base64.b64encode(msgpack.packb(r.get("token_ids", []))).decode(),
        "sampled_tokens": base64.b64encode(msgpack.packb(r.get("sampled_tokens", []))).decode(),
        "logprobs": base64.b64encode(msgpack.packb(r["logprobs"])).decode(),
        "model": r["model"],
        "messages": r["messages"],
        "extra": r.get("extra", {}),
        # grading fields
        "extracted_code": r.get("extracted_code"),
        "fraction_passed": r.get("fraction_passed"),
        "result_type": r.get("result_type"),
        "correct": r.get("correct"),
    }
    return json.dumps(row)


def _deserialize_result(line: str) -> Dict:
    """Deserialize a JSON line back to a result dict."""
    row = json.loads(line)
    row["token_ids"] = msgpack.unpackb(base64.b64decode(row["token_ids"]))
    row["sampled_tokens"] = msgpack.unpackb(base64.b64decode(row["sampled_tokens"]))
    row["logprobs"] = msgpack.unpackb(base64.b64decode(row["logprobs"]))
    # Ensure grading fields present (for backward compat with pre-grading progress files)
    row.setdefault("extracted_code", None)
    row.setdefault("fraction_passed", None)
    row.setdefault("result_type", None)
    row.setdefault("correct", None)
    # usaco_problem_id may be missing in very early progress files
    row.setdefault("usaco_problem_id", "")
    row.setdefault("task", "usaco")
    row.setdefault("answer", "")
    return row


def load_completed_ids(out_dir: Path) -> Tuple[Set[int], List[Dict]]:
    """Load previously completed results. Returns (completed_ids_set, results_list)."""
    ppath = _progress_path(out_dir)
    if not ppath.exists():
        return set(), []
    completed_ids: Set[int] = set()
    results: List[Dict] = []
    with ppath.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = _deserialize_result(line)
                completed_ids.add(r["problem_id"])
                results.append(r)
            except Exception as e:
                tqdm.write(f"  [WARN] Skipping corrupt progress line {line_no}: {e}")
    return completed_ids, results


def save_result_incremental(result: Dict, out_dir: Path) -> None:
    """Append a single result to the progress file (thread-safe)."""
    ppath = _progress_path(out_dir)
    line = _serialize_result(result) + "\n"
    with _progress_lock:
        with ppath.open("a", encoding="utf-8") as f:
            f.write(line)


# ---------------------------------------------------------------------------
# Serialization: parquet + compat JSONL
# ---------------------------------------------------------------------------
def save_trajectories_parquet(results: List[Dict], output_path: Path) -> None:
    """Save trajectories to Parquet with msgpack-encoded binary columns."""
    rows = []
    for r in results:
        rows.append({
            "problem_id": r["problem_id"],
            "usaco_problem_id": r["usaco_problem_id"],
            "problem": r["problem_text"],
            "solution": r["solution"],
            "domain": r.get("domain", ""),
            "num_tokens": r["num_tokens"],
            "finish_reason": r["finish_reason"],
            "token_ids": msgpack.packb(r.get("token_ids", [])),
            "sampled_tokens": msgpack.packb(r.get("sampled_tokens", [])),
            "logprobs": msgpack.packb(r["logprobs"]),
            "model": r["model"],
            "prompt_messages": json.dumps(r["messages"]),
            # grading columns (None if not graded)
            "extracted_code": r.get("extracted_code"),
            "fraction_passed": r.get("fraction_passed"),
            "result_type": r.get("result_type"),
            "correct": r.get("correct"),
        })

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(output_path), compression="snappy")
    print(f"Saved Parquet: {output_path} ({len(rows)} rows, {output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def save_compat_jsonl(results: List[Dict], output_path: Path) -> None:
    """Save backward-compatible JSONL (Problem + Solution) for score_proxy_metric.py."""
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            line = json.dumps({"Problem": r["problem_text"], "Solution": r["solution"]})
            f.write(line + "\n")
    print(f"Saved JSONL:   {output_path} ({len(results)} problems)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    print(f"\n{'=' * 60}")
    print(f"Task: usaco  |  Model: {args.model}  |  Backend: {args.backend}")
    print(f"{'=' * 60}\n")

    out_dir = Path(args.out_dir) / args.model_short_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load problems
    problems, problem_dict = load_task_data(args)
    if args.debug:
        problems = problems[:args.debug]
        print(f"[DEBUG] Truncated to {len(problems)} problems")

    # Resume: check for previously completed problems
    completed_ids, prev_results = load_completed_ids(out_dir)
    if args.force_finish:
        if not prev_results:
            print(f"[FORCE-FINISH] No progress.jsonl found in {out_dir}, nothing to finalize.")
            return
        print(f"[FORCE-FINISH] Using {len(prev_results)} completed results (skipping generation).")
        all_results = prev_results
        problems = []  # no generation needed
    elif completed_ids:
        remaining = [p for p in problems if p["problem_id"] not in completed_ids]
        print(f"[RESUME] Found {len(completed_ids)} completed, {len(remaining)} remaining")
        if not remaining:
            print("[RESUME] All problems already completed, skipping generation.")
            all_results = prev_results
            problems = []
        else:
            problems = remaining
            all_results = prev_results
    else:
        all_results = []

    # Initialize vLLM once (expensive — weight loading)
    vllm_engine = None
    if args.backend == "vllm" and problems:
        vllm_engine = init_vllm(args)

    # Generate
    if problems:
        t0 = time.time()
        if args.backend == "vllm":
            new_results = generate_with_vllm(problems, args, out_dir, llm=vllm_engine)
        else:
            new_results = generate_with_together(problems, args, out_dir)
        elapsed = time.time() - t0
        print(f"Generation complete: {len(new_results)} problems in {elapsed:.1f}s")
        all_results.extend(new_results)

    # Cleanup vLLM
    if vllm_engine is not None:
        import torch
        del vllm_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Sort by problem_id for consistent output
    all_results.sort(key=lambda r: r["problem_id"])

    # Optional grading
    if args.grade and all_results:
        all_results = grade_results(all_results, problem_dict, n_workers=args.grade_workers)

    # Save final outputs
    save_trajectories_parquet(all_results, out_dir / "trajectories.parquet")
    save_compat_jsonl(all_results, out_dir / f"usaco_{args.model_short_name}.jsonl")

    # Results summary
    summary: Dict[str, Any] = {
        "model": args.model,
        "model_short_name": args.model_short_name,
        "task": "usaco",
        "backend": args.backend,
        "total": len(all_results),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "top_k_logprobs": args.top_k,
        "sampling_top_k": args.sampling_top_k,
    }
    if args.grade:
        n_accepted = sum(1 for r in all_results if r.get("correct"))
        avg_frac = sum(r.get("fraction_passed", 0.0) for r in all_results) / len(all_results) if all_results else 0.0
        summary["accepted"] = n_accepted
        summary["accuracy"] = n_accepted / len(all_results) if all_results else 0.0
        summary["avg_fraction_passed"] = avg_frac

    with (out_dir / "results_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal: {len(all_results)} problems saved to {out_dir}")
    if args.grade:
        print(f"Accepted: {summary['accepted']}/{summary['total']} "
              f"= {summary['accuracy']:.4f}")
        print(f"Avg fraction passed: {summary['avg_fraction_passed']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
