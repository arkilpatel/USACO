'''
Contains all code to duplicate experiments in "Can language models solve olympiad programming questions?"
To utilize open models, create your own callable model function in models.py, and import it as with GPTs/Claude.

Supported backends:
  - gpt: OpenAI GPT models (gpt-4, gpt-3.5-turbo, etc.)
  - claude: Anthropic Claude models
  - vllm-local: Local vLLM server
  - vllm-remote: Remote vLLM server via SSH tunnel
  - together: Together AI API
'''

import argparse
import atexit
import os
import shutil
import tempfile
from functools import partial
from pathlib import Path

import torch
from rank_bm25 import BM25Okapi

from models import gpts, claude, vllm_local, vllm_remote, together, start_vllm_server, stop_vllm_server
from utils import load_json, save_json, generate_episodic_retrieval_queries, generate_semantic_retrieval_queries, generate_episodic_semantic_retrieval_queries
from USACOBench.prompts import solve_prompt_fn, retrieval_prompt_fn, reflexion_prompt_fn, RetrievalType
from USACOBench.data_utils import load_corpus, load_problem_dict, load_problems
from evaluate import evaluate_model
from USACOBench.evaluation import print_metrics
from dotenv import load_dotenv
from utils import run_solve, run_retrieval, run_reflexion, calculate_final_rs
from evaluate_streaming import run_solve_streaming
from collections import Counter

load_dotenv()


# ============================================================================
# LoRA Checkpoint Merging Functions (adapted from score_finetuned.py)
# ============================================================================

def select_dtype(device: torch.device) -> torch.dtype:
    """Select appropriate dtype based on device capabilities."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def dtype_to_vllm_string(dtype: torch.dtype) -> str:
    """Convert torch dtype to vLLM string format."""
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


def available_gpu_count() -> int:
    """Return number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def prepare_merged_model(
    base_model: str,
    checkpoint_path: Path,
    dtype: torch.dtype,
    gpu_count: int,
) -> Path:
    """
    Load base model, apply LoRA adapters, merge, and save to temp directory.

    Args:
        base_model: HuggingFace model ID or local path
        checkpoint_path: Path to LoRA adapter checkpoint
        dtype: Torch dtype for model loading
        gpu_count: Number of available GPUs

    Returns:
        Path to temporary directory containing merged model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Use /work/scratch/tmp for merged models (consistent with score_finetuned.py)
    base_tmp = Path(os.getenv("MERGE_TMPDIR", "/work/scratch/tmp"))
    base_tmp.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="merged-model-", dir=str(base_tmp)))
    atexit.register(shutil.rmtree, tmp_dir, True)  # clean up on exit

    print(f"Loading base model: {base_model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=(
            "auto"
            if gpu_count > 1
            else {"": f"cuda:{device.index or 0}"} if device.type == "cuda" else {"": "cpu"}
        ),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapters from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base, checkpoint_path)

    print("Merging adapters into base model...")
    merged_model = model.merge_and_unload()
    merged_model.to("cpu")

    print(f"Saving merged model to: {tmp_dir}")
    merged_model.save_pretrained(tmp_dir)

    # Also save tokenizer from checkpoint or base model
    tokenizer_source = checkpoint_path if (checkpoint_path / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.save_pretrained(tmp_dir)

    # Copy chat template if present
    chat_template_path = checkpoint_path / "chat_template.jinja"
    if chat_template_path.exists():
        shutil.copy2(chat_template_path, tmp_dir / "chat_template.jinja")

    # Clean up to free memory
    del merged_model
    del model
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Merged model ready at: {tmp_dir}")
    return tmp_dir


# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', help='model endpoint: ie. gpt-4-1106-preview, deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', default='gpt-3.5-turbo')
parser.add_argument('-b', '--backend', help='backend to use: gpt, claude, vllm-local, vllm-remote, together', default=None)
parser.add_argument('-e', '--episodic_retrieval', help='whether to use episodic retrieval', action="store_true", default=False)
parser.add_argument('-f', '--num_retrieved', help='number of documents retrieved', default=2)
parser.add_argument('-s', '--semantic_retrieval', help='whether to use semantic retrieval', action="store_true", default=False)
parser.add_argument('-r', '--reflexion', help='whether to use reflexion', action="store_true", default=False)
parser.add_argument('-a', '--attempts', help='number of attempts', default=1)
parser.add_argument('-n', '--num_reflexion', help='number of reflexion iterations', default=2)
parser.add_argument('--streaming', action='store_true', help='Use streaming evaluation (judges as generation completes, logs to logs/)')
parser.add_argument('--judge-workers', type=int, default=8, help='Number of parallel judge workers for streaming mode (default: 8)')
parser.add_argument('--llm-workers', type=int, default=32, help='Number of parallel LLM workers for streaming mode with Together backend (default: 1)')
# vLLM-specific arguments
parser.add_argument('--port', type=int, default=8000, help='Port for vLLM server (default: 8000)')
parser.add_argument('--base-url', type=str, default='http://localhost', help='Base URL for vLLM local server (default: http://localhost)')
parser.add_argument('--serve', action='store_true', help='Start a local vLLM server before running (for vllm-local backend)')
parser.add_argument('--tensor-parallel-size', type=int, default=None, help='Number of GPUs for tensor parallelism (default: auto-detect)')
parser.add_argument('--gpu-memory-utilization', type=float, default=0.95, help='Fraction of GPU memory to use (default: 0.95)')
parser.add_argument('--max-model-len', type=int, default=32000, help='Maximum sequence length for vLLM server (default: 32000)')
# Generation parameters
parser.add_argument('--batch-size', '-bs', type=int, default=1, help='Number of problems to send to the LLM engine concurrently (default: 1)')
parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for generation (default: 0.6)')
parser.add_argument('--max-tokens', type=int, default=28000, help='Max new tokens for generation (default: 28000)')
# Problem filtering
parser.add_argument('--expert-passed', action='store_true', help='Only evaluate on problems that expert model passed (loads from data/expert_passed_problems.json)')
# LoRA checkpoint support
parser.add_argument('--checkpoint', type=str, default=None,
    help='Checkpoint name (e.g., openr1_qwen3_1.7b_math_frac1_lr2em05_ep1_bs4_ga2). '
         'Full path: /work/scratch/forecast_generalization/runs/{checkpoint}/model')
args = parser.parse_args()

model_name = args.model_name

# Determine backend: explicit flag or infer from model name
backend = args.backend
if backend is None:
    # Infer backend from model name for backward compatibility
    if 'gpt' in model_name.lower():
        backend = 'gpt'
    elif 'claude' in model_name.lower():
        backend = 'claude'
    else:
        raise Exception(
            f"Cannot infer backend from model name '{model_name}'. "
            "Please specify --backend explicitly: gpt, claude, vllm-local, vllm-remote, or together"
        )

# Select model function based on backend
if backend == 'gpt':
    model_fn = gpts
elif backend == 'claude':
    model_fn = claude
elif backend == 'vllm-local':
    model_fn = partial(vllm_local, port=args.port, base_url=args.base_url,
                       temperature=args.temperature, max_tokens=args.max_tokens)
elif backend == 'vllm-remote':
    model_fn = partial(vllm_remote, port=args.port,
                       temperature=args.temperature, max_tokens=args.max_tokens)
elif backend == 'together':
    model_fn = partial(together, temperature=args.temperature, max_tokens=args.max_tokens)
else:
    raise Exception(f"Unknown backend: {backend}. Use: gpt, claude, vllm-local, vllm-remote, or together")

print(f"Using backend: {backend}")
print(f"Model: {model_name}")

# Create log directory for this run (used by both vLLM server and streaming evaluation)
from datetime import datetime
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_model_name = model_name.replace('/', '_').replace('\\', '_')
# Include checkpoint name in run_id if provided
if args.checkpoint:
    run_id = f"{safe_model_name}_{args.checkpoint}_{timestamp_str}"
else:
    run_id = f"{safe_model_name}_{timestamp_str}"
run_log_dir = Path(__file__).parent / 'logs' / run_id
run_log_dir.mkdir(parents=True, exist_ok=True)
print(f"Run log directory: {run_log_dir}")

# Initialize sandbox with run-specific ID (will be cleaned up at end)
from USACOBench.evaluation.judges.sandbox_config import initialize_sandbox, cleanup_sandbox
sandbox_run_id = initialize_sandbox(run_id)
print(f"Sandbox initialized with run ID: {sandbox_run_id}")

# Handle LoRA checkpoint merging if requested
model_path = model_name  # Default: use model_name directly
if args.checkpoint:
    if backend not in ('vllm-local', 'vllm-remote'):
        print(f"Warning: --checkpoint is only supported with vllm-local/vllm-remote backends, ignoring.")
    else:
        # Construct full checkpoint path from checkpoint name
        checkpoint_path = Path(f"/work/scratch/forecast_generalization/runs/{args.checkpoint}/model")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        print(f"Checkpoint: {args.checkpoint}")
        print(f"Checkpoint path: {checkpoint_path}")
        gpu_count = available_gpu_count()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = select_dtype(device)

        print(f"Merging LoRA adapters into base model...")
        merged_model_dir = prepare_merged_model(
            base_model=model_name,
            checkpoint_path=checkpoint_path,
            dtype=dtype,
            gpu_count=gpu_count,
        )
        model_path = str(merged_model_dir)
        print(f"Using merged model at: {model_path}")

# Start local vLLM server if requested
vllm_process = None
if args.serve:
    if backend != 'vllm-local':
        print("Warning: --serve flag is only used with vllm-local backend, ignoring.")
    else:
        vllm_process = start_vllm_server(
            model=model_path,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            log_dir=str(run_log_dir),
            max_num_seqs=args.batch_size,
        )

problem_dict = load_problem_dict('usaco_subset307')

# Filter to expert-passed problems if requested
if args.expert_passed:
    import json
    import os
    expert_data_path = os.path.join(os.path.dirname(__file__), 'data', 'expert_passed_problems.json')
    if not os.path.exists(expert_data_path):
        print(f"ERROR: Expert passed problems file not found: {expert_data_path}")
        print("Please run: python extract_expert_passed.py path/to/results.csv")
        exit(1)
    with open(expert_data_path, 'r') as f:
        expert_passed_ids = set(json.load(f).keys())
    original_count = len(problem_dict)
    problem_dict = {pid: prob for pid, prob in problem_dict.items() if pid in expert_passed_ids}
    print(f"Filtered to {len(problem_dict)}/{original_count} expert-passed problems")

# Use model_path (merged model if checkpoint provided, otherwise original model_name)
model_fn = partial(model_fn, model=model_path)

# A little redundant but it does the job and it's readable...
if not args.episodic_retrieval and not args.semantic_retrieval and not args.reflexion:
    if args.streaming:
        rdict, sdict, rs, ss = run_solve_streaming(
            model_fn, model_name, problem_dict, args.attempts,
            n_judge_workers=args.judge_workers,
            n_llm_workers=args.llm_workers,
            backend=backend,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            log_dir=str(run_log_dir),
            batch_size=args.batch_size,
        )
    else:
        rdict, sdict, rs, ss = run_solve(model_fn, model_name, problem_dict, args.attempts)

elif args.episodic_retrieval and not args.semantic_retrieval and not args.reflexion:
    rdict, sdict, rs, ss = run_solve(model_fn, model_name, problem_dict, args.attempts)
    rdict, sdict, rs, ss = run_retrieval(model_fn, model_name, problem_dict, args.attempts, ss, args.num_retrieved, RetrievalType.EPISODIC)

elif not args.episodic_retrieval and args.semantic_retrieval and not args.reflexion:
    rdict, sdict, rs, ss = run_solve(model_fn, model_name, problem_dict, args.attempts)
    rdict, sdict, rs, ss = run_retrieval(model_fn, model_name, problem_dict, args.attempts, ss, args.num_retrieved, RetrievalType.SEMANTIC)

elif args.episodic_retrieval and args.semantic_retrieval and not args.reflexion:
    rdict, sdict, rs, ss = run_solve(model_fn, model_name, problem_dict, args.attempts)
    rdict, sdict, rs, ss = run_retrieval(model_fn, model_name, problem_dict, args.attempts, ss, args.num_retrieved, RetrievalType.EPISODIC_SEMANTIC)

elif not args.episodic_retrieval and not args.semantic_retrieval and args.reflexion:
    rdict, sdict, rs, ss = run_solve(model_fn, model_name, problem_dict, args.attempts)
    reflexions = [rdict]
    query_dict = None
    for i in range(args.num_reflexion):
        rdict, sdict, rs, ss, query_dict = run_reflexion(model_fn, model_name, problem_dict, args.attempts, rdict, sdict, query_dict, i, return_queries=True)
        reflexions.append(rdict)

    rs = calculate_final_rs(reflexions, problem_dict)

elif args.episodic_retrieval and not args.semantic_retrieval and args.reflexion:
    rdict, sdict, rs, ss = run_solve(model_fn, model_name, problem_dict, args.attempts)
    rdict, sdict, rs, ss = run_retrieval(model_fn, model_name, problem_dict, args.attempts, ss, args.num_retrieved, RetrievalType.EPISODIC)

    reflexions = [rdict]
    query_dict = None
    for i in range(args.num_reflexion):
        rdict, sdict, rs, ss, query_dict = run_reflexion(model_fn, model_name, problem_dict, args.attempts, rdict, sdict, query_dict, i, return_queries=True, retrieval=True)
        reflexions.append(rdict)

    rs = calculate_final_rs(reflexions, problem_dict)
    
elif not args.episodic_retrieval and args.semantic_retrieval and args.reflexion:
    rdict, sdict, rs, ss = run_solve(model_fn, model_name, problem_dict, args.attempts)
    rdict, sdict, rs, ss = run_retrieval(model_fn, model_name, problem_dict, args.attempts, ss, args.num_retrieved, RetrievalType.SEMANTIC)

    reflexions = [rdict]
    query_dict = None
    for i in range(args.num_reflexion):
        rdict, sdict, rs, ss, query_dict = run_reflexion(model_fn, model_name, problem_dict, args.attempts, rdict, sdict, query_dict, i, return_queries=True, retrieval=True)
        reflexions.append(rdict)

    rs = calculate_final_rs(reflexions, problem_dict)

elif args.episodic_retrieval and args.semantic_retrieval and args.reflexion:
    rdict, sdict, rs, ss = run_solve(model_fn, model_name, problem_dict, args.attempts)
    rdict, sdict, rs, ss = run_retrieval(model_fn, model_name, problem_dict, args.attempts, ss, args.num_retrieved, RetrievalType.EPISODIC_SEMANTIC)

    reflexions = [rdict]
    query_dict = None
    for i in range(args.num_reflexion):
        rdict, sdict, rs, ss, query_dict = run_reflexion(model_fn, model_name, problem_dict, args.attempts, rdict, sdict, query_dict, i, return_queries=True, retrieval=True)
        reflexions.append(rdict)

    rs = calculate_final_rs(reflexions, problem_dict)

print_metrics(rs)
print('Result summary:')
result_types = [result['result_type'] for result_set in rs for result in result_set]
print(Counter(result_types))
print()

# Cleanup vLLM server if we started one
if vllm_process is not None:
    stop_vllm_server(vllm_process)

# Cleanup sandbox directory
cleanup_sandbox()



