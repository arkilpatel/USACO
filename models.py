import os
import openai
import asyncio
from openai import AsyncOpenAI, OpenAI
from typing import List, Dict, Union, Any, Tuple, Callable, Iterator
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
import backoff

from dotenv import load_dotenv
load_dotenv()

completion_tokens = {"gpt-4": 0,
                     "gpt-4-1106-preview": 0,
                     "gpt-3.5-turbo": 0,
                     "gpt-3.5-turbo-16k": 0}
prompt_tokens = {"gpt-4": 0,
                 "gpt-4-1106-preview": 0,
                 "gpt-3.5-turbo": 0,
                 "gpt-3.5-turbo-16k": 0}

async def generate_from_openai_chat_completion(
    messages_list: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: Union[str, List[str]],
    requests_per_minute: int = 300,
    verbose=False,
    **kwargs,
) -> List[str]:
    client = AsyncOpenAI()
    async_responses = []
    for message in messages_list:
        task = asyncio.create_task(generate_answer(message, client, model))
        async_responses.append(task)
    responses = await tqdm_asyncio.gather(*async_responses, disable=not verbose)
    return responses

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def generate_answer(prompt, client, model):
    """
    Send a prompt to OpenAI API and get the answer.
    :param prompt: the prompt to send.
    :return: the answer.
    """
    response = await client.chat.completions.create(
        model=model,
        messages=prompt,
    )
    return response

async def generate_from_anthropic_chat_completion(
    messages_list: List[str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: Union[str, List[str]],
    requests_per_minute: int = 300,
    verbose=False,
    **kwargs,
) -> List[str]:
    client = anthropic.AsyncAnthropic()
    async_responses = []
    for message in messages_list:
        task = asyncio.create_task(generate_answer_anthropic(message, client, model, max_tokens))
        async_responses.append(task)
    responses = await tqdm_asyncio.gather(*async_responses, disable=not verbose)
    return responses

@backoff.on_exception(backoff.expo, anthropic.RateLimitError)
async def generate_answer_anthropic(message, client, model, max_tokens):
    """
    Send a prompt to OpenAI API and get the answer.
    :param prompt: the prompt to send.
    :return: the answer.
    """
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=message,
    )
    return response
    
def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=2000, n=1, stop=None, **kwargs) -> list:
    return gpts([prompt] * n, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)[0]

def gpts(prompts, model="gpt-4", temperature=0.7, max_tokens=2000, stop=None,
         system_prompt: str = None,
         **kwargs) -> list:
    '''
    system_prompt: string added as a special system message at the beginning of the conversation
    '''
    if system_prompt is not None:
        messages_list = [[{'role': 'system', 'content': system_prompt},
                          {"role": "user", "content": prompt}] for prompt in prompts]
    else:
        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    return chatgpts(messages_list, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)

def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=2000, n=1, stop=None, **kwargs) -> list:
    return chatgpts([messages] * n, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)[0]

def chatgpt_raw(messages, model="gpt-4", temperature=0.7, max_tokens=2000, n=1, stop=None, **kwargs) -> list:
    return chatgpts_raw([messages] * n, model=model, temperature=temperature, max_tokens=max_tokens, stop=stop, **kwargs)[0]

def chatgpts(messages_list, model="gpt-4", temperature=0.7, max_tokens=2000, stop=None, max_messages=400, **kwargs) -> list:
    texts = []
    for i in range(0, len(messages_list), max_messages):
        responses = asyncio.run(generate_from_openai_chat_completion(model=model, messages_list=messages_list[i: i + max_messages], temperature=temperature, max_tokens=max_tokens, top_p=1, stop=stop, **kwargs))
        texts.extend([x.choices[0].message.content for x in responses])
        # global completion_tokens, prompt_tokens
        # completion_tokens[model] += sum(x["usage"]["completion_tokens"] for x in responses if "usage" in x and "completion_tokens" in x["usage"])
        # prompt_tokens[model] += sum(x["usage"]["prompt_tokens"] for x in responses if "usage" in x and "prompt_tokens" in x["usage"])
    return texts

def chatgpts_raw(messages_list, model="gpt-4", temperature=0.7, max_tokens=2000, stop=None, max_messages=400, **kwargs) -> list:
    '''
    Returns raw response messages, not just the text content
    '''
    responses_all = []
    for i in range(0, len(messages_list), max_messages):
        responses = asyncio.run(generate_from_openai_chat_completion(model=model, messages_list=messages_list[i: i + max_messages], temperature=temperature, max_tokens=max_tokens, top_p=1, stop=stop, **kwargs))
        responses_all.extend([x["choices"][0]["message"] for x in responses])
        # global completion_tokens, prompt_tokens
        # completion_tokens[model] += sum(x["usage"]["completion_tokens"] for x in responses if "usage" in x and "completion_tokens" in x["usage"])
        # prompt_tokens[model] += sum(x["usage"]["prompt_tokens"] for x in responses if "usage" in x and "prompt_tokens" in x["usage"])
    return responses_all

def claude(prompts, model="claude-3-sonnet-20240229", temperature=0.7, max_tokens=3000, stop=None, max_messages=400, system_prompt=None, **kwargs) -> list:
    texts = []
    if system_prompt is not None:
        messages_list = [[{'role': 'system', 'content': system_prompt},
                          {"role": "user", "content": prompt}] for prompt in prompts]
    else:
        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
    for i in range(0, len(prompts), max_messages):
        responses = asyncio.run(generate_from_anthropic_chat_completion(model=model, messages_list=messages_list, temperature=temperature, max_tokens=max_tokens, top_p=1, stop=stop, **kwargs))
        texts.extend([x.content[0].text for x in responses])
        # global completion_tokens, prompt_tokens
        # completion_tokens[model] += sum(x["usage"]["completion_tokens"] for x in responses if "usage" in x and "completion_tokens" in x["usage"])
        # prompt_tokens[model] += sum(x["usage"]["prompt_tokens"] for x in responses if "usage" in x and "prompt_tokens" in x["usage"])
    return texts

def gpt_usage():
    global completion_tokens, prompt_tokens
    cost = completion_tokens["gpt-4"] / 1000 * 0.06 + prompt_tokens["gpt-4"] / 1000 * 0.03
    cost += completion_tokens["gpt-4-1106-preview"] / 1000 * 0.03 + prompt_tokens["gpt-4-1106-preview"] / 1000 * 0.01
    cost += completion_tokens["gpt-3.5-turbo"] / 1000 * 0.002 + prompt_tokens["gpt-3.5-turbo"] / 1000 * 0.0015
    cost += completion_tokens["gpt-3.5-turbo-16k"] / 1000 * 0.004 + prompt_tokens["gpt-3.5-turbo-16k"] / 1000 * 0.003
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


####################################################################################################
# vLLM Backend (Local and Remote)
####################################################################################################

import subprocess
import time
import signal
import atexit

# Global reference to vLLM server process for cleanup
_vllm_server_process = None


def _get_num_gpus() -> int:
    """Auto-detect the number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpu_count = len(result.stdout.strip().split('\n'))
            return gpu_count if gpu_count > 0 else 1
    except Exception:
        pass
    return 1


def start_vllm_server(
    model: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    tensor_parallel_size: int = None,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 32000,
    dtype: str = "auto",
    trust_remote_code: bool = True,
    wait_for_ready: bool = True,
    timeout: int = 900,
    log_dir: str = None,
    **kwargs,
) -> subprocess.Popen:
    """
    Start a local vLLM server as a subprocess.

    Args:
        model: HuggingFace model name (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        port: Port to serve on (default: 8000)
        host: Host to bind to (default: "0.0.0.0")
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: None, auto-detect)
        gpu_memory_utilization: Fraction of GPU memory to use (default: 0.95)
        max_model_len: Maximum sequence length (default: 32000)
        dtype: Data type for model weights (default: "auto")
        trust_remote_code: Whether to trust remote code from HuggingFace (default: True)
        wait_for_ready: Wait for server to be ready before returning (default: True)
        timeout: Timeout in seconds when waiting for server (default: 900, i.e. 15 minutes)
        **kwargs: Additional arguments passed to vllm serve command

    Returns:
        subprocess.Popen: The server process handle

    Example:
        >>> process = start_vllm_server("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", port=8000)
        >>> # ... use the server ...
        >>> stop_vllm_server(process)  # or it auto-stops on exit
    """
    global _vllm_server_process

    # Auto-detect number of GPUs if not specified
    if tensor_parallel_size is None:
        tensor_parallel_size = _get_num_gpus()
        print(f"Auto-detected {tensor_parallel_size} GPU(s)")

    # Build command - use python -m vllm.entrypoints.openai.api_server for better compatibility
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--host", host,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])

    # Add Ministral-specific flags for proper tokenization and reasoning
    if _is_ministral_model(model):
        print(f"Detected Ministral model, adding special vLLM flags...")
        cmd.extend([
            "--tokenizer_mode", "mistral",
            "--config_format", "mistral",
            "--load_format", "mistral",
            "--reasoning-parser", "mistral",
        ])

    # Add any additional kwargs as command line arguments
    for key, value in kwargs.items():
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        else:
            cmd.extend([arg_name, str(value)])

    print(f"Starting vLLM server with command:")
    print(f"  {' '.join(cmd)}")

    # Create logs directory if it doesn't exist
    from pathlib import Path
    from datetime import datetime

    if log_dir is not None:
        # Use provided log directory
        logs_dir = Path(log_dir)
    else:
        # Fall back to default logs directory
        logs_dir = Path(__file__).parent / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create log file for vLLM server output
    vllm_log_path = logs_dir / "vllm_server.log"
    vllm_log_file = open(vllm_log_path, 'w')

    print(f"vLLM server logs will be written to: {vllm_log_path}")

    # Start the server process with output redirected to log file
    process = subprocess.Popen(
        cmd,
        stdout=vllm_log_file,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,  # Create new process group for clean shutdown
    )

    _vllm_server_process = process

    # Store log file handle for cleanup
    process._vllm_log_file = vllm_log_file
    process._vllm_log_path = vllm_log_path

    # Register cleanup on exit
    atexit.register(lambda: stop_vllm_server(process))

    if wait_for_ready:
        _wait_for_vllm_server(port, host, timeout, process)

    return process


def _wait_for_vllm_server(port: int, host: str, timeout: int, process: subprocess.Popen):
    """Wait for vLLM server to be ready to accept requests."""
    import httpx

    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}/v1/models"
    start_time = time.time()

    print(f"Waiting for vLLM server to be ready (timeout: {timeout}s)...")

    while time.time() - start_time < timeout:
        # Check if process died
        if process.poll() is not None:
            # Read any output
            stdout, _ = process.communicate()
            raise RuntimeError(f"vLLM server process died unexpectedly. Output:\n{stdout}")

        try:
            response = httpx.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ vLLM server is ready on port {port}")
                return
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass

        time.sleep(2)

    raise TimeoutError(f"vLLM server did not become ready within {timeout} seconds")


def stop_vllm_server(process: subprocess.Popen = None):
    """
    Stop a vLLM server process.

    Args:
        process: The process to stop. If None, stops the global server process.
    """
    global _vllm_server_process

    if process is None:
        process = _vllm_server_process

    if process is None:
        return

    if process.poll() is None:  # Process is still running
        print("Stopping vLLM server...")
        try:
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
            print("✓ vLLM server stopped")
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop gracefully
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
            print("✓ vLLM server force killed")
        except Exception as e:
            print(f"Warning: Error stopping vLLM server: {e}")

    # Close the log file if it exists
    if hasattr(process, '_vllm_log_file') and process._vllm_log_file:
        try:
            process._vllm_log_file.close()
            print(f"✓ vLLM server logs saved to: {process._vllm_log_path}")
        except Exception:
            pass

    if process == _vllm_server_process:
        _vllm_server_process = None


MINISTRAL_SYSTEM_PROMPT = """# HOW YOU SHOULD THINK AND ANSWER

First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response."""


def _get_vllm_served_model(client: AsyncOpenAI, model_name: str) -> str:
    """
    Fetch the best available model from the vLLM server synchronously.
    Prioritizes 'finetuned' model if available, otherwise uses the first model.
    """
    import httpx
    try:
        # Use sync request to get model list
        response = httpx.get(f"{client.base_url}models", timeout=10)
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            if not models_data:
                print(f"WARNING: No models found on vLLM server, using provided model name: {model_name}")
                return model_name

            # Look for 'finetuned' model first
            for model in models_data:
                if model.get("id") == "finetuned":
                    print(f"✓ Using fine-tuned model: 'finetuned'")
                    return "finetuned"

            # Fall back to first available model
            first_model = models_data[0].get("id", model_name)
            print(f"✓ Using model from server: '{first_model}'")
            return first_model
    except Exception as e:
        print(f"WARNING: Could not fetch model list from server: {e}")

    return model_name


async def generate_from_vllm(
    messages_list: List[List[Dict[str, str]]],
    model: str,
    temperature: float = 0.6,
    max_tokens: int = 16000,
    base_url: str = "http://localhost:8000/v1",
    verbose: bool = False,
    **kwargs,
) -> List[Any]:
    """
    Generate responses from a vLLM server using OpenAI-compatible API.
    """
    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)

    # Get the actual served model name
    served_model = _get_vllm_served_model(client, model)

    async def generate_single(messages):
        try:
            response = await client.chat.completions.create(
                model=served_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    async_responses = [generate_single(messages) for messages in messages_list]
    responses = await tqdm_asyncio.gather(*async_responses, disable=not verbose)
    return responses


# Cache for Nemotron tokenizer to avoid reloading
_NEMOTRON_TOKENIZER_CACHE = {}


def _get_nemotron_tokenizer(model: str):
    """Get or create cached Nemotron tokenizer."""
    if model not in _NEMOTRON_TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        print(f"Loading Nemotron tokenizer for {model}...")
        _NEMOTRON_TOKENIZER_CACHE[model] = AutoTokenizer.from_pretrained(model)
    return _NEMOTRON_TOKENIZER_CACHE[model]


def _format_nemotron_prompt(prompt: str, model: str) -> str:
    """
    Format a prompt for Nemotron models using their chat template.
    Nemotron requires manual chat template application with enable_thinking=True.
    """
    tokenizer = _get_nemotron_tokenizer(model)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return formatted


def _is_nemotron_model(model: str) -> bool:
    """Check if the model is a Nemotron model requiring special handling."""
    return "nemotron" in model.lower()


def _is_ministral_model(model: str) -> bool:
    """Check if the model is a Ministral model requiring special vLLM flags."""
    return "ministral" in model.lower()


def vllm_local(
    prompts: List[str],
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    temperature: float = 0.6,
    max_tokens: int = 28000,
    port: int = 8000,
    base_url: str = "http://localhost",
    system_prompt: str = None,
    max_messages: int = 400,
    verbose: bool = False,
    **kwargs,
) -> List[str]:
    """
    Generate responses from a local vLLM server.

    Args:
        prompts: List of prompt strings
        model: Model name (used for logging, actual model determined by server)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        port: vLLM server port
        base_url: Base URL for vLLM server
        system_prompt: Optional system prompt (auto-set for Ministral models)
        max_messages: Batch size for requests
        verbose: Whether to show progress bar

    Returns:
        List of response strings
    """
    from openai import OpenAI

    full_base_url = f"{base_url}:{port}/v1"

    # Check if this is a Nemotron model requiring manual chat template
    use_nemotron_template = _is_nemotron_model(model)

    if use_nemotron_template:
        # For Nemotron: apply chat template manually and use completions API
        print("Using Nemotron chat template with enable_thinking=True")
        formatted_prompts = [_format_nemotron_prompt(p, model) for p in prompts]

        # Use SYNC client for single-item generation
        if len(prompts) == 1:
            client = OpenAI(api_key="EMPTY", base_url=full_base_url)
            try:
                # Use completions API (not chat) since we pre-formatted the prompt
                response = client.completions.create(
                    model=model,
                    prompt=formatted_prompts[0],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if response and response.choices:
                    return [response.choices[0].text]
                else:
                    return [""]
            except Exception as e:
                print(f"Error generating response: {e}")
                return [""]
        else:
            # Batch generation for Nemotron
            texts = []
            client = OpenAI(api_key="EMPTY", base_url=full_base_url)
            for formatted_prompt in formatted_prompts:
                try:
                    response = client.completions.create(
                        model=model,
                        prompt=formatted_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if response and response.choices:
                        texts.append(response.choices[0].text)
                    else:
                        texts.append("")
                except Exception as e:
                    print(f"Error generating response: {e}")
                    texts.append("")
            return texts

    # Non-Nemotron models: use standard chat completions API
    # Auto-detect Ministral models and use special system prompt
    if system_prompt is None and "ministral" in model.lower():
        print("Using Ministral system prompt.")
        system_prompt = MINISTRAL_SYSTEM_PROMPT

    if system_prompt is not None:
        messages_list = [
            [{'role': 'system', 'content': system_prompt}, {"role": "user", "content": prompt}]
            for prompt in prompts
        ]
    else:
        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

    # Use SYNC client for single-item generation, ASYNC for batches
    if len(prompts) == 1:
        # Synchronous path for single-item generation (streaming mode)
        client = OpenAI(api_key="EMPTY", base_url=full_base_url)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages_list[0],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response and response.choices:
                return [response.choices[0].message.content]
            else:
                return [""]
        except Exception as e:
            print(f"Error generating response: {e}")
            return [""]
    else:
        # Async path for batch generation (original behavior)
        texts = []
        for i in range(0, len(messages_list), max_messages):
            batch = messages_list[i:i + max_messages]
            responses = asyncio.run(
                generate_from_vllm(
                    messages_list=batch,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=full_base_url,
                    verbose=verbose,
                    **kwargs,
                )
            )
            for resp in responses:
                if resp is not None and resp.choices:
                    texts.append(resp.choices[0].message.content)
                else:
                    texts.append("")

        return texts


def vllm_remote(
    prompts: List[str],
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    temperature: float = 0.6,
    max_tokens: int = 28000,
    port: int = 8000,
    system_prompt: str = None,
    max_messages: int = 400,
    verbose: bool = False,
    **kwargs,
) -> List[str]:
    """
    Generate responses from a remote vLLM server via SSH tunnel.
    Assumes SSH tunnel is already set up (e.g., ssh -L 8000:localhost:8000 user@remote).

    Args:
        prompts: List of prompt strings
        model: Model name (used for logging, actual model determined by server)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        port: Local port forwarded to remote vLLM server
        system_prompt: Optional system prompt
        max_messages: Batch size for requests
        verbose: Whether to show progress bar

    Returns:
        List of response strings
    """
    return vllm_local(
        prompts=prompts,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        port=port,
        base_url="http://localhost",
        system_prompt=system_prompt,
        max_messages=max_messages,
        verbose=verbose,
        **kwargs,
    )


####################################################################################################
# Together AI Backend
####################################################################################################

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=10)
async def generate_answer_together(
    messages: List[Dict[str, str]],
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    """
    Send a prompt to Together AI API and get the answer with exponential backoff.
    """
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response


async def generate_from_together(
    messages_list: List[List[Dict[str, str]]],
    model: str,
    temperature: float = 0.6,
    max_tokens: int = 16000,
    verbose: bool = False,
    **kwargs,
) -> List[Any]:
    """
    Generate responses from Together AI API with parallel requests.
    """
    api_key = os.getenv("TOGNAME")
    if not api_key:
        raise ValueError("Missing Together AI API key. Set TOGNAME environment variable.")

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")

    async_responses = [
        asyncio.create_task(
            generate_answer_together(messages, client, model, temperature, max_tokens)
        )
        for messages in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses, disable=not verbose)
    return responses


def together(
    prompts: List[str],
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    temperature: float = 0.6,
    max_tokens: int = 28000,
    system_prompt: str = None,
    max_messages: int = 400,
    verbose: bool = False,
    **kwargs,
) -> List[str]:
    """
    Generate responses using Together AI API.

    Args:
        prompts: List of prompt strings
        model: Model name on Together AI (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt (auto-set for Ministral models)
        max_messages: Batch size for requests (Together handles parallelism)
        verbose: Whether to show progress bar

    Returns:
        List of response strings
    """
    # Auto-detect Ministral models and use special system prompt
    if system_prompt is None and "ministral" in model.lower():
        print("Using Ministral system prompt.")
        system_prompt = MINISTRAL_SYSTEM_PROMPT

    if system_prompt is not None:
        messages_list = [
            [{'role': 'system', 'content': system_prompt}, {"role": "user", "content": prompt}]
            for prompt in prompts
        ]
    else:
        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

    texts = []
    for i in range(0, len(messages_list), max_messages):
        batch = messages_list[i:i + max_messages]
        responses = asyncio.run(
            generate_from_together(
                messages_list=batch,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=verbose,
                **kwargs,
            )
        )
        for resp in responses:
            if resp is not None and resp.choices:
                texts.append(resp.choices[0].message.content)
            else:
                texts.append("")

    return texts


####################################################################################################
# Together AI Streaming Backend (Parallel generation with per-result callbacks)
####################################################################################################

def _generate_single_together(
    idx: int,
    prompt: str,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str = None,
) -> Tuple[int, str]:
    """
    Generate a single response from Together AI API (synchronous, for use in ThreadPoolExecutor).
    Returns (index, response_text) tuple to maintain ordering.
    """
    if system_prompt is not None:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ]
    else:
        messages = [{'role': 'user', 'content': prompt}]

    # Retry with exponential backoff on rate limit errors
    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if response and response.choices:
                return (idx, response.choices[0].message.content)
            return (idx, "")
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                import time
                wait_time = (2 ** attempt) + 1  # Exponential backoff
                time.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            # For other errors, return empty string with error info
            return (idx, f"[ERROR: {str(e)}]")


def together_streaming(
    prompts: List[str],
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    temperature: float = 0.6,
    max_tokens: int = 28000,
    system_prompt: str = None,
    num_workers: int = 8,
    on_result: Callable[[int, str, str], None] = None,
    verbose: bool = False,
) -> List[str]:
    """
    Generate responses using Together AI API with parallel workers.
    Results are yielded as they complete (not in order) via the on_result callback.

    Args:
        prompts: List of prompt strings
        model: Model name on Together AI
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt (auto-set for Ministral models)
        num_workers: Number of parallel worker threads
        on_result: Callback function(idx, prompt, response) called as each result completes
        verbose: Whether to show progress bar

    Returns:
        List of response strings (in original order)
    """
    api_key = os.getenv("TOGNAME")
    if not api_key:
        raise ValueError("Missing Together AI API key. Set TOGNAME environment variable.")

    client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")

    # Auto-detect Ministral models and use special system prompt
    if system_prompt is None and "ministral" in model.lower():
        print("Using Ministral system prompt.")
        system_prompt = MINISTRAL_SYSTEM_PROMPT

    # Results storage (indexed to maintain order)
    results = [""] * len(prompts)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all generation tasks
        future_to_idx = {
            executor.submit(
                _generate_single_together,
                idx, prompt, client, model, temperature, max_tokens, system_prompt
            ): idx
            for idx, prompt in enumerate(prompts)
        }

        # Process completed generations as they finish
        pbar = tqdm(total=len(prompts), desc="Generating", disable=not verbose)
        for future in as_completed(future_to_idx):
            idx, response = future.result()
            results[idx] = response

            # Call the callback with the result
            if on_result is not None:
                on_result(idx, prompts[idx], response)

            pbar.update(1)
        pbar.close()

    return results
