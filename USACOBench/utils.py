import re
from typing import Union


def _extract_last_fenced_block(solution: str, delimiter: str) -> Union[str, None]:
    """Extract content from the last occurrence of a fenced code block with the given delimiter."""
    end_delim = "```"
    begin_idx = solution.rfind(delimiter)
    if begin_idx == -1:
        return None
    after_delim = begin_idx + len(delimiter)
    end_idx = solution.find(end_delim, after_delim)
    if end_idx == -1:
        # No closing delimiter — take everything after the opening delimiter
        return solution[after_delim:]
    return solution[after_delim:end_idx]


def _extract_last_bare_code_block(solution: str) -> Union[str, None]:
    """Extract content from the last bare ``` block (no language tag)."""
    # Match ``` followed by a newline (no language specifier)
    # We find all ``` positions and pair them
    pattern = re.compile(r'```\s*\n')
    matches = list(pattern.finditer(solution))
    if len(matches) < 2:
        return None

    # Take the last pair of ``` delimiters
    # Walk backwards to find a pair: an opening ``` and a closing ```
    all_triple = [(m.start(), m.end()) for m in re.finditer(r'```', solution)]
    if len(all_triple) < 2:
        return None

    # The last closing ``` and the one before it as opening
    close_pos = all_triple[-1][0]
    open_match = None
    for start, end in reversed(all_triple[:-1]):
        # Check this is a bare ``` (not ```python, ```py, etc.)
        after = solution[end:end + 20].lstrip()
        if after.startswith('\n') or after.startswith('\r') or end == len(solution):
            open_match = (start, end)
            # Skip past the newline
            content_start = solution.find('\n', end)
            if content_start == -1:
                continue
            content_start += 1
            break

    if open_match is None:
        return None

    content = solution[content_start:close_pos]
    # Sanity check: does it look like code?
    if _looks_like_python(content):
        return content
    return None


def _looks_like_python(text: str) -> bool:
    """Heuristic check whether a text block looks like Python code."""
    text = text.strip()
    if not text:
        return False
    # Check for common Python patterns
    python_indicators = [
        r'^\s*(import |from \w+ import )',
        r'^\s*def \w+\s*\(',
        r'^\s*class \w+',
        r'^\s*(for |while |if |elif |else:)',
        r'^\s*print\s*\(',
        r'^\s*input\s*\(',
        r'^\s*sys\.stdin',
        r'^\s*\w+\s*=\s*',
    ]
    for pattern in python_indicators:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def _extract_code_after_think_block(solution: str) -> Union[str, None]:
    """Extract Python code that appears after a </think> block."""
    # Find the last </think> tag
    think_end = solution.rfind('</think>')
    if think_end == -1:
        return None

    after_think = solution[think_end + len('</think>'):].strip()
    if not after_think:
        return None

    # If there's a fenced block after </think>, extract it
    for delim in ['```python', '```Python', '```py', '```']:
        idx = after_think.find(delim)
        if idx != -1:
            content_start = idx + len(delim)
            end_idx = after_think.find('```', content_start)
            content = after_think[content_start:end_idx] if end_idx != -1 else after_think[content_start:]
            if content.strip():
                return content

    # No fenced block — check if the remaining text looks like raw Python code
    if _looks_like_python(after_think):
        return after_think
    return None


def _extract_raw_python_tail(solution: str) -> Union[str, None]:
    """
    Last resort: find the last contiguous block of Python-looking code at the end
    of the response, or the last block starting with an import/def/class statement.
    """
    lines = solution.split('\n')

    # Strategy: scan backwards from the end to find a contiguous code block
    # First, strip trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None

    # Find the last line that looks like the START of a Python program
    # (import statement, or reading input)
    start_patterns = re.compile(
        r'^\s*(import |from \w+ import |sys\.stdin|input\s*\(|#!/)'
    )

    best_start = None
    for i in range(len(lines) - 1, -1, -1):
        if start_patterns.match(lines[i]):
            best_start = i
            break

    if best_start is None:
        return None

    candidate = '\n'.join(lines[best_start:])
    if _looks_like_python(candidate):
        return candidate
    return None


def get_code_from_solution(solution: str) -> Union[str, None]:
    '''
    Extract Python code from a model response, trying multiple strategies:
    1. Last ```python block (most common)
    2. Last ```Python or ```py block (capitalization/abbreviation variants)
    3. Code after </think> block (thinking models)
    4. Last bare ``` block that looks like Python
    5. Last contiguous block of raw Python code (import-based detection)
    6. Fallback: return entire solution (will likely fail at judging)
    '''
    if not solution or not solution.strip():
        return solution

    # Strategy 1: ```python (standard, most common)
    code = _extract_last_fenced_block(solution, '```python')
    if code is not None and code.strip():
        return code

    # Strategy 2: ```Python or ```py variants
    for delim in ['```Python', '```py']:
        code = _extract_last_fenced_block(solution, delim)
        if code is not None and code.strip():
            return code

    # Strategy 3: code after </think> block
    code = _extract_code_after_think_block(solution)
    if code is not None and code.strip():
        return code

    # Strategy 4: bare ``` block containing Python-looking code
    code = _extract_last_bare_code_block(solution)
    if code is not None and code.strip():
        return code

    # Strategy 5: raw Python code at the tail of the response
    code = _extract_raw_python_tail(solution)
    if code is not None and code.strip():
        return code

    # Strategy 6: fallback — return entire solution
    print('Could not extract Python code from generated solution — returning entire solution')
    return solution
