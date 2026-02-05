from typing import Union

def get_code_from_solution(solution: str) -> Union[str, None]:
    '''
    Assume code is in a Markdown block delimited by ```python and ```.
    Returns string of just the code from the *last* python block, or None if not found.
    '''
    try:
        begin_delim = "```python"
        end_delim = "```"
        # Find the last occurrence of ```python
        begin_idx = solution.rfind(begin_delim)
        if begin_idx == -1:
            raise ValueError("No ```python block found")
        end_idx = solution.index(end_delim, begin_idx + len(begin_delim))
        return solution[begin_idx + len(begin_delim) : end_idx]
    except Exception as e:
        print('Could not parse code from generated solution — returning entire solution')
        print(e)
        return solution
