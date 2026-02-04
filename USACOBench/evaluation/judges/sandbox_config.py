"""
Centralized sandbox configuration for USACO evaluation.

This module manages sandbox directory paths with support for unique run IDs
and cleanup after runs complete.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

# Global state for sandbox configuration
_sandbox_state = {
    'run_id': None,
    'sandbox_dir': None,
    'predictions_dir': None,
    'solutions_dir': None,
    'initialized': False,
}


def get_repo_root() -> str:
    """Get the repository root directory."""
    return os.environ.get(
        'USACO_ROOT',
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    )


def initialize_sandbox(run_id: Optional[str] = None) -> str:
    """
    Initialize sandbox directories with a unique run ID.

    Args:
        run_id: Optional run identifier. If not provided, generates a UUID.

    Returns:
        The run_id used for this sandbox.
    """
    global _sandbox_state

    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    repo_root = get_repo_root()
    sandbox_dir = os.path.join(repo_root, f'usaco_sandbox_{run_id}')
    predictions_dir = os.path.join(sandbox_dir, 'predictions', 'usaco')
    solutions_dir = os.path.join(sandbox_dir, 'solutions', 'usaco')

    # Create directories
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    Path(solutions_dir).mkdir(parents=True, exist_ok=True)

    _sandbox_state['run_id'] = run_id
    _sandbox_state['sandbox_dir'] = sandbox_dir
    _sandbox_state['predictions_dir'] = predictions_dir
    _sandbox_state['solutions_dir'] = solutions_dir
    _sandbox_state['initialized'] = True

    return run_id


def get_sandbox_dir() -> str:
    """Get the sandbox directory, initializing if needed."""
    if not _sandbox_state['initialized']:
        initialize_sandbox()
    return _sandbox_state['sandbox_dir']


def get_predictions_dir() -> str:
    """Get the predictions directory, initializing if needed."""
    if not _sandbox_state['initialized']:
        initialize_sandbox()
    return _sandbox_state['predictions_dir']


def get_solutions_dir() -> str:
    """Get the solutions directory, initializing if needed."""
    if not _sandbox_state['initialized']:
        initialize_sandbox()
    return _sandbox_state['solutions_dir']


def get_predictions_path() -> str:
    """Get the predictions path template."""
    return os.path.join(get_predictions_dir(), '{}_{}.pred')


def get_solutions_path() -> str:
    """Get the solutions path template."""
    return os.path.join(get_solutions_dir(), '{}_{}.py')


def get_run_id() -> Optional[str]:
    """Get the current run ID, or None if not initialized."""
    return _sandbox_state['run_id']


def cleanup_sandbox() -> bool:
    """
    Remove the sandbox directory and reset state.

    Returns:
        True if cleanup was successful, False otherwise.
    """
    global _sandbox_state

    if not _sandbox_state['initialized'] or _sandbox_state['sandbox_dir'] is None:
        return False

    sandbox_dir = _sandbox_state['sandbox_dir']

    try:
        if os.path.exists(sandbox_dir):
            shutil.rmtree(sandbox_dir)
            print(f"Cleaned up sandbox directory: {sandbox_dir}")

        # Reset state
        _sandbox_state['run_id'] = None
        _sandbox_state['sandbox_dir'] = None
        _sandbox_state['predictions_dir'] = None
        _sandbox_state['solutions_dir'] = None
        _sandbox_state['initialized'] = False

        return True
    except Exception as e:
        print(f"Warning: Failed to cleanup sandbox directory {sandbox_dir}: {e}")
        return False


def is_initialized() -> bool:
    """Check if sandbox has been initialized."""
    return _sandbox_state['initialized']
