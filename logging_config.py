"""
Centralized logging configuration for Graph RAG.

All entry points (main.py, mcp_server.py) call setup_logging() once at
startup. All other modules just do logging.getLogger(__name__) — they
never configure logging themselves.

Log format:
    12:34:56 [INFO] ingestion.pipeline: Document stored: abc-123
"""

from __future__ import annotations
import logging
import sys


_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(verbose: bool = False, stderr_only: bool = False) -> None:
    """
    Configure the root logger once.

    Args:
        verbose:     If True, set level to DEBUG. Otherwise INFO.
        stderr_only: If True, direct all output to stderr (used by MCP stdio
                     mode so stdout stays clean for the MCP protocol).
    """
    level = logging.DEBUG if verbose else logging.INFO
    stream = sys.stderr if stderr_only else sys.stdout

    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt=_DATEFMT,
        stream=stream,
        force=True,  # override any earlier basicConfig calls
    )
