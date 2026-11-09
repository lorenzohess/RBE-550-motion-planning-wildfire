#!/usr/bin/env python3
"""Wildfire entry point. See `python main.py --help`."""

import os
import sys

# One BLAS thread: idle worker threads spin and inflate process_time(), the CPU-time metric.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from wildfire.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
