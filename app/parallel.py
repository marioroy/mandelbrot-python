# -*- coding: utf-8 -*-
"""
Provides auto-selection of various classes for multi-platform support.
"""

__all__ = ['USE_FORK', 'Barrier', 'Queue', 'Thread']

import os, sys

# By default use threading on macOS and Windows. Use fork otherwise.
# On UNIX platforms, one may set an environment variable to override
# USE_FORK=0 or USE_FORK=1.

if sys.platform == 'win32':
    USE_FORK = 0
else:
    val = os.getenv('USE_FORK')
    if val is None or val == 'auto':
        USE_FORK = 0 if sys.platform == 'darwin' else 1
    else:
        USE_FORK = int(val)

if USE_FORK:
    import multiprocessing
    ctx = multiprocessing.get_context('fork')
    Barrier = ctx.Barrier
    Queue = ctx.SimpleQueue
    Thread = ctx.Process
else:
    import threading 
    import queue 
    Barrier = threading.Barrier
    # Prefer SimpleQueue for lesser overhead. Python 3.7 onwards.
    Queue = queue.SimpleQueue if hasattr(queue, 'SimpleQueue') else queue.Queue
    Thread = threading.Thread

if __name__ == '__main__':
    print("use_fork: {}".format(USE_FORK))

