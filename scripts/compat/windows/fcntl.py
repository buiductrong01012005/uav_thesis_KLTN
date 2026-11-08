"""Windows byte-range lock adapter for the frozen runners' exclusive flock call.

Enable only on Windows by adding this directory to PYTHONPATH. Linux uses its
native fcntl. Closing the file releases the OS lock, including on process exit.
"""
import msvcrt
import os

LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(file, operation):
    descriptor = file if isinstance(file, int) else file.fileno()
    if operation not in (LOCK_EX, LOCK_EX | LOCK_NB, LOCK_UN):
        raise ValueError("Only exclusive runner locks are supported")
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        mode = (msvcrt.LK_UNLCK if operation == LOCK_UN else
                msvcrt.LK_NBLCK if operation & LOCK_NB else msvcrt.LK_LOCK)
        msvcrt.locking(descriptor, mode, 1)
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)
