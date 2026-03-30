# cython: language_level=3, boundscheck=False, cdivision=True

"""Cython-compiled timer event loop (tight timing loop)."""

import time as py_time
from typing import Callable, Optional


cdef class TimerServiceCy:
    """Cythonic timer with optimized event loop."""

    cdef double _start_time
    cdef double _total_seconds
    cdef bint _running
    cdef object _on_tick
    cdef object _on_complete
    cdef object _on_interrupted
    cdef double _tick_interval

    def __init__(
        self,
        total_seconds: int,
        on_tick: Optional[Callable[[], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        on_interrupted: Optional[Callable[[int, int], None]] = None,
        tick_interval: float = 1.0,
    ) -> None:
        self._total_seconds = <double>total_seconds
        self._on_tick = on_tick or (lambda: None)
        self._on_complete = on_complete or (lambda: None)
        self._on_interrupted = on_interrupted or (lambda e, t: None)
        self._tick_interval = tick_interval
        self._running = False
        self._start_time = 0.0

    cpdef void run(self):
        """Optimized event loop with minimal Python overhead."""
        cdef double elapsed
        cdef double last_tick = 0.0
        cdef double tick_threshold

        self._running = True
        self._start_time = py_time.time()

        while self._running:
            elapsed = py_time.time() - self._start_time

            # Check if time to tick
            if elapsed - last_tick >= self._tick_interval:
                self._on_tick()
                last_tick = elapsed

            # Check if complete
            if elapsed >= self._total_seconds:
                self._running = False
                self._on_complete()
                break

            # Minimal sleep to reduce CPU spin
            py_time.sleep(0.01)

    cpdef void stop(self):
        """Stop timer and report interruption."""
        if self._running:
            elapsed = <int>(py_time.time() - self._start_time)
            self._running = False
            self._on_interrupted(elapsed, <int>self._total_seconds)
