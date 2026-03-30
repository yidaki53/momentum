"""Type stubs for momentum._timer_cy (Cython-compiled timer)."""

from typing import Callable, Optional

class TimerServiceCy:
    def __init__(
        self,
        total_seconds: int,
        on_tick: Optional[Callable[[], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
        on_interrupted: Optional[Callable[[int, int], None]] = None,
        tick_interval: float = 1.0,
    ) -> None: ...
    def run(self) -> None: ...
    def stop(self) -> None: ...
