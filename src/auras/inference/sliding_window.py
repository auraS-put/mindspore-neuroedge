from __future__ import annotations

from collections import deque
from typing import Deque, List


class WindowBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buf: Deque[float] = deque(maxlen=size)

    def push(self, x: float) -> None:
        self.buf.append(x)

    def ready(self) -> bool:
        return len(self.buf) == self.size

    def get(self) -> List[float]:
        return list(self.buf)
