from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator


class ActiveRequestRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, set[asyncio.Task]] = defaultdict(set)

    @contextmanager
    def track(self, model: str) -> Iterator[asyncio.Task | None]:
        task = asyncio.current_task()
        if task is None:
            yield None
            return

        self._tasks[model].add(task)
        try:
            yield task
        finally:
            tasks = self._tasks.get(model)
            if tasks is not None:
                tasks.discard(task)
                if not tasks:
                    self._tasks.pop(model, None)

    def cancel_others(self, model: str, current_task: asyncio.Task | None) -> None:
        for task in tuple(self._tasks.get(model, set())):
            if task is not current_task and not task.done():
                task.cancel()
