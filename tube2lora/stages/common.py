from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StageReport:
    stage: str
    total: int
    success: int
    failed: int
    skipped: int
    output_path: str | None = None

    @property
    def failure_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.failed / float(self.total)
