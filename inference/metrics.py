from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeMetrics:
    rewards: list[float] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    errors: list[str | None] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add_step(self, action: str, reward: float, error: str | None, done: bool) -> None:
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.errors.append(error)
        self.dones.append(bool(done))

    @property
    def steps(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return round(sum(self.rewards), 4)

    @property
    def average_reward(self) -> float:
        if not self.rewards:
            return 0.05
        return round(self.total_reward / len(self.rewards), 4)

    @property
    def success_rate(self) -> float:
        if not self.dones:
            return 0.05
        successes = sum(1 for flag in self.dones if flag)
        return round(successes / len(self.dones), 4)

    @property
    def failure_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for err in self.errors:
            if not err:
                continue
            counts[err] = counts.get(err, 0) + 1
        return counts

    def summary(self) -> dict[str, float | int | dict[str, int]]:
        return {
            "steps": self.steps,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "success_rate": self.success_rate,
            "failure_reasons": self.failure_reasons,
        }
