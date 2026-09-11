"""Minimal Flappy Bird boundary for the MaleCNS experiment.

The controller receives rendered RGB pixels only. Game-state telemetry is exposed
for audit/evaluation, but it must never be used to choose actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Config:
    width: int = 640
    height: int = 480
    frame_hz: int = 35
    bird_x: float = 160.0
    bird_radius: int = 12
    gravity: float = 0.55
    flap_velocity: float = -7.0
    terminal_velocity: float = 10.0
    pipe_width: int = 70
    pipe_gap: int = 150
    pipe_spacing: int = 260
    pipe_speed: float = 4.0
    first_pipe_x: float = 500.0
    margin: int = 70
    episode_seconds: int = 60


class Game:
    """Deterministic, pixel-only Flappy Bird environment.

    ``act`` accepts the existing Doomfly action dictionary and intentionally
    ignores ``turn`` and ``forward``. The existing Boolean ``attack`` output is
    the sole control signal: ``attack == True`` means flap.
    """

    def __init__(self, seed: int = 41027, config: Config | None = None):
        self.seed = int(seed)
        self.config = config or Config()
        if self.config.width < 64 or self.config.height < 64:
            raise ValueError("Frame dimensions are too small")
        if self.config.pipe_gap <= 2 * self.config.bird_radius:
            raise ValueError("Pipe gap must be wider than the bird")
        self.assets = {
            "environment": "flappy-fly-v0",
            "renderer": "numpy-rgb",
            "frame_hz": self.config.frame_hz,
        }
        self.episode = 0
        self.tick = 0
        self.episodes: list[dict] = []
        self.new_episode()

    @property
    def max_ticks(self) -> int:
        return self.config.frame_hz * self.config.episode_seconds

    def new_episode(self) -> None:
        self.episode += 1
        self.tick = 0
        self.finished = False
        self.score = 0
        self.bird_y = self.config.height / 2.0
        self.bird_v = 0.0
        # Each episode is repeatable for a given seed and episode number.
        self._rng = np.random.default_rng(self.seed + self.episode - 1)
        self._pipes: list[dict] = []
        x = self.config.first_pipe_x
        while x < self.config.width + self.config.pipe_spacing:
            self._pipes.append(self._new_pipe(x))
            x += self.config.pipe_spacing

    def _new_pipe(self, x: float) -> dict:
        half = self.config.pipe_gap / 2.0
        low = self.config.margin + half
        high = self.config.height - self.config.margin - half
        if low >= high:
            raise ValueError("Pipe gap/margin do not fit inside frame")
        return {
            "x": float(x),
            "gap_y": float(self._rng.uniform(low, high)),
            "passed": False,
        }

    def pixels(self) -> np.ndarray:
        if self.finished:
            raise RuntimeError("Episode finished; reset is required")
        c = self.config
        rgb = np.full((c.height, c.width, 3), 238, dtype=np.uint8)

        # High-contrast geometry keeps the visual stimulus simple and auditable.
        pipe_rgb = np.asarray([42, 72, 42], dtype=np.uint8)
        for pipe in self._pipes:
            left = max(0, int(round(pipe["x"])))
            right = min(c.width, int(round(pipe["x"] + c.pipe_width)))
            if left >= right:
                continue
            gap_top = max(0, int(round(pipe["gap_y"] - c.pipe_gap / 2)))
            gap_bottom = min(c.height, int(round(pipe["gap_y"] + c.pipe_gap / 2)))
            rgb[:gap_top, left:right] = pipe_rgb
            rgb[gap_bottom:, left:right] = pipe_rgb

        bx = int(round(c.bird_x))
        by = int(round(self.bird_y))
        r = c.bird_radius
        y0, y1 = max(0, by - r), min(c.height, by + r + 1)
        x0, x1 = max(0, bx - r), min(c.width, bx + r + 1)
        if y0 < y1 and x0 < x1:
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (xx - bx) ** 2 + (yy - by) ** 2 <= r**2
            patch = rgb[y0:y1, x0:x1]
            patch[mask] = np.asarray([25, 25, 25], dtype=np.uint8)
        return rgb

    def act(self, action: Mapping[str, object] | bool) -> float:
        if self.finished:
            raise RuntimeError("Episode finished; reset is required")
        if isinstance(action, Mapping):
            flap = bool(action.get("attack", False))
        else:
            flap = bool(action)

        c = self.config
        if flap:
            self.bird_v = c.flap_velocity
        self.bird_v = min(self.bird_v + c.gravity, c.terminal_velocity)
        self.bird_y += self.bird_v

        reward = 0.0
        for pipe in self._pipes:
            pipe["x"] -= c.pipe_speed
            if not pipe["passed"] and pipe["x"] + c.pipe_width < c.bird_x:
                pipe["passed"] = True
                self.score += 1
                reward += 1.0

        while self._pipes and self._pipes[0]["x"] + c.pipe_width < 0:
            self._pipes.pop(0)
        while not self._pipes or self._pipes[-1]["x"] < c.width + c.pipe_spacing:
            next_x = (
                c.width + c.pipe_spacing
                if not self._pipes
                else self._pipes[-1]["x"] + c.pipe_spacing
            )
            self._pipes.append(self._new_pipe(next_x))

        self.tick += 1
        collided = self._collision()
        if collided or self.tick >= self.max_ticks:
            self.finished = True
            if collided:
                reward -= 1.0
        return float(reward)

    def _collision(self) -> bool:
        c = self.config
        r = c.bird_radius
        if self.bird_y - r <= 0 or self.bird_y + r >= c.height:
            return True
        bird_left = c.bird_x - r
        bird_right = c.bird_x + r
        for pipe in self._pipes:
            pipe_left = pipe["x"]
            pipe_right = pipe["x"] + c.pipe_width
            if bird_right < pipe_left or bird_left > pipe_right:
                continue
            gap_top = pipe["gap_y"] - c.pipe_gap / 2
            gap_bottom = pipe["gap_y"] + c.pipe_gap / 2
            if self.bird_y - r < gap_top or self.bird_y + r > gap_bottom:
                return True
        return False

    def observation(self) -> dict:
        return {
            "episode": self.episode,
            "tick": self.tick,
            "finished": bool(self.finished),
            "score": int(self.score),
            "bird_y": round(float(self.bird_y), 3),
            "bird_v": round(float(self.bird_v), 3),
        }

    def close(self) -> None:
        pass
