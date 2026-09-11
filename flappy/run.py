"""Run the fixed Doomfly MaleCNS controller against Flappy Fly.

Pixels are the only game input to the neural controller. The game telemetry in
this module is audit/evaluation output only and never selects an action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np

from flappy.game import Game

ROOT = Path(__file__).resolve().parents[1]
GAME_HZ = 35


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MaleCNS v1.0 plays a one-button Flappy Bird environment")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=41027)
    p.add_argument("--decoder", choices=["biological", "bci"], default="bci")
    p.add_argument(
        "--condition",
        choices=["intact", "blank_vision", "controls_clamped"],
        default="intact",
        help="Prespecified control condition; telemetry never affects the action.",
    )
    p.add_argument(
        "--audit",
        type=Path,
        default=None,
        help="JSONL output path (default: outputs/flappy/run-<uuid>.jsonl).",
    )
    return p


def run(args: argparse.Namespace) -> dict:
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")

    # Importing NativeBrain verifies the reviewed C++ kernel binary against its
    # recorded source hash before simulation begins.
    from doom.engine import NeuralControls
    from doom.game import retinal_samples
    from doom.native import NativeBrain

    graph = ROOT / "outputs/doom/malecns_v1/graph.npz"
    manifest_path = ROOT / "outputs/doom/malecns_v1/manifest.json"
    if not graph.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "MaleCNS artifacts are missing. Build the existing Doomfly neural stack first "
            "(python -m doom.connectome malecns_v1; python -m doom.prepare; "
            "python -m doom.audit_data; python -m doom.build_kernel)."
        )

    manifest = json.loads(manifest_path.read_text())
    brain = NativeBrain(graph)
    controls = NeuralControls(manifest["readouts"], mode=args.decoder)
    game = Game(seed=args.seed)

    run_id = str(uuid.uuid4())
    audit_path = args.audit or ROOT / "outputs/flappy" / f"run-{run_id}.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    neural_tick = 0
    episode_results: list[dict] = []
    total_flaps = 0

    try:
        with audit_path.open("w", encoding="utf-8") as audit:
            while len(episode_results) < args.episodes:
                before = game.observation()
                if before["finished"]:
                    episode_results.append(before)
                    if len(episode_results) >= args.episodes:
                        break
                    game.new_episode()

                frame = game.pixels()
                light = retinal_samples(frame, brain.uv)
                if args.condition == "blank_vision":
                    light.fill(0)

                # Match Doomfly's 35 Hz game clock to the audited 0.1 ms neural
                # timestep. Rounding yields the same alternating 285/286-step
                # intervals used by doom.server.
                neural_tick += 1
                target_cursor = int(round(neural_tick * 10000 / GAME_HZ))
                steps = target_cursor - brain.cursor
                if steps < 1:
                    raise RuntimeError("Neural/game clocks lost monotonic alignment")

                counts, neural_wall_seconds = brain.step(light, steps * brain.dt)
                requested = controls.decode(counts, steps * brain.dt / 1000.0)
                applied = dict(requested)
                if args.condition == "controls_clamped":
                    applied["turn"] = 0.0
                    applied["forward"] = 0.0
                    applied["attack"] = False

                # Flappy Fly intentionally consumes only the pre-existing attack
                # Boolean. No score, position, pipe geometry, or other telemetry
                # can steer the controller.
                reward = game.act(applied)
                after = game.observation()
                total_flaps += int(bool(applied["attack"]))

                event = {
                    "run_id": run_id,
                    "condition": args.condition,
                    "decoder": args.decoder,
                    "seed": args.seed,
                    "episode": after["episode"],
                    "tick": after["tick"],
                    "neural_ms": round(float(brain.sim_ms), 3),
                    "neural_interval_ms": round(float(steps * brain.dt), 3),
                    "neural_wall_ms": round(float(neural_wall_seconds * 1000), 3),
                    "input_sha256": hashlib.sha256(light.tobytes()).hexdigest(),
                    "source_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "spike_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
                    "requested": {
                        "turn": float(requested["turn"]),
                        "forward": float(requested["forward"]),
                        "attack": bool(requested["attack"]),
                    },
                    "applied": {
                        "turn": float(applied["turn"]),
                        "forward": float(applied["forward"]),
                        "attack": bool(applied["attack"]),
                    },
                    "game": after,
                    "reward": float(reward),
                }
                audit.write(json.dumps(event, separators=(",", ":")) + "\n")

            summary = {
                "run_id": run_id,
                "condition": args.condition,
                "decoder": args.decoder,
                "seed": args.seed,
                "episodes": episode_results,
                "episode_count": len(episode_results),
                "mean_score": float(np.mean([e["score"] for e in episode_results])),
                "mean_survival_seconds": float(
                    np.mean([e["tick"] / GAME_HZ for e in episode_results])
                ),
                "max_score": int(max(e["score"] for e in episode_results)),
                "total_flaps": int(total_flaps),
                "audit": str(audit_path),
            }
            print(json.dumps(summary, indent=2))
            return summary
    finally:
        game.close()


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
