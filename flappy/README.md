# Flappy Fly v0

A deliberately small one-button environment for testing the existing Doomfly
MaleCNS controller in a simpler closed loop.

## What is fixed

The neural side is unchanged: the same MaleCNS graph, neural kernel, retinal
sampling, readout manifest, and `NeuralControls` decoder are used. By default the
existing `bci` decoder is selected. Its Boolean `attack` output is reinterpreted
as the only Flappy Bird action: **flap**. `turn` and `forward` are ignored.

The game never feeds bird position, velocity, pipe position, score, reward, or
collision state into the controller. Those values exist only for audit and
evaluation. Every action therefore follows this path:

`RGB frame -> retinal samples -> MaleCNS neural propagation -> fixed decoder -> attack -> flap`

This is a simulation using a connectome-derived model, not a literal living fly
or a complete biological emulation.

## First run

Build the existing Doomfly neural artifacts first, following the repository's
normal setup. Then:

```bash
python -m flappy.run --episodes 10 --decoder bci --condition intact
```

Useful prespecified controls:

```bash
python -m flappy.run --episodes 10 --decoder bci --condition blank_vision
python -m flappy.run --episodes 10 --decoder bci --condition controls_clamped
```

Each run writes per-tick JSONL under `outputs/flappy/` and prints a summary with
mean score, mean survival time, maximum score, and total flaps.

## Why no learning yet?

v0 is intentionally a fixed-controller baseline. It asks whether the existing
connectome simulation plus fixed decoder produces nontrivial behavior in a
one-button visual task before adding plasticity or reinforcement. Any learning
experiment should be introduced separately with prespecified controls.

## Next controls

A matched-rate random-flap baseline should use the intact run's observed flap
rate, then be evaluated on separately seeded episodes. That avoids choosing an
arbitrary random-flap probability after seeing performance.
