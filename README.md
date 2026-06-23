# Hebbiano

**Interactive Hebbian Learning Visualizer** — play music on a piano and watch a neural network rewire itself in real time.

> *"Cells that fire together wire together."* — Donald Hebb, 1949

## What It Does

A 16×16 grid of synapses represents connections between 16 chromatic notes (C4–D#5). When two notes are played close together, the synapse connecting them strengthens.

Hebbiano features two distinct modes:

### 🎓 Learn Mode (Educational)
Experiment with the mechanics of neural learning.
- **Parameter Sliders**: Adjust learning rate (η), decay speeds, and coincidence windows in real time.
- **Learning Rules**: Toggle between standard **Hebbian** (strengthening), **Oja's rule** (self-normalizing), and **Anti-Hebbian** (decorrelation/weakening).
- **Live Math**: See the actual Δw equation update as you change parameters.
- **Guided Experiments**: Try 4 built-in challenges (like "Selective Wiring" and "Decay Race") with automatic success detection.

### 🎹 Create Mode (Generative)
Use the network as a probabilistic generative instrument.
- **Generative Playback**: Play a note, and the network will probabilistically trigger connected notes based on learned weight strengths.
- **Temperature Control**: Slide from 0.1 (deterministic, plays only strongest links) to 3.0 (highly random/chaotic).
- **Cascade**: Allow generated notes to trigger further generation, creating cascading melodies.
- **Auto-Play**: Starts a continuous loop where the network plays itself and learns from its own output.
- **Session Recording**: Record everything that plays and export the event data to a `.json` file for use elsewhere.

## Quick Start

Open `index.html` in your browser. No server, no build step, no dependencies.

```
open index.html
```

## Keyboard Layout

Piano-style chromatic mapping on QWERTY:

```
Black keys:  W  E     T  Y  U     O  P
            C# D#    F# G# A#    C# D#
White keys: A  S  D  F  G  H  J  K  L
            C  D  E  F  G  A  B  C  D
            ←——— Octave 4 ———→  ←— 5 —→
```

## Tech Stack

- Vanilla JavaScript (no frameworks, no build step)
- Canvas2D for rendering
- Web Audio API for synthesis
- Optional Web MIDI API (feature-detected, degrades gracefully)

## Deploy

Push to GitHub and enable GitHub Pages, or deploy to Vercel/Netlify as a static site. Zero config needed.

## License

MIT
