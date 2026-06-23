# Hebbiano

**Interactive Hebbian Learning Visualizer** — play music on a piano and watch a neural network rewire itself in real time.

> *"Cells that fire together wire together."* — Donald Hebb, 1949

## What It Does

A 16×16 grid of synapses represents connections between 16 chromatic notes (C4–D#5). When two notes are played close together, the synapse connecting them strengthens — a literal demonstration of Hebbian learning. Click any strengthened cell to play its two notes back, feeding into the learning loop.

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

## Controls

| Button | Action |
|--------|--------|
| ▶ Play Demo | Runs a scripted sequence of 3 chord progressions |
| ↺ Reset Network | Zeros all weights instantly |
| ⬇ Save Snapshot | Exports the grid as a PNG |

## How It Works

- **Activity trace**: Each note maintains a decaying activation signal (`SHORT_DECAY = 0.95` per frame)
- **Learning rule**: `w[i][j] += η × activity[i] × activity[j]` each frame, clamped to [0, 1]
- **Long-term decay**: `w[i][j] *= 0.9998` per frame (~58s half-life) — patterns persist ~60s then fade
- **Pulse animation**: Coincident notes (within 600ms) trigger expanding ring animations on the grid

## Tuning

Constants are at the top of `hebbiano.js`:

```javascript
const ETA         = 0.015;   // Learning rate
const SHORT_DECAY = 0.95;    // Activity decay per frame
const LONG_DECAY  = 0.9998;  // Weight decay per frame
const COINCIDENCE_WINDOW = 600; // ms
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
