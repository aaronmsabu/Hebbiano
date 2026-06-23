'use strict';

// ==========================================
// HEBBIANO — Hebbian Learning Visualizer
// ==========================================

// === TUNING CONSTANTS (edit these to adjust behavior) ===
const ETA         = 0.015;    // Hebbian learning rate
const SHORT_DECAY = 0.95;     // Activity trace decay per frame (~230ms half-life at 60fps)
const LONG_DECAY  = 0.9998;   // Weight decay per frame (~58s half-life at 60fps)
const COINCIDENCE_WINDOW = 600; // ms — window for pulse animation trigger
const DISPLAY_LERP = 0.12;   // Smoothing factor for display weight interpolation

// === LAYOUT CONSTANTS ===
const N          = 16;        // Number of notes (fixed topology)
const LABEL_SIZE = 44;        // Pixel margin for axis labels on canvas
const CELL_SIZE  = 36;        // Pixel size per grid cell
const CELL_GAP   = 1;         // Visual gap between cells (pixels)
const CANVAS_SIZE = LABEL_SIZE + N * CELL_SIZE; // 44 + 576 = 620

// === NOTE DEFINITIONS ===
// 16 chromatic notes, C4 through D#5
// Keyboard bindings mirror a piano layout on QWERTY:
//   Bottom row (white): A S D F G H J K L
//   Top row (black):    W E   T Y U   O P
const NOTES = [
  { name: 'C4',  freq: 261.63, key: 'a', black: false },
  { name: 'C#4', freq: 277.18, key: 'w', black: true  },
  { name: 'D4',  freq: 293.66, key: 's', black: false },
  { name: 'D#4', freq: 311.13, key: 'e', black: true  },
  { name: 'E4',  freq: 329.63, key: 'd', black: false },
  { name: 'F4',  freq: 349.23, key: 'f', black: false },
  { name: 'F#4', freq: 369.99, key: 't', black: true  },
  { name: 'G4',  freq: 392.00, key: 'g', black: false },
  { name: 'G#4', freq: 415.30, key: 'y', black: true  },
  { name: 'A4',  freq: 440.00, key: 'h', black: false },
  { name: 'A#4', freq: 466.16, key: 'u', black: true  },
  { name: 'B4',  freq: 493.88, key: 'j', black: false },
  { name: 'C5',  freq: 523.25, key: 'k', black: false },
  { name: 'C#5', freq: 554.37, key: 'o', black: true  },
  { name: 'D5',  freq: 587.33, key: 'l', black: false },
  { name: 'D#5', freq: 622.25, key: 'p', black: true  },
];

// Build keyboard-key → note-index lookup
const KEY_MAP = {};
NOTES.forEach(function (n, i) { KEY_MAP[n.key] = i; });

// === COLOR LOOKUP TABLE ===
// Multi-stop heatmap: near-black → indigo → blue → cyan → green → yellow → white
const COLOR_STOPS = [
  [0.00,  10,  10,  20 ],
  [0.06,  18,  15,  55 ],
  [0.15,  28,  24, 100 ],
  [0.30,  35,  60, 165 ],
  [0.48,  20, 130, 200 ],
  [0.62,  25, 195, 195 ],
  [0.76, 100, 225, 160 ],
  [0.88, 210, 240, 120 ],
  [1.00, 255, 255, 235 ],
];

var COLOR_LUT = new Array(256);
(function buildColorLUT() {
  for (var idx = 0; idx <= 255; idx++) {
    var w = idx / 255;
    var lo = COLOR_STOPS[0];
    var hi = COLOR_STOPS[COLOR_STOPS.length - 1];
    for (var s = 0; s < COLOR_STOPS.length - 1; s++) {
      if (w >= COLOR_STOPS[s][0] && w <= COLOR_STOPS[s + 1][0]) {
        lo = COLOR_STOPS[s];
        hi = COLOR_STOPS[s + 1];
        break;
      }
    }
    var t = (hi[0] - lo[0]) > 0 ? (w - lo[0]) / (hi[0] - lo[0]) : 0;
    var r = Math.round(lo[1] + (hi[1] - lo[1]) * t);
    var g = Math.round(lo[2] + (hi[2] - lo[2]) * t);
    var b = Math.round(lo[3] + (hi[3] - lo[3]) * t);
    COLOR_LUT[idx] = 'rgb(' + r + ',' + g + ',' + b + ')';
  }
})();

var DIAG_COLOR = '#111118';
var BG_COLOR   = '#08080e';
var LABEL_FONT = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif';

// === STATE ===
var weights        = [];   // 16×16 Float64Array — actual Hebbian weights
var displayWeights = [];   // 16×16 Float64Array — smoothed for rendering
var i;
for (i = 0; i < N; i++) {
  weights.push(new Float64Array(N));
  displayWeights.push(new Float64Array(N));
}
var activity     = new Float64Array(N);  // Decaying activation trace per note
var lastPlayTime = new Float64Array(N);  // performance.now() timestamp of last play
var pulses       = [];   // { i, j, startTime } — active pulse animations

var audioCtx     = null;
var masterGain   = null;
var demoRunning  = false;
var demoTimeouts = [];
var hoverCell    = { row: -1, col: -1 };
var keyElements  = [];

// DOM references (set in init)
var canvas, ctx, gridInfoEl;

// ==========================================
// AUDIO
// ==========================================

function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    masterGain = audioCtx.createGain();
    masterGain.gain.value = 0.4;
    masterGain.connect(audioCtx.destination);
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}

/** Play a single note. This is THE canonical play function — every code path
 *  (keyboard, click, grid click, MIDI, demo) routes through here so that
 *  audio synthesis, Hebbian activity update, and pulse animation are never
 *  duplicated. */
function play(noteIndex) {
  ensureAudio();

  var note = NOTES[noteIndex];
  var now  = audioCtx.currentTime;

  // --- Synth: triangle oscillator with ADSR envelope ---
  var osc = audioCtx.createOscillator();
  var env = audioCtx.createGain();

  osc.type = 'triangle';
  osc.frequency.setValueAtTime(note.freq, now);

  // Attack 10ms → Decay 100ms → Sustain (0.2) hold → Release 300ms
  env.gain.setValueAtTime(0.001, now);
  env.gain.exponentialRampToValueAtTime(0.5, now + 0.01);     // Attack peak
  env.gain.exponentialRampToValueAtTime(0.2, now + 0.11);     // Decay to sustain
  env.gain.setValueAtTime(0.2, now + 0.30);                   // Hold sustain
  env.gain.exponentialRampToValueAtTime(0.001, now + 0.60);   // Release

  osc.connect(env);
  env.connect(masterGain);
  osc.start(now);
  osc.stop(now + 0.65);

  // --- Hebbian activity trace ---
  activity[noteIndex] = 1.0;

  // --- Coincidence detection for pulse animation ---
  var currentMs = performance.now();
  for (var j = 0; j < N; j++) {
    if (j !== noteIndex && lastPlayTime[j] > 0 &&
        currentMs - lastPlayTime[j] < COINCIDENCE_WINDOW) {
      // Pulse both (i,j) and (j,i) mirror cells
      pulses.push({ i: noteIndex, j: j, startTime: currentMs });
      pulses.push({ i: j, j: noteIndex, startTime: currentMs });
    }
  }

  lastPlayTime[noteIndex] = currentMs;
}

// ==========================================
// HEBBIAN UPDATE
// ==========================================

function update() {
  var i, j;

  // Hebbian learning rule + long-term decay + clamping
  for (i = 0; i < N; i++) {
    for (j = i + 1; j < N; j++) {
      weights[i][j] += ETA * activity[i] * activity[j];
      weights[i][j] *= LONG_DECAY;
      if (weights[i][j] > 1) weights[i][j] = 1;
      if (weights[i][j] < 1e-6) weights[i][j] = 0; // Snap near-zero to zero
      weights[j][i] = weights[i][j]; // Mirror — symmetric matrix
    }
  }

  // Decay activity traces
  for (i = 0; i < N; i++) {
    activity[i] *= SHORT_DECAY;
  }

  // Smooth display weights toward actual weights for fluid rendering
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      displayWeights[i][j] += (weights[i][j] - displayWeights[i][j]) * DISPLAY_LERP;
    }
  }
}

// ==========================================
// RENDERING
// ==========================================

function render() {
  var now = performance.now();

  // Clear
  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // --- Axis labels ---
  ctx.font = LABEL_FONT;
  ctx.fillStyle = '#4a5568';

  // Row labels (left side, horizontal)
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (var i = 0; i < N; i++) {
    ctx.fillText(
      NOTES[i].name,
      LABEL_SIZE - 6,
      LABEL_SIZE + i * CELL_SIZE + CELL_SIZE / 2
    );
  }

  // Column labels (top, rotated vertical)
  for (var j = 0; j < N; j++) {
    ctx.save();
    ctx.translate(
      LABEL_SIZE + j * CELL_SIZE + CELL_SIZE / 2 + 3,
      LABEL_SIZE - 6
    );
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(NOTES[j].name, 0, 0);
    ctx.restore();
  }

  // --- Grid cells ---
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      var x    = LABEL_SIZE + j * CELL_SIZE + CELL_GAP;
      var y    = LABEL_SIZE + i * CELL_SIZE + CELL_GAP;
      var size = CELL_SIZE - CELL_GAP * 2;

      if (i === j) {
        ctx.fillStyle = DIAG_COLOR;
      } else {
        var ci = Math.min(255, Math.max(0, Math.round(displayWeights[i][j] * 255)));
        ctx.fillStyle = COLOR_LUT[ci];
      }

      ctx.fillRect(x, y, size, size);
    }
  }

  // --- Hover highlight ---
  if (hoverCell.row >= 0 && hoverCell.col >= 0 &&
      hoverCell.row !== hoverCell.col) {
    var hx = LABEL_SIZE + hoverCell.col * CELL_SIZE;
    var hy = LABEL_SIZE + hoverCell.row * CELL_SIZE;
    ctx.strokeStyle = 'rgba(180, 200, 240, 0.45)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(hx + 1, hy + 1, CELL_SIZE - 2, CELL_SIZE - 2);
  }

  // --- Pulse animations ---
  for (var p = pulses.length - 1; p >= 0; p--) {
    var pulse   = pulses[p];
    var elapsed = now - pulse.startTime;
    if (elapsed > 400) {
      pulses.splice(p, 1);
      continue;
    }
    var t      = elapsed / 400;
    var radius = CELL_SIZE * 0.15 + CELL_SIZE * 0.5 * t;
    var alpha  = (1 - t) * 0.75;

    var cx = LABEL_SIZE + pulse.j * CELL_SIZE + CELL_SIZE / 2;
    var cy = LABEL_SIZE + pulse.i * CELL_SIZE + CELL_SIZE / 2;

    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(100, 220, 255, ' + alpha + ')';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // --- Subtle grid border ---
  ctx.strokeStyle = '#16161e';
  ctx.lineWidth = 1;
  ctx.strokeRect(LABEL_SIZE - 0.5, LABEL_SIZE - 0.5,
                 N * CELL_SIZE + 1, N * CELL_SIZE + 1);
}

// ==========================================
// PIANO
// ==========================================

function createPiano() {
  var piano = document.getElementById('piano');

  NOTES.forEach(function (note, i) {
    var key = document.createElement('div');
    key.className = 'key ' + (note.black ? 'black' : 'white');
    key.id = 'key-' + i;

    var nameSpan = document.createElement('span');
    nameSpan.className = 'key-note';
    nameSpan.textContent = note.name;

    var bindSpan = document.createElement('span');
    bindSpan.className = 'key-bind';
    bindSpan.textContent = note.key.toUpperCase();

    key.appendChild(nameSpan);
    key.appendChild(bindSpan);

    // Mouse and touch handlers — immediate response, no 300ms delay
    key.addEventListener('mousedown', function (e) {
      e.preventDefault();
      play(i);
    });
    key.addEventListener('touchstart', function (e) {
      e.preventDefault();
      play(i);
    }, { passive: false });

    keyElements.push(key);
    piano.appendChild(key);
  });
}

function updatePianoHighlights() {
  for (var i = 0; i < N; i++) {
    if (activity[i] > 0.15) {
      keyElements[i].classList.add('active');
    } else {
      keyElements[i].classList.remove('active');
    }
  }
}

// ==========================================
// GRID INTERACTION
// ==========================================

function getGridCell(clientX, clientY) {
  var rect   = canvas.getBoundingClientRect();
  var scaleX = CANVAS_SIZE / rect.width;
  var scaleY = CANVAS_SIZE / rect.height;
  var x = (clientX - rect.left) * scaleX;
  var y = (clientY - rect.top)  * scaleY;
  var col = Math.floor((x - LABEL_SIZE) / CELL_SIZE);
  var row = Math.floor((y - LABEL_SIZE) / CELL_SIZE);
  if (col >= 0 && col < N && row >= 0 && row < N) {
    return { row: row, col: col };
  }
  return null;
}

function setupGridInteraction() {
  canvas.addEventListener('mousemove', function (e) {
    var cell = getGridCell(e.clientX, e.clientY);
    if (cell && cell.row !== cell.col) {
      hoverCell = cell;
      canvas.style.cursor = 'pointer';
    } else {
      hoverCell = { row: -1, col: -1 };
      canvas.style.cursor = 'default';
    }
  });

  canvas.addEventListener('mouseleave', function () {
    hoverCell = { row: -1, col: -1 };
    canvas.style.cursor = 'default';
  });

  // Click-to-play: clicking cell (i,j) plays both notes, feeding back
  // into the Hebbian update — an intended emergent learning loop.
  canvas.addEventListener('click', function (e) {
    var cell = getGridCell(e.clientX, e.clientY);
    if (cell && cell.row !== cell.col) {
      play(cell.row);
      play(cell.col);
    }
  });

  // Touch support for mobile grid interaction
  canvas.addEventListener('touchstart', function (e) {
    e.preventDefault();
    var touch = e.touches[0];
    var cell  = getGridCell(touch.clientX, touch.clientY);
    if (cell && cell.row !== cell.col) {
      play(cell.row);
      play(cell.col);
    }
  }, { passive: false });
}

// ==========================================
// GRID INFO DISPLAY
// ==========================================

var lastInfoText = '';

function updateGridInfo() {
  var text = '\u00A0'; // non-breaking space when empty
  if (hoverCell.row >= 0 && hoverCell.col >= 0 &&
      hoverCell.row !== hoverCell.col) {
    var w = weights[hoverCell.row][hoverCell.col];
    text = NOTES[hoverCell.row].name + ' \u2194 ' +
           NOTES[hoverCell.col].name + '  \u00B7  strength: ' + w.toFixed(3);
  }
  if (text !== lastInfoText) {
    gridInfoEl.textContent = text;
    lastInfoText = text;
  }
}

// ==========================================
// KEYBOARD INPUT
// ==========================================

function setupKeyboard() {
  document.addEventListener('keydown', function (e) {
    if (e.repeat) return;
    // Don't intercept when typing in form fields
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    var noteIndex = KEY_MAP[e.key.toLowerCase()];
    if (noteIndex !== undefined) {
      e.preventDefault();
      play(noteIndex);
    }
  });
}

// ==========================================
// MIDI INPUT (optional, feature-detected)
// ==========================================

function initMIDI() {
  if (!navigator.requestMIDIAccess) return; // Safari/iOS — silently skip

  navigator.requestMIDIAccess({ sysex: false })
    .then(function (access) {
      function connectInputs() {
        var inputs = access.inputs;
        for (var input of inputs.values()) {
          input.onmidimessage = handleMIDIMessage;
        }
        document.getElementById('midi-status').textContent =
          inputs.size > 0 ? '\uD83C\uDFB9 MIDI connected' : '';
      }

      connectInputs();
      access.onstatechange = connectInputs;
    })
    .catch(function () {
      // Silently skip — MIDI is optional
    });
}

function handleMIDIMessage(msg) {
  var data     = msg.data;
  var status   = data[0];
  var note     = data[1];
  var velocity = data[2];

  // Note-on (0x90) with non-zero velocity
  if ((status & 0xF0) === 0x90 && velocity > 0) {
    var noteIndex = note - 60; // MIDI note 60 = C4
    if (noteIndex >= 0 && noteIndex < N) {
      play(noteIndex);
    }
  }
}

// ==========================================
// DEMO SEQUENCE
// ==========================================

function buildDemoSequence() {
  var seq = [];
  var r;

  // — C major arpeggio (C4 → E4 → G4) × 6 repetitions —
  for (r = 0; r < 6; r++) {
    seq.push({ note: 0,  delayMs: r === 0 ? 300 : 280 }); // C4
    seq.push({ note: 4,  delayMs: 160 });                   // E4
    seq.push({ note: 7,  delayMs: 160 });                   // G4
  }

  seq.push({ note: -1, delayMs: 700 }); // Pause

  // — F major arpeggio (F4 → A4 → C5) × 5 repetitions —
  for (r = 0; r < 5; r++) {
    seq.push({ note: 5,  delayMs: r === 0 ? 0 : 280 });   // F4
    seq.push({ note: 9,  delayMs: 160 });                   // A4
    seq.push({ note: 12, delayMs: 160 });                   // C5
  }

  seq.push({ note: -1, delayMs: 700 }); // Pause

  // — G major arpeggio (G4 → B4 → D5) × 5 repetitions —
  for (r = 0; r < 5; r++) {
    seq.push({ note: 7,  delayMs: r === 0 ? 0 : 280 });   // G4
    seq.push({ note: 11, delayMs: 160 });                   // B4
    seq.push({ note: 14, delayMs: 160 });                   // D5
  }

  return seq;
}

function playDemo() {
  if (demoRunning) return;
  ensureAudio(); // Must create AudioContext during user gesture (Chrome policy)
  demoRunning = true;

  var btnDemo = document.getElementById('btn-demo');
  btnDemo.textContent = '\u23F8 Playing\u2026';
  btnDemo.disabled = true;

  var sequence   = buildDemoSequence();
  var totalDelay = 0;
  demoTimeouts   = [];

  sequence.forEach(function (step) {
    totalDelay += step.delayMs;
    if (step.note >= 0) {
      var noteToPlay = step.note;
      var delay      = totalDelay;
      demoTimeouts.push(setTimeout(function () { play(noteToPlay); }, delay));
    }
  });

  // Re-enable button after sequence completes
  demoTimeouts.push(setTimeout(function () {
    demoRunning = false;
    btnDemo.textContent = '\u25B6 Play Demo';
    btnDemo.disabled = false;
  }, totalDelay + 1000));
}

// ==========================================
// CONTROLS
// ==========================================

function resetNetwork() {
  for (var i = 0; i < N; i++) {
    weights[i].fill(0);
    displayWeights[i].fill(0);
  }
  activity.fill(0);
  lastPlayTime.fill(0);
  pulses.length = 0;

  // Cancel running demo if any
  if (demoRunning) {
    demoTimeouts.forEach(clearTimeout);
    demoTimeouts = [];
    demoRunning  = false;
    var btnDemo  = document.getElementById('btn-demo');
    btnDemo.textContent = '\u25B6 Play Demo';
    btnDemo.disabled = false;
  }
}

function saveSnapshot() {
  var link = document.createElement('a');
  link.download = 'hebbiano-snapshot.png';
  link.href = canvas.toDataURL('image/png');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ==========================================
// MAIN LOOP
// ==========================================

function tick() {
  update();
  render();
  updatePianoHighlights();
  updateGridInfo();
  requestAnimationFrame(tick);
}

// ==========================================
// INITIALIZATION
// ==========================================

function init() {
  canvas    = document.getElementById('grid');
  ctx       = canvas.getContext('2d');
  gridInfoEl = document.getElementById('grid-info');

  // High-DPI canvas for crisp rendering on retina displays
  var dpr = window.devicePixelRatio || 1;
  canvas.width  = CANVAS_SIZE * dpr;
  canvas.height = CANVAS_SIZE * dpr;
  canvas.style.width  = CANVAS_SIZE + 'px';
  canvas.style.height = CANVAS_SIZE + 'px';
  ctx.scale(dpr, dpr);

  // Build UI
  createPiano();
  setupGridInteraction();
  setupKeyboard();
  initMIDI();

  // Wire up control buttons
  document.getElementById('btn-demo').addEventListener('click', playDemo);
  document.getElementById('btn-reset').addEventListener('click', resetNetwork);
  document.getElementById('btn-save').addEventListener('click', saveSnapshot);

  // Start animation loop
  requestAnimationFrame(tick);
}

init();
