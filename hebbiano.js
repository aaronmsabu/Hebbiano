'use strict';

// ==========================================
// HEBBIANO — Hebbian Learning Visualizer
// ==========================================

// === TUNING CONSTANTS (edit these to adjust behavior) ===
var ETA         = 0.015;    // Hebbian learning rate
var SHORT_DECAY = 0.95;     // Activity trace decay per frame (~230ms half-life at 60fps)
var LONG_DECAY  = 0.9998;   // Weight decay per frame (~58s half-life at 60fps)
var COINCIDENCE_WINDOW = 600; // ms — window for pulse animation trigger
const DISPLAY_LERP = 0.12;   // Smoothing factor for display weight interpolation

// Defaults for reset
var DEFAULTS = {
  ETA: 0.015,
  SHORT_DECAY: 0.95,
  LONG_DECAY: 0.9998,
  COINCIDENCE_WINDOW: 600
};

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
var currentMode  = 'learn'; // 'learn' or 'create'
var currentRule  = 'hebb';  // 'hebb', 'oja', or 'anti'

// Generative state
var temperature  = 1.0;
var tempoBpm     = 120;
var allowCascade = false;
var cascadeDepth = 0;
var MAX_CASCADE  = 4;

var CAPTIONS = {
  learn:  'Cells that fire together wire together \u2014 play some notes and watch the network learn. Click the grid to play it back.',
  create: 'The network is your instrument \u2014 play a note and let learned connections generate melodies. Teach it patterns, then jam.'
};

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

  // --- Generative Playback (Create Mode) ---
  if (currentMode === 'create') {
    var depth = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 0;
    cascadeDepth = depth;
    if (depth < MAX_CASCADE && (depth === 0 || allowCascade)) {
      generateFromNote(noteIndex, depth);
    }
    
    // --- Session Recording ---
    if (sessionRecording) {
      sessionEvents.push({
        note: note.name,
        noteIndex: noteIndex,
        timeMs: Math.round(performance.now() - sessionStartTime),
        depth: depth
      });
      document.querySelector('.rec-status').textContent = 'Recording... ' + sessionEvents.length + ' notes';
    }
  }
}

function generateFromNote(noteIndex, depth) {
  var row = weights[noteIndex];
  var probs = new Float64Array(N);
  var sum = 0;

  // 1. Extract weights
  for (var j = 0; j < N; j++) {
    probs[j] = row[j];
    sum += probs[j];
  }

  if (sum < 0.01) return; // No strong connections to generate from

  // 2. Apply temperature scaling: p' = p^(1/T)
  var scaledSum = 0;
  for (var j = 0; j < N; j++) {
    if (probs[j] > 0) {
      probs[j] = Math.pow(probs[j], 1 / temperature);
      scaledSum += probs[j];
    }
  }

  // 3. Normalize
  for (var j = 0; j < N; j++) {
    probs[j] /= scaledSum;
  }

  // 4. Determine how many notes to generate (1 to 3, depending on connection strength)
  var numNotes = sum > 1.5 ? 3 : sum > 0.8 ? 2 : 1;
  var generatedCount = 0;
  
  // 5. Sample and schedule
  for (var attempt = 0; attempt < numNotes * 3; attempt++) {
    if (generatedCount >= numNotes) break;
    
    var r = Math.random();
    var cumulative = 0;
    var selectedNote = -1;
    
    for (var j = 0; j < N; j++) {
      cumulative += probs[j];
      if (r <= cumulative) {
        selectedNote = j;
        break;
      }
    }
    
    if (selectedNote !== -1) {
      // Temporarily clear probability so we don't pick the same note twice
      var p = probs[selectedNote];
      probs[selectedNote] = 0;
      
      // Renormalize remaining
      var remSum = 0;
      for (var j = 0; j < N; j++) { remSum += probs[j]; }
      if (remSum > 0) {
        for (var j = 0; j < N; j++) { probs[j] /= remSum; }
      }
      
      // Schedule playback (steady 8th notes based on tempo)
      var eighthNoteMs = 30000 / tempoBpm;
      var delayMs = eighthNoteMs * (generatedCount + 1);
      (function(noteToPlay, currentDepth) {
        setTimeout(function() {
          play(noteToPlay, currentDepth + 1);
        }, delayMs);
      })(selectedNote, depth);
      
      generatedCount++;
    }
  }
}

// ==========================================
// HEBBIAN UPDATE
// ==========================================

function update() {
  var i, j, dw;

  // Learning rule + long-term decay + clamping
  for (i = 0; i < N; i++) {
    for (j = i + 1; j < N; j++) {
      // Compute weight delta based on active learning rule
      switch (currentRule) {
        case 'hebb':
          // Standard Hebbian: Δw = η · aᵢ · aⱼ
          dw = ETA * activity[i] * activity[j];
          break;
        case 'oja':
          // Oja's rule: Δw = η · aᵢ · (aⱼ − w · aᵢ)  — self-normalizing
          dw = ETA * activity[i] * (activity[j] - weights[i][j] * activity[i]);
          break;
        case 'anti':
          // Anti-Hebbian: Δw = −η · aᵢ · aⱼ  — decorrelation
          dw = -ETA * activity[i] * activity[j];
          break;
        default:
          dw = ETA * activity[i] * activity[j];
      }

      weights[i][j] += dw;
      weights[i][j] *= LONG_DECAY;
      if (weights[i][j] > 1) weights[i][j] = 1;
      if (weights[i][j] < 0) weights[i][j] = 0;
      if (weights[i][j] < 1e-6) weights[i][j] = 0; // Snap near-zero
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
// LEARN PANEL — Parameter Sliders
// ==========================================

var SLIDER_DEFS = [
  { key: 'ETA',                label: '\u03B7 (learning rate)',       min: 0.001, max: 0.05,   step: 0.001, fmt: function (v) { return v.toFixed(3); }  },
  { key: 'SHORT_DECAY',       label: 'Short decay',              min: 0.80,  max: 0.99,   step: 0.01,  fmt: function (v) { return v.toFixed(2); }  },
  { key: 'LONG_DECAY',        label: 'Long decay',               min: 0.9990, max: 0.9999, step: 0.0001, fmt: function (v) { return v.toFixed(4); } },
  { key: 'COINCIDENCE_WINDOW', label: 'Coincidence window (ms)', min: 100,   max: 1200,   step: 50,    fmt: function (v) { return v + 'ms'; }      },
];

var sliderInputs = {}; // key → input element
var sliderValues = {}; // key → value display element

function getParam(key) {
  switch (key) {
    case 'ETA': return ETA;
    case 'SHORT_DECAY': return SHORT_DECAY;
    case 'LONG_DECAY': return LONG_DECAY;
    case 'COINCIDENCE_WINDOW': return COINCIDENCE_WINDOW;
  }
}

function setParam(key, val) {
  switch (key) {
    case 'ETA': ETA = val; break;
    case 'SHORT_DECAY': SHORT_DECAY = val; break;
    case 'LONG_DECAY': LONG_DECAY = val; break;
    case 'COINCIDENCE_WINDOW': COINCIDENCE_WINDOW = val; break;
  }
}

function createLearnPanel() {
  var panel = document.getElementById('learn-panel');

  // — Parameter sliders section —
  var section = document.createElement('div');
  section.className = 'panel-section';

  var heading = document.createElement('div');
  heading.className = 'panel-heading';
  heading.textContent = 'Parameters';
  section.appendChild(heading);

  SLIDER_DEFS.forEach(function (def) {
    var row = document.createElement('div');
    row.className = 'slider-row';

    var label = document.createElement('label');
    label.className = 'slider-label';
    label.textContent = def.label;

    var val = document.createElement('span');
    val.className = 'slider-value';
    val.textContent = def.fmt(getParam(def.key));
    sliderValues[def.key] = val;

    var input = document.createElement('input');
    input.type = 'range';
    input.className = 'slider';
    input.min = def.min;
    input.max = def.max;
    input.step = def.step;
    input.value = getParam(def.key);
    sliderInputs[def.key] = input;

    // Closure to capture def
    (function (d) {
      input.addEventListener('input', function () {
        var v = parseFloat(this.value);
        setParam(d.key, v);
        sliderValues[d.key].textContent = d.fmt(v);
      });
    })(def);

    row.appendChild(label);
    row.appendChild(val);
    row.appendChild(input);
    section.appendChild(row);
  });

  // Reset defaults button
  var resetBtn = document.createElement('button');
  resetBtn.className = 'panel-btn';
  resetBtn.textContent = 'Reset defaults';
  resetBtn.addEventListener('click', function () {
    Object.keys(DEFAULTS).forEach(function (key) {
      setParam(key, DEFAULTS[key]);
      if (sliderInputs[key]) {
        sliderInputs[key].value = DEFAULTS[key];
      }
      SLIDER_DEFS.forEach(function (d) {
        if (d.key === key && sliderValues[key]) {
          sliderValues[key].textContent = d.fmt(DEFAULTS[key]);
        }
      });
    });
  });
  section.appendChild(resetBtn);

  panel.appendChild(section);

  // — Learning rule selector section —
  var ruleSection = document.createElement('div');
  ruleSection.className = 'panel-section';

  var ruleHeading = document.createElement('div');
  ruleHeading.className = 'panel-heading';
  ruleHeading.textContent = 'Learning Rule';
  ruleSection.appendChild(ruleHeading);

  var RULES = [
    { key: 'hebb', label: 'Hebbian',       desc: 'Connections strengthen when notes co-occur' },
    { key: 'oja',  label: 'Oja\u2019s',    desc: 'Self-normalizing \u2014 prevents saturation' },
    { key: 'anti', label: 'Anti-Hebbian',   desc: 'Connections weaken when notes co-occur' },
  ];

  var ruleContainer = document.createElement('div');
  ruleContainer.className = 'rule-selector';

  var ruleDesc = document.createElement('div');
  ruleDesc.className = 'rule-desc';
  ruleDesc.id = 'rule-desc';
  ruleDesc.textContent = RULES[0].desc;

  RULES.forEach(function (rule) {
    var btn = document.createElement('button');
    btn.className = 'rule-pill' + (rule.key === currentRule ? ' active' : '');
    btn.textContent = rule.label;
    btn.dataset.rule = rule.key;
    btn.addEventListener('click', function () {
      currentRule = rule.key;
      // Update active pill
      ruleContainer.querySelectorAll('.rule-pill').forEach(function (p) {
        p.classList.toggle('active', p.dataset.rule === rule.key);
      });
      ruleDesc.textContent = rule.desc;
    });
    ruleContainer.appendChild(btn);
  });

  ruleSection.appendChild(ruleContainer);
  ruleSection.appendChild(ruleDesc);
  panel.appendChild(ruleSection);

  // — Live equation display section —
  var eqSection = document.createElement('div');
  eqSection.className = 'panel-section';

  var eqHeading = document.createElement('div');
  eqHeading.className = 'panel-heading';
  eqHeading.textContent = 'Active Equation';
  eqSection.appendChild(eqHeading);

  var eqDisplay = document.createElement('div');
  eqDisplay.className = 'equation-display';
  eqDisplay.id = 'equation-display';
  eqSection.appendChild(eqDisplay);

  panel.appendChild(eqSection);

  // Add experiments UI
  createExperimentsUI();
}

// --- Equation update (called each frame) ---
var EQUATION_TEMPLATES = {
  hebb: '\u0394w = \u03B7 \u00D7 a\u1D62 \u00D7 a\u2C7C',
  oja:  '\u0394w = \u03B7 \u00D7 a\u1D62 \u00D7 (a\u2C7C \u2212 w \u00D7 a\u1D62)',
  anti: '\u0394w = \u2212\u03B7 \u00D7 a\u1D62 \u00D7 a\u2C7C',
};

var lastEquationText = '';

function updateEquation() {
  var eqEl = document.getElementById('equation-display');
  if (!eqEl) return;

  var formula = EQUATION_TEMPLATES[currentRule] || EQUATION_TEMPLATES.hebb;
  var detail = '\u03B7 = ' + ETA.toFixed(3) +
               '    decay: ' + SHORT_DECAY.toFixed(2) +
               '/frame    weight decay: ' + LONG_DECAY.toFixed(4) + '/frame';
  var text = formula + '\n' + detail;

  if (text !== lastEquationText) {
    eqEl.innerHTML = '<div class="eq-formula">' + formula + '</div>' +
                     '<div class="eq-detail">' + detail + '</div>';
    lastEquationText = text;
  }
}

// ==========================================
// GUIDED EXPERIMENTS
// ==========================================

var EXPERIMENTS = [
  {
    id: 'chord',
    name: 'Chord Memory',
    instruction: 'Play C\u2013E\u2013G together 5 times. Watch the triangle of connections form.',
    setup: function () { currentRule = 'hebb'; syncRulePills(); },
    check: function () {
      return weights[0][4] > 0.5 && weights[0][7] > 0.5 && weights[4][7] > 0.5;
    }
  },
  {
    id: 'selective',
    name: 'Selective Wiring',
    instruction: 'Wire D\u2013F# without any other connection going above 0.3. Play only those two notes.',
    setup: function () { currentRule = 'hebb'; syncRulePills(); },
    check: function () {
      if (weights[2][6] < 0.5) return false; // D-F# must be strong
      for (var i = 0; i < N; i++) {
        for (var j = i + 1; j < N; j++) {
          if (i === 2 && j === 6) continue;
          if (weights[i][j] > 0.3) return false;
        }
      }
      return true;
    }
  },
  {
    id: 'decay',
    name: 'Decay Race',
    instruction: 'Teach C\u2013E (play 6\u00D7), then F\u2013A (play 3\u00D7). Watch which fades first!',
    setup: function () { currentRule = 'hebb'; syncRulePills(); },
    check: function () {
      // Complete when both patterns have been taught (both > 0.3)
      return weights[0][4] > 0.3 && weights[5][9] > 0.15;
    }
  },
  {
    id: 'antierase',
    name: 'Anti-Learning',
    instruction: 'First teach C\u2013E\u2013G (Hebbian), then switch to Anti-Hebbian and erase it.',
    setup: function () { currentRule = 'hebb'; syncRulePills(); },
    check: function () {
      // Succeeds when a previously strong connection is erased
      return currentRule === 'anti' &&
             weights[0][4] < 0.1 && weights[0][7] < 0.1 && weights[4][7] < 0.1;
    }
  }
];

var activeExperiment = null; // index into EXPERIMENTS, or null
var experimentCompleted = {}; // id → true
var experimentStatusEls = {}; // id → status element

function createExperimentsUI() {
  var panel = document.getElementById('learn-panel');

  var section = document.createElement('div');
  section.className = 'panel-section';

  var heading = document.createElement('div');
  heading.className = 'panel-heading';
  heading.textContent = 'Guided Experiments';
  section.appendChild(heading);

  EXPERIMENTS.forEach(function (exp, idx) {
    var card = document.createElement('div');
    card.className = 'experiment-card';
    card.id = 'exp-' + exp.id;

    var header = document.createElement('div');
    header.className = 'exp-header';

    var title = document.createElement('span');
    title.className = 'exp-title';
    title.textContent = (idx + 1) + '. ' + exp.name;

    var status = document.createElement('span');
    status.className = 'exp-status';
    status.id = 'exp-status-' + exp.id;
    experimentStatusEls[exp.id] = status;

    header.appendChild(title);
    header.appendChild(status);

    var instEl = document.createElement('div');
    instEl.className = 'exp-instruction';
    instEl.textContent = exp.instruction;

    var startBtn = document.createElement('button');
    startBtn.className = 'panel-btn exp-start';
    startBtn.textContent = 'Start';
    startBtn.addEventListener('click', function () {
      startExperiment(idx);
    });

    card.appendChild(header);
    card.appendChild(instEl);
    card.appendChild(startBtn);
    section.appendChild(card);
  });

  panel.appendChild(section);
}

function startExperiment(idx) {
  // Reset network
  resetNetwork();
  activeExperiment = idx;
  var exp = EXPERIMENTS[idx];
  if (exp.setup) exp.setup();

  // Update all experiment status indicators
  EXPERIMENTS.forEach(function (e) {
    var el = experimentStatusEls[e.id];
    if (el) {
      if (e === exp) {
        el.textContent = '\u25CF active';
        el.className = 'exp-status active';
      } else if (experimentCompleted[e.id]) {
        el.textContent = '\u2713';
        el.className = 'exp-status completed';
      } else {
        el.textContent = '';
        el.className = 'exp-status';
      }
    }
  });
}

function checkExperiments() {
  if (activeExperiment === null) return;
  var exp = EXPERIMENTS[activeExperiment];
  if (exp.check && exp.check()) {
    experimentCompleted[exp.id] = true;
    activeExperiment = null;
    var el = experimentStatusEls[exp.id];
    if (el) {
      el.textContent = '\u2713 Complete!';
      el.className = 'exp-status completed';
    }
    // Brief highlight on the card
    var card = document.getElementById('exp-' + exp.id);
    if (card) {
      card.classList.add('success');
      setTimeout(function () { card.classList.remove('success'); }, 1500);
    }
  }
}

function syncRulePills() {
  var pills = document.querySelectorAll('.rule-pill');
  pills.forEach(function (p) {
    p.classList.toggle('active', p.dataset.rule === currentRule);
  });
  // Update description
  var desc = document.getElementById('rule-desc');
  if (desc) {
    var RULE_DESCS = { hebb: 'Connections strengthen when notes co-occur', oja: 'Self-normalizing \u2014 prevents saturation', anti: 'Connections weaken when notes co-occur' };
    desc.textContent = RULE_DESCS[currentRule] || '';
  }
}

// ==========================================
// CREATE PANEL — Generative Controls & Recording
// ==========================================

var sessionRecording = false;
var sessionStartTime = 0;
var sessionEvents = [];
var autoPlayRunning = false;
var autoPlayTimeout = null;

function createCreatePanel() {
  var panel = document.getElementById('create-panel');

  // — Playback controls —
  var playSection = document.createElement('div');
  playSection.className = 'panel-section';

  var playHeading = document.createElement('div');
  playHeading.className = 'panel-heading';
  playHeading.textContent = 'Generative Settings';
  playSection.appendChild(playHeading);

  // Temperature slider
  var tempRow = document.createElement('div');
  tempRow.className = 'slider-row';

  var tempLabel = document.createElement('label');
  tempLabel.className = 'slider-label';
  tempLabel.textContent = 'Temperature (randomness)';

  var tempVal = document.createElement('span');
  tempVal.className = 'slider-value';
  tempVal.textContent = temperature.toFixed(1);

  var tempInput = document.createElement('input');
  tempInput.type = 'range';
  tempInput.className = 'slider';
  tempInput.min = 0.1;
  tempInput.max = 3.0;
  tempInput.step = 0.1;
  tempInput.value = temperature;

  tempInput.addEventListener('input', function () {
    temperature = parseFloat(this.value);
    tempVal.textContent = temperature.toFixed(1);
  });

  tempRow.appendChild(tempLabel);
  tempRow.appendChild(tempVal);
  tempRow.appendChild(tempInput);
  playSection.appendChild(tempRow);

  // Tempo slider
  var bpmRow = document.createElement('div');
  bpmRow.className = 'slider-row';

  var bpmLabel = document.createElement('label');
  bpmLabel.className = 'slider-label';
  bpmLabel.textContent = 'Tempo (BPM)';

  var bpmVal = document.createElement('span');
  bpmVal.className = 'slider-value';
  bpmVal.textContent = tempoBpm;

  var bpmInput = document.createElement('input');
  bpmInput.type = 'range';
  bpmInput.className = 'slider';
  bpmInput.min = 60;
  bpmInput.max = 240;
  bpmInput.step = 1;
  bpmInput.value = tempoBpm;

  bpmInput.addEventListener('input', function () {
    tempoBpm = parseInt(this.value, 10);
    bpmVal.textContent = tempoBpm;
  });

  bpmRow.appendChild(bpmLabel);
  bpmRow.appendChild(bpmVal);
  bpmRow.appendChild(bpmInput);
  playSection.appendChild(bpmRow);

  // Cascade toggle
  var cascadeRow = document.createElement('div');
  cascadeRow.className = 'toggle-row';
  
  var cascadeLabel = document.createElement('span');
  cascadeLabel.textContent = 'Cascade generation (notes can trigger more notes)';
  
  var cascadeToggle = document.createElement('input');
  cascadeToggle.type = 'checkbox';
  cascadeToggle.className = 'create-toggle';
  cascadeToggle.checked = allowCascade;
  cascadeToggle.addEventListener('change', function () {
    allowCascade = this.checked;
  });
  
  cascadeRow.appendChild(cascadeToggle);
  cascadeRow.appendChild(cascadeLabel);
  playSection.appendChild(cascadeRow);

  // Auto-play toggle
  var autoRow = document.createElement('div');
  autoRow.className = 'toggle-row';
  
  var autoLabel = document.createElement('span');
  autoLabel.textContent = 'Auto-play (continuous generative loop)';
  
  var autoToggle = document.createElement('input');
  autoToggle.type = 'checkbox';
  autoToggle.className = 'create-toggle';
  autoToggle.checked = autoPlayRunning;
  autoToggle.addEventListener('change', function () {
    autoPlayRunning = this.checked;
    if (autoPlayRunning) {
      runAutoPlay();
    } else {
      clearTimeout(autoPlayTimeout);
    }
  });
  
  autoRow.appendChild(autoToggle);
  autoRow.appendChild(autoLabel);
  playSection.appendChild(autoRow);

  panel.appendChild(playSection);

  // — Session Recording —
  var recSection = document.createElement('div');
  recSection.className = 'panel-section';

  var recHeading = document.createElement('div');
  recHeading.className = 'panel-heading';
  recHeading.textContent = 'Session Recording';
  recSection.appendChild(recHeading);

  var recControls = document.createElement('div');
  recControls.className = 'rec-controls';

  var btnRec = document.createElement('button');
  btnRec.className = 'panel-btn rec-btn';
  btnRec.innerHTML = '<span class="rec-dot"></span> Record';

  var btnStop = document.createElement('button');
  btnStop.className = 'panel-btn';
  btnStop.textContent = '\u25A0 Stop';
  btnStop.disabled = true;

  var btnExport = document.createElement('button');
  btnExport.className = 'panel-btn';
  btnExport.textContent = '\u2193 Export JSON';
  btnExport.disabled = true;

  var recStatus = document.createElement('span');
  recStatus.className = 'rec-status';
  recStatus.textContent = '';

  btnRec.addEventListener('click', function () {
    sessionRecording = true;
    sessionEvents = [];
    sessionStartTime = performance.now();
    btnRec.disabled = true;
    btnRec.classList.add('recording');
    btnStop.disabled = false;
    btnExport.disabled = true;
    recStatus.textContent = 'Recording...';
  });

  btnStop.addEventListener('click', function () {
    sessionRecording = false;
    btnRec.disabled = false;
    btnRec.classList.remove('recording');
    btnStop.disabled = true;
    btnExport.disabled = sessionEvents.length === 0;
    recStatus.textContent = sessionEvents.length + ' notes recorded';
  });

  btnExport.addEventListener('click', function () {
    var dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(sessionEvents, null, 2));
    var link = document.createElement('a');
    link.href = dataStr;
    link.download = 'hebbiano-session.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  recControls.appendChild(btnRec);
  recControls.appendChild(btnStop);
  recControls.appendChild(btnExport);
  recControls.appendChild(recStatus);
  recSection.appendChild(recControls);

  panel.appendChild(recSection);
}

function runAutoPlay() {
  if (!autoPlayRunning) return;
  
  // Pick a semi-random note to kick off generation
  // Biased toward notes that have connections
  var candidates = [];
  for (var i = 0; i < N; i++) {
    var sum = 0;
    for (var j = 0; j < N; j++) sum += weights[i][j];
    if (sum > 0.1) {
      // Weight candidate by sum
      for (var k = 0; k < Math.max(1, Math.floor(sum * 10)); k++) {
        candidates.push(i);
      }
    }
  }
  
  // Fallback to purely random if network is empty
  var noteToPlay = candidates.length > 0 
    ? candidates[Math.floor(Math.random() * candidates.length)]
    : Math.floor(Math.random() * N);
    
  play(noteToPlay);
  
  // Schedule next auto-play kick (steady quarter notes based on tempo)
  var quarterNoteMs = 60000 / tempoBpm;
  autoPlayTimeout = setTimeout(runAutoPlay, quarterNoteMs);
}

// ==========================================
// MODE SWITCHING
// ==========================================

function setMode(mode) {
  currentMode = mode;

  // Toggle button active states
  document.getElementById('mode-learn').classList.toggle('active', mode === 'learn');
  document.getElementById('mode-create').classList.toggle('active', mode === 'create');

  // Toggle panel visibility
  document.getElementById('learn-panel').classList.toggle('hidden', mode !== 'learn');
  document.getElementById('create-panel').classList.toggle('hidden', mode !== 'create');

  // Update caption
  document.getElementById('caption').textContent = CAPTIONS[mode];
}

// ==========================================
// MAIN LOOP
// ==========================================

function tick() {
  update();
  render();
  updatePianoHighlights();
  updateGridInfo();
  updateEquation();
  checkExperiments();
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
  createLearnPanel();
  createCreatePanel();
  setupGridInteraction();
  setupKeyboard();
  initMIDI();

  // Wire up control buttons
  document.getElementById('btn-demo').addEventListener('click', playDemo);
  document.getElementById('btn-reset').addEventListener('click', resetNetwork);
  document.getElementById('btn-save').addEventListener('click', saveSnapshot);

  // Wire up mode toggle
  document.getElementById('mode-learn').addEventListener('click', function () { setMode('learn'); });
  document.getElementById('mode-create').addEventListener('click', function () { setMode('create'); });

  // Start animation loop
  requestAnimationFrame(tick);
}

init();
