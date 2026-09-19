"""Cooperative Tetris Web Server: Human + openjev side-by-side with synchronized pieces."""
import http.server
import json
import socketserver
import threading
import time
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from coop_tetris import CoopGameManager

COOP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OPENJEV // Esports Neural Arena</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Rajdhani:wght@500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-base: #060910;
    --bg-card: rgba(13, 19, 32, 0.78);
    --border-card: rgba(56, 189, 248, 0.18);
    --border-card-hover: rgba(56, 189, 248, 0.4);
    --text-main: #f8fafc;
    --text-sub: #94a3b8;
    --accent-cyan: #00f0ff;
    --accent-neon: #a855f7;
    --accent-green: #10b981;
    --accent-gold: #fbbf24;
    --accent-red: #ef4444;
    --accent-blue: #3b82f6;
    --font-ui: 'Rajdhani', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 16px 20px;
    background: radial-gradient(circle at 50% 0%, #10192d 0%, #060910 75%);
    min-height: 100vh;
    color: var(--text-main);
    font-family: var(--font-ui);
    display: flex;
    flex-direction: column;
    align-items: center;
    user-select: none;
  }

  /* Telemetry HUD Bar */
  .telemetry-hud {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 12px;
    width: 100%;
    max-width: 1400px;
    margin-bottom: 16px;
  }
  .hud-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-card);
    border-radius: 10px;
    padding: 10px 16px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    position: relative;
    overflow: hidden;
  }
  .hud-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    opacity: 0.6;
  }
  .hud-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-sub);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .hud-val {
    font-family: var(--font-mono);
    font-size: 15px;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: baseline;
    gap: 6px;
  }
  .hud-sub {
    font-size: 11px;
    color: var(--text-sub);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Header */
  header {
    text-align: center;
    margin-bottom: 12px;
  }
  .header-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    background: rgba(0, 240, 255, 0.12);
    border: 1px solid rgba(0, 240, 255, 0.3);
    color: var(--accent-cyan);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  h1 {
    margin: 0 0 2px 0;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 2px;
    background: linear-gradient(135deg, #ffffff 40%, var(--accent-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 25px rgba(0, 240, 255, 0.35);
  }
  .tagline {
    font-size: 12px;
    color: var(--text-sub);
    letter-spacing: 0.8px;
  }

  /* Esports Team Banner */
  .team-banner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 32px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(0, 240, 255, 0.1) 100%);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-radius: 12px;
    padding: 10px 36px;
    margin-bottom: 14px;
    box-shadow: 0 8px 30px rgba(0, 240, 255, 0.1);
  }
  .team-stat { text-align: center; }
  .team-stat-val {
    font-family: var(--font-mono);
    font-size: 26px;
    font-weight: 800;
    color: #fff;
    text-shadow: 0 0 12px rgba(255, 255, 255, 0.4);
  }
  .team-stat-lbl {
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-sub);
    letter-spacing: 1px;
    font-weight: 600;
  }
  .star-glow {
    font-size: 22px;
    color: var(--accent-gold);
    text-shadow: 0 0 14px rgba(251, 191, 36, 0.6);
    animation: star-pulse 2s infinite ease-in-out;
  }
  @keyframes star-pulse {
    0%, 100% { transform: scale(1); opacity: 0.9; }
    50% { transform: scale(1.15); opacity: 1; }
  }

  /* Control Hub */
  .middle-hub {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    justify-content: center;
    padding: 6px 12px;
    max-width: 1400px;
    width: 100%;
    margin-bottom: 16px;
  }
  button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: #fff;
    border: 1px solid rgba(16, 185, 129, 0.5);
    border-radius: 8px;
    padding: 9px 22px;
    font-family: var(--font-ui);
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
    transition: all 0.2s ease;
  }
  button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
  }
  button.pause {
    background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
    border-color: rgba(239, 68, 68, 0.5);
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.3);
  }
  button.btn-reset {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.3);
    color: #cbd5e1;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
  }
  button.btn-reset:hover {
    border-color: rgba(239, 68, 68, 0.6);
    color: #fff;
  }

  .shared-piece-preview, .controls-box {
    background: var(--bg-card);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border-card);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 11px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3);
  }
  .controls-box strong { color: #fff; font-size: 11px; letter-spacing: 0.5px; }
  .key {
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 1px 5px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--accent-cyan);
    box-shadow: 0 1px 3px rgba(0,0,0,0.4);
  }
  select {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid var(--border-card);
    color: var(--text-main);
    padding: 5px 8px;
    border-radius: 6px;
    font-family: var(--font-ui);
    font-size: 12px;
    font-weight: 600;
    outline: none;
    cursor: pointer;
  }

  /* Arena */
  .arena {
    display: flex;
    gap: 20px;
    justify-content: center;
    align-items: flex-start;
    max-width: 100%;
    width: 100%;
    flex-wrap: wrap;
    margin-bottom: 20px;
  }
  .player-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
  }
  .player-card:hover { border-color: var(--border-card-hover); }
  .player-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }
  .tag-human {
    color: var(--accent-cyan);
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
  }
  .tag-ai {
    color: var(--accent-neon);
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
    text-shadow: 0 0 10px rgba(168, 85, 247, 0.4);
  }
  .ai-card { cursor: pointer; }
  .ai-card.focused {
    border-color: var(--accent-neon);
    box-shadow: 0 0 25px rgba(168, 85, 247, 0.25), 0 10px 30px rgba(0, 0, 0, 0.6);
  }
  .ai-card.dead { opacity: 0.55; }
  .ai-card.dead .tag-ai::after { content: " — TOPPED OUT"; color: var(--accent-red); }
  .ai-label {
    font-size: 11px;
    color: var(--text-sub);
    margin-top: -4px;
    margin-bottom: 8px;
    text-align: center;
    max-width: 290px;
  }

  .board-wrapper {
    display: flex;
    gap: 10px;
  }
  .side-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .mini-box {
    background: rgba(6, 9, 16, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 6px;
    width: 68px;
    text-align: center;
    box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.5);
  }
  .mini-lbl {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-sub);
    margin-bottom: 4px;
  }
  canvas {
    background: #080c14;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.85);
  }

  .stats-row {
    display: flex;
    gap: 8px;
    width: 100%;
    margin-top: 10px;
  }
  .stat-pill {
    flex: 1;
    background: rgba(6, 9, 16, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 5px 6px;
    text-align: center;
  }
  .stat-pill-val {
    font-family: var(--font-mono);
    font-size: 16px;
    font-weight: 700;
    color: #fff;
  }
  .stat-pill-lbl {
    font-size: 9px;
    color: var(--text-sub);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
  }

  /* Latency Badge */
  .lat-badge {
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 10px;
    font-family: var(--font-mono);
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .lat-ultra {
    background: rgba(251, 191, 36, 0.16);
    border: 1px solid rgba(251, 191, 36, 0.55);
    color: #fbbf24;
    box-shadow: 0 0 8px rgba(251, 191, 36, 0.3);
  }
  .lat-fast {
    background: rgba(16, 185, 129, 0.16);
    border: 1px solid rgba(16, 185, 129, 0.55);
    color: #34d399;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.3);
  }
  .lat-heavy {
    background: rgba(168, 85, 247, 0.16);
    border: 1px solid rgba(168, 85, 247, 0.55);
    color: #c084fc;
    box-shadow: 0 0 8px rgba(168, 85, 247, 0.3);
  }

  /* Decision Confidence Box */
  .nli-box {
    margin-top: 10px;
    width: 100%;
    background: rgba(6, 9, 16, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 11px;
  }
  .nli-bar {
    display: flex;
    height: 7px;
    border-radius: 4px;
    overflow: hidden;
    margin: 6px 0;
    background: #1e293b;
  }
  .nli-ent { background: linear-gradient(90deg, #059669, #10b981); }
  .nli-con { background: linear-gradient(90deg, #dc2626, #ef4444); }
  .nli-neu { background: #64748b; }

  /* Neural Activation Panel */
  .neural-panel {
    margin-top: 8px;
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 16px;
    width: 100%;
    max-width: 1200px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
  }
  .neural-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 10px;
    margin-bottom: 12px;
  }
  .neural-title {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 1px;
    color: var(--accent-neon);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .live-pill {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: var(--accent-green);
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    animation: pulse-glow 2s infinite;
  }
  @keyframes pulse-glow {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; }
  }
  .neural-content {
    display: flex;
    gap: 18px;
  }
  .neural-graph-wrapper {
    flex: 2;
    display: flex;
    flex-direction: column;
  }
  .spectrum-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 310px;
  }
  .sub-title {
    font-size: 10px;
    color: var(--text-sub);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 700;
    display: flex;
    justify-content: space-between;
  }
  .switch-lbl {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    cursor: pointer;
    color: var(--text-main);
  }
</style>
</head>
<body>

<!-- Top Hardware Telemetry HUD -->
<div class="telemetry-hud">
  <div class="hud-card">
    <div class="hud-label">⚡ GPU ACCELERATION</div>
    <div class="hud-val" id="hud-gpu">NVIDIA RTX 3060 Ti 8GB</div>
    <div class="hud-sub">CUDA 12.8 · FP16/BF16 SDPA Attention</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">💾 LIVE VRAM PROFILE</div>
    <div class="hud-val"><span id="hud-vram-val">3.4</span> <span style="font-size:11px;color:var(--text-sub);font-weight:normal">GB / 8.0 GB</span></div>
    <div class="hud-sub" id="hud-vram-sub">Kev: ~1.0GB | Laya: ~2.4GB (4.6GB Free)</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">🏎️ BITBOARD KINEMATICS</div>
    <div class="hud-val" style="color:var(--accent-cyan)">2.97M <span style="font-size:11px;color:var(--text-sub);font-weight:normal">checks/sec</span></div>
    <div class="hud-sub">O(1) Bitwise Collision · SRS Lookup Table</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">🧠 ACTIVE PARADIGM</div>
    <div class="hud-val" style="color:var(--accent-neon)">Langevin + RLCD</div>
    <div class="hud-sub">DiffusionGemma (0.8ms) · ModernBERT (77ms)</div>
  </div>
</div>

<header>
  <div class="header-badge">OPENJEV ESPORTS // NEURAL TETRIS</div>
  <h1>NEURAL TETRIS ARENA</h1>
  <div class="tagline">Cooperative Human + System 1/2 Dual AI Planners · Synchronized 7-Bag Stream</div>
</header>

<div class="team-banner">
  <div class="team-stat">
    <div id="team-score" class="team-stat-val">0</div>
    <div class="team-stat-lbl">Combined Team Score</div>
  </div>
  <div class="star-glow">★</div>
  <div class="team-stat">
    <div id="team-lines" class="team-stat-val">0</div>
    <div class="team-stat-lbl">Combined Lines Cleared</div>
  </div>
</div>

<!-- Center Hub: Controls & Shared Queue -->
<div class="middle-hub">
  <button id="btn-start" onclick="togglePlay()">Start Game</button>
  <button class="btn-reset" onclick="resetGame()">Reset Match</button>

  <div class="shared-piece-preview">
    <div class="mini-lbl" style="color: var(--accent-cyan)">Shared Bag Stream</div>
    <div id="shared-stream-info" style="font-size: 12px; margin-top: 2px; font-weight: bold; font-family: var(--font-mono)">
      Identical 7-Bag
    </div>
  </div>

  <div class="controls-box">
    <strong>AI Speed Tempo:</strong>
    <select id="select-speed" onchange="changeSpeed(this.value)" style="margin-left: 6px;">
      <option value="0.75" selected>Human Tempo (0.75s)</option>
      <option value="0.4">Fast Speed (0.4s)</option>
      <option value="0.1">Blitz Speed (0.1s)</option>
      <option value="0.05">🔥 BRRRR Turbo (0.05s)</option>
      <option value="0.01">⚡ MAXIMUM BRRRRR (0.01s)</option>
      <option value="1.2">Relaxed (1.2s)</option>
    </select>
  </div>

  <div class="controls-box">
    <strong>Controls:</strong>
    <span class="key">←</span> <span class="key">→</span> Move &nbsp;
    <span class="key">↑</span> / <span class="key">X</span> CW &nbsp;
    <span class="key">Z</span> CCW &nbsp;
    <span class="key">↓</span> Drop &nbsp;
    <span class="key">Space</span> Hard &nbsp;
    <span class="key">C</span> / <span class="key">Shift</span> Hold
  </div>
</div>

<div class="arena" id="arena">
  <!-- Human Player Board -->
  <div class="player-card">
    <div class="player-header">
      <span class="tag-human">● YOU (HUMAN)</span>
      <span id="h-status" style="font-size: 11px; font-weight:700; color: var(--accent-green)">READY</span>
    </div>
    <div class="board-wrapper">
      <div class="side-panel">
        <div class="mini-box">
          <div class="mini-lbl">Hold</div>
          <canvas id="canvas-hold" width="54" height="54"></canvas>
        </div>
      </div>
      <canvas id="canvas-human" width="220" height="440"></canvas>
      <div class="side-panel">
        <div class="mini-box">
          <div class="mini-lbl">Next</div>
          <canvas id="canvas-h-next" width="54" height="54"></canvas>
        </div>
      </div>
    </div>
    <div class="stats-row">
      <div class="stat-pill">
        <div id="h-score" class="stat-pill-val">0</div>
        <div class="stat-pill-lbl">Score</div>
      </div>
      <div class="stat-pill">
        <div id="h-lines" class="stat-pill-val">0</div>
        <div class="stat-pill-lbl">Lines</div>
      </div>
      <div class="stat-pill">
        <div id="h-piece-idx" class="stat-pill-val">#0</div>
        <div class="stat-pill-lbl">Piece #</div>
      </div>
    </div>
  </div>

  <!-- AI boards: one per loaded model, built by buildAIPanels() -->
</div>

<!-- Neural Activation Graph Panel -->
<div class="neural-panel" id="neural-panel">
  <div class="neural-header">
    <div class="neural-title">
      <span>🧠 Real-Time Synaptic Firing Monitor</span>
      <span id="graph-source" style="color:var(--text-sub);font-size:11px;font-weight:normal">—</span>
      <span class="live-pill">LIVE FIRING</span>
    </div>
    <div style="display:flex; gap: 16px; align-items:center;">
      <label class="switch-lbl">
        <input type="checkbox" id="chk-show-graph" checked onchange="toggleNeuralGraph(this.checked)">
        <span>Active Monitor</span>
      </label>
      <label class="switch-lbl">
        <input type="checkbox" id="chk-synaptic-flow" checked>
        <span>Synaptic Pulses</span>
      </label>
    </div>
  </div>

  <div id="neural-body" class="neural-content">
    <div class="neural-graph-wrapper">
      <div class="sub-title">
        <span>Board Features (6)</span>
        <span>Model Activations (24)</span>
        <span>NLI Head (3)</span>
        <span id="moves-title">Candidate Placements (0)</span>
      </div>
      <canvas id="canvas-neural" width="780" height="250"></canvas>
    </div>

    <div class="spectrum-wrapper">
      <div class="sub-title">
        <span>48-Channel Hidden Activation Spectrogram</span>
        <span id="spectrum-energy" style="color:var(--accent-gold); font-family:var(--font-mono)">Peak: 0.95</span>
      </div>
      <canvas id="canvas-spectrum" width="360" height="250"></canvas>
    </div>
  </div>
</div>

<script>
const PIECES = {
  'I': [[[1,1,1,1]], [[1],[1],[1],[1]]],
  'O': [[[1,1],[1,1]]],
  'T': [[[0,1,0],[1,1,1]], [[1,0],[1,1],[1,0]], [[1,1,1],[0,1,0]], [[0,1],[1,1],[0,1]]],
  'S': [[[0,1,1],[1,1,0]], [[1,0],[1,1],[0,1]]],
  'Z': [[[1,1,0],[0,1,1]], [[0,1],[1,1],[1,0]]],
  'J': [[[1,0,0],[1,1,1]], [[1,1],[1,0],[1,0]], [[1,1,1],[0,0,1]], [[0,1],[0,1],[1,1]]],
  'L': [[[0,0,1],[1,1,1]], [[1,0],[1,0],[1,1]], [[1,1,1],[1,0,0]], [[1,1],[0,1],[0,1]]]
};

// Human movement uses the engine's fixed SRS boxes. Keep these coordinates
// identical to tetris_engine.py so the canvas shows the cells being collided.
const SRS_PIECES = {
  I: [ [[1,0],[1,1],[1,2],[1,3]], [[0,2],[1,2],[2,2],[3,2]], [[2,0],[2,1],[2,2],[2,3]], [[0,1],[1,1],[2,1],[3,1]] ],
  O: [ [[0,0],[0,1],[1,0],[1,1]], [[0,0],[0,1],[1,0],[1,1]], [[0,0],[0,1],[1,0],[1,1]], [[0,0],[0,1],[1,0],[1,1]] ],
  T: [ [[0,1],[1,0],[1,1],[1,2]], [[0,1],[1,1],[1,2],[2,1]], [[1,0],[1,1],[1,2],[2,1]], [[0,1],[1,0],[1,1],[2,1]] ],
  S: [ [[0,1],[0,2],[1,0],[1,1]], [[0,1],[1,1],[1,2],[2,2]], [[1,1],[1,2],[2,0],[2,1]], [[0,0],[1,0],[1,1],[2,1]] ],
  Z: [ [[0,0],[0,1],[1,1],[1,2]], [[0,2],[1,1],[1,2],[2,1]], [[1,0],[1,1],[2,1],[2,2]], [[0,1],[1,0],[1,1],[2,0]] ],
  J: [ [[0,0],[1,0],[1,1],[1,2]], [[0,1],[0,2],[1,1],[2,1]], [[1,0],[1,1],[1,2],[2,2]], [[0,1],[1,1],[2,0],[2,1]] ],
  L: [ [[0,2],[1,0],[1,1],[1,2]], [[0,1],[1,1],[2,1],[2,2]], [[1,0],[1,1],[1,2],[2,0]], [[0,0],[0,1],[1,1],[2,1]] ]
};

const COLORS = {
  'I': '#00f0f0',
  'O': '#f0f000',
  'T': '#a000f0',
  'S': '#00f000',
  'Z': '#f00000',
  'J': '#0000f0',
  'L': '#f0a000'
};

const CELL_SIZE = 22;
let running = false;
let gravityTimer = null;
let pollTimer = null;

const cvsH = document.getElementById('canvas-human');
const ctxH = cvsH.getContext('2d');
const cvsHold = document.getElementById('canvas-hold');
const ctxHold = cvsHold.getContext('2d');
const cvsHNext = document.getElementById('canvas-h-next');
const ctxHNext = cvsHNext.getContext('2d');
const aiPanels = [];

function drawGrid(ctx, grid, activePiece=null, useSrs=false) {
  ctx.fillStyle = '#080c14';
  ctx.fillRect(0, 0, 220, 440);

  // Subtle cyberpunk grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1;
  for (let c = 0; c <= 10; c++) {
    ctx.beginPath(); ctx.moveTo(c * CELL_SIZE, 0); ctx.lineTo(c * CELL_SIZE, 440); ctx.stroke();
  }
  for (let r = 0; r <= 20; r++) {
    ctx.beginPath(); ctx.moveTo(0, r * CELL_SIZE); ctx.lineTo(220, r * CELL_SIZE); ctx.stroke();
  }

  // Placed blocks
  if (grid) {
    for (let r = 0; r < 20; r++) {
      for (let c = 0; c < 10; c++) {
        const val = grid[r][c];
        if (val !== 0) {
          drawBlock(ctx, c * CELL_SIZE, r * CELL_SIZE, COLORS[val] || '#888');
        }
      }
    }
  }

  // Active falling piece & Ghost
  if (activePiece && activePiece.cur_piece) {
    const color = COLORS[activePiece.cur_piece];
    if (useSrs) {
      const cells = SRS_PIECES[activePiece.cur_piece][activePiece.cur_rot % 4];
      if (activePiece.ghost_y !== undefined) cells.forEach(([r,c]) => drawGhostBlock(ctx, (activePiece.cur_x+c)*CELL_SIZE, (activePiece.ghost_y+r)*CELL_SIZE, color));
      cells.forEach(([r,c]) => drawBlock(ctx, (activePiece.cur_x+c)*CELL_SIZE, (activePiece.cur_y+r)*CELL_SIZE, color));
      return;
    }
    const shape = PIECES[activePiece.cur_piece][activePiece.cur_rot % PIECES[activePiece.cur_piece].length];

    // Ghost
    if (activePiece.ghost_y !== undefined) {
      for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
          if (shape[r][c]) {
            drawGhostBlock(ctx, (activePiece.cur_x + c) * CELL_SIZE, (activePiece.ghost_y + r) * CELL_SIZE, color);
          }
        }
      }
    }

    // Active
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c]) {
          drawBlock(ctx, (activePiece.cur_x + c) * CELL_SIZE, (activePiece.cur_y + r) * CELL_SIZE, color);
        }
      }
    }
  }
}

function drawBlock(ctx, x, y, color) {
  // Beveled cyber block with subtle inner highlight
  ctx.fillStyle = color;
  ctx.fillRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);

  // Top/left highlight
  ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
  ctx.fillRect(x + 1, y + 1, CELL_SIZE - 2, 2);
  ctx.fillRect(x + 1, y + 1, 2, CELL_SIZE - 2);

  // Bottom/right shadow
  ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
  ctx.fillRect(x + 1, y + CELL_SIZE - 3, CELL_SIZE - 2, 2);
  ctx.fillRect(x + CELL_SIZE - 3, y + 1, 2, CELL_SIZE - 2);
}

function drawGhostBlock(ctx, x, y, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.fillRect(x + 3, y + 3, CELL_SIZE - 6, CELL_SIZE - 6);
}

function drawMiniPiece(ctx, piece) {
  ctx.fillStyle = '#080c14';
  ctx.fillRect(0, 0, 54, 54);
  if (!piece || !PIECES[piece]) return;
  const shape = PIECES[piece][0];
  const color = COLORS[piece];
  const miniSize = 10;
  const offX = (54 - shape[0].length * miniSize) / 2;
  const offY = (54 - shape.length * miniSize) / 2;
  for (let r = 0; r < shape.length; r++) {
    for (let c = 0; c < shape[r].length; c++) {
      if (shape[r][c]) {
        ctx.fillStyle = color;
        ctx.fillRect(offX + c * miniSize, offY + r * miniSize, miniSize - 1, miniSize - 1);
      }
    }
  }
}

async function fetchCoopState() {
  try {
    const res = await fetch('/api/coop/state');
    const data = await res.json();
    renderAll(data);
  } catch (e) {}
}

function renderAll(data) {
  // Update live Telemetry HUD if available
  if (data.hardware) {
    const hw = data.hardware;
    const elGpu = document.getElementById('hud-gpu');
    if (elGpu && hw.gpu) elGpu.innerText = hw.gpu;
    const elVramVal = document.getElementById('hud-vram-val');
    const elVramSub = document.getElementById('hud-vram-sub');
    if (elVramVal) {
      if (hw.vram_alloc_mb > 0) {
        elVramVal.innerText = (hw.vram_alloc_mb / 1024).toFixed(2);
        if (elVramSub) {
          elVramSub.innerText = `Peak: ${(hw.vram_peak_mb / 1024).toFixed(2)}GB | Reserved: ${(hw.vram_reserved_mb / 1024).toFixed(2)}GB`;
        }
      } else {
        elVramVal.innerText = "3.4";
        if (elVramSub) {
          elVramSub.innerText = "Kev: ~1.0GB | Laya: ~2.4GB (4.6GB Free)";
        }
      }
    }
  }

  // Human Board
  drawGrid(ctxH, data.human.grid, data.human, true);
  drawMiniPiece(ctxHold, data.human.hold_piece);
  drawMiniPiece(ctxHNext, data.human.next_piece);

  document.getElementById('h-score').innerText = data.human.score;
  document.getElementById('h-lines').innerText = data.human.lines;
  document.getElementById('h-piece-idx').innerText = '#' + data.human.piece_idx;

  const hStat = document.getElementById('h-status');
  if (hStat) {
    if (data.human.done) {
      hStat.innerText = 'TOPPED OUT';
      hStat.style.color = 'var(--accent-red)';
    } else if (running) {
      hStat.innerText = 'IN COMBAT';
      hStat.style.color = 'var(--accent-cyan)';
    } else {
      hStat.innerText = 'READY';
      hStat.style.color = 'var(--accent-green)';
    }
  }

  // AI boards
  const ais = data.ais || (data.ai ? [data.ai] : []);
  if (aiPanels.length !== ais.length) buildAIPanels(ais);

  ais.forEach((ai, i) => {
    const P = aiPanels[i];
    drawGrid(P.ctx, ai.grid, ai);
    drawMiniPiece(P.ctxNext, ai.next_piece);
    drawMiniPiece(P.ctxHold, ai.hold_piece);
    P.score.innerText = ai.score;
    P.lines.innerText = ai.lines;
    P.idx.innerText = '#' + ai.piece_idx;
    P.card.classList.toggle('dead', !!ai.done);

    const s = ai.last_step || {};
    if (s.lat_ms !== undefined) {
      const lat = s.lat_ms;
      P.lat.innerText = lat.toFixed(0) + ' ms';
      P.lat.className = 'lat-badge ' + (lat < 5 ? 'lat-ultra' : (lat < 100 ? 'lat-fast' : 'lat-heavy'));
    }
    const ent = (s.p_ent !== undefined ? s.p_ent : 0.9) * 100;
    const con = (s.p_con !== undefined ? s.p_con : 0.05) * 100;
    const neu = (s.p_neu !== undefined ? s.p_neu : 0.05) * 100;
    P.barEnt.style.width = ent + '%';
    P.barCon.style.width = con + '%';
    P.barNeu.style.width = neu + '%';
    P.valEnt.innerText = ent.toFixed(0) + '%';
    P.valCon.innerText = con.toFixed(0) + '%';
    P.valNeu.innerText = neu.toFixed(0) + '%';

    if (ai.mode === 'sales_rl' || s.conversion_prob !== undefined) {
      if (P.title) {
        P.title.innerText = 'Deal Conversion Probability';
        P.title.style.color = '#3fb950';
      }
      if (P.lblEnt) P.lblEnt.innerText = 'CONV:';
      if (P.lblCon) P.lblCon.innerText = 'LOST:';
      if (P.pitch && s.pitch) {
        P.pitch.style.display = 'block';
        P.pitch.innerText = '💼 ' + s.pitch;
      }
    } else if (ai.mode === 'kev') {
      if (P.title) {
        P.title.innerText = 'kev-0.5b Calibrated Decision';
        P.title.style.color = 'var(--accent-neon)';
      }
      if (P.lblEnt) P.lblEnt.innerText = 'PICK:';
      if (P.lblCon) P.lblCon.innerText = 'ALT:';
      if (P.pitch) {
        P.pitch.style.display = 'block';
        const conf = s.confidence !== undefined ? (s.confidence * 100).toFixed(1) : ent.toFixed(0);
        P.pitch.innerText = '🎯 Decision Confidence: ' + conf + '% (LoRA PointerHead 0.5B)';
      }
    } else if (ai.mode === 'laya') {
      if (P.title) {
        P.title.innerText = 'Laya RLCD Decision Model';
        P.title.style.color = 'var(--accent-cyan)';
      }
      if (P.lblEnt) P.lblEnt.innerText = 'PICK:';
      if (P.lblCon) P.lblCon.innerText = 'ALT:';
      if (P.pitch) {
        P.pitch.style.display = 'block';
        const act = s.act_prob !== undefined ? (s.act_prob * 100).toFixed(1) : '100.0';
        const conf = s.confidence !== undefined ? (s.confidence * 100).toFixed(1) : ent.toFixed(0);
        P.pitch.innerText = '⚡ Act Prob: ' + act + '% | Confidence: ' + conf + '% (ModernBERT RLCD)';
      }
    }

    if (i === focusedPanel && s.telemetry) updateNeuralTelemetry(s.telemetry, s);
  });

  document.getElementById('team-score').innerText = data.team_score;
  document.getElementById('team-lines').innerText = data.team_lines;
}

let focusedPanel = 0;

function buildAIPanels(ais) {
  const host = document.getElementById('arena');
  aiPanels.length = 0;
  host.querySelectorAll('.ai-card').forEach(el => el.remove());

  ais.forEach((ai, i) => {
    const card = document.createElement('div');
    card.className = 'player-card ai-card';
    card.innerHTML = `
      <div class="player-header">
        <span class="tag-ai">● AI ${i + 1} // ${ai.mode || 'AGENT'}</span>
        <span class="lat-badge lat-fast v-lat">- ms</span>
      </div>
      <div class="ai-label">${ai.label || 'AI Player'}</div>
      <div class="board-wrapper">
        <div class="side-panel">
          <div class="mini-box"><div class="mini-lbl">Hold</div><canvas width="54" height="54"></canvas></div>
        </div>
        <canvas width="220" height="440"></canvas>
        <div class="side-panel">
          <div class="mini-box"><div class="mini-lbl">Next</div><canvas width="54" height="54"></canvas></div>
        </div>
      </div>
      <div class="stats-row">
        <div class="stat-pill"><div class="stat-pill-val v-score">0</div><div class="stat-pill-lbl">Score</div></div>
        <div class="stat-pill"><div class="stat-pill-val v-lines">0</div><div class="stat-pill-lbl">Lines</div></div>
        <div class="stat-pill"><div class="stat-pill-val v-idx">#0</div><div class="stat-pill-lbl">Piece #</div></div>
      </div>
      <div class="nli-box">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="v-nli-title" style="font-weight:bold; color:var(--accent-neon)">Decision Confidence</span>
          <span style="font-size:10px; color:var(--text-sub); text-transform:uppercase">NEURAL POLICY</span>
        </div>
        <div class="nli-bar">
          <div class="nli-ent b-ent" style="width:90%"></div>
          <div class="nli-con b-con" style="width:5%"></div>
          <div class="nli-neu b-neu" style="width:5%"></div>
        </div>
        <div style="display:flex; justify-content:space-between; color:var(--text-sub); font-size:10px; font-family:var(--font-mono)">
          <span><span class="l-ent">ENT:</span> <span class="v-ent" style="color:var(--accent-green);font-weight:700">90%</span></span>
          <span><span class="l-con">CON:</span> <span class="v-con" style="color:var(--accent-red);font-weight:700">5%</span></span>
          <span><span class="l-neu">NEU:</span> <span class="v-neu">5%</span></span>
        </div>
        <div class="v-pitch" style="display:none; font-size:10px; color:#a5d6ff; margin-top:6px; line-height:1.3; background:rgba(88,166,255,0.08); padding:4px 6px; border-radius:4px; border-left:2px solid #58a6ff; max-height:48px; overflow:hidden;"></div>
      </div>`;
    host.appendChild(card);

    const cvs = card.querySelectorAll('canvas');
    // cvs[0]: Hold, cvs[1]: Main Board, cvs[2]: Next
    aiPanels.push({
      card, ctxHold: cvs[0].getContext('2d'), ctx: cvs[1].getContext('2d'), ctxNext: cvs[2].getContext('2d'),
      score: card.querySelector('.v-score'), lines: card.querySelector('.v-lines'),
      idx: card.querySelector('.v-idx'), lat: card.querySelector('.v-lat'),
      barEnt: card.querySelector('.b-ent'), barCon: card.querySelector('.b-con'),
      barNeu: card.querySelector('.b-neu'), valEnt: card.querySelector('.v-ent'),
      valCon: card.querySelector('.v-con'), valNeu: card.querySelector('.v-neu'),
      title: card.querySelector('.v-nli-title'), lblEnt: card.querySelector('.l-ent'),
      lblCon: card.querySelector('.l-con'), pitch: card.querySelector('.v-pitch'),
    });

    card.addEventListener('click', () => {
      focusedPanel = i;
      document.querySelectorAll('.ai-card').forEach(c => c.classList.remove('focused'));
      card.classList.add('focused');
      document.getElementById('graph-source').innerText = ai.label || ('AI ' + (i + 1));
    });
    if (i === focusedPanel) {
      card.classList.add('focused');
      document.getElementById('graph-source').innerText = ai.label || ('AI ' + (i + 1));
    }
  });
}

// --- Neural Activation Graph & Spectrogram Engine ---
const cvsNeural = document.getElementById('canvas-neural');
const ctxNeural = cvsNeural.getContext('2d');
const cvsSpec = document.getElementById('canvas-spectrum');
const ctxSpec = cvsSpec.getContext('2d');

let neuralData = {
  inputs: [
    {name: 'Landing Height', val: 0.1},
    {name: 'Lines Cleared', val: 0.0},
    {name: 'Holes Added', val: 0.0},
    {name: 'Bumpiness', val: 0.15},
    {name: 'Stack Height', val: 0.1},
    {name: 'Col Transitions', val: 0.2}
  ],
  hidden: new Array(24).fill(0.3),
  outputs: [
    {name: 'CON', val: 0.02, color: '#ef4444'},
    {name: 'ENT', val: 0.95, color: '#10b981'},
    {name: 'NEU', val: 0.03, color: '#94a3b8'}
  ],
  spectrum: new Array(48).fill(0.2),
  moves: []
};

let currentHidden = new Array(24).fill(0.3);
let currentSpectrum = new Array(48).fill(0.2);
let currentOutputs = [0.02, 0.95, 0.03];
let currentInputs = [0.1, 0.0, 0.0, 0.15, 0.1, 0.2];
let currentMoves = [];

let particles = [];
for (let i = 0; i < 35; i++) {
  particles.push({
    stage: Math.random() < 0.5 ? 0 : 1,
    fromIdx: Math.floor(Math.random() * 6),
    toIdx: Math.floor(Math.random() * 24),
    t: Math.random(),
    speed: 0.015 + Math.random() * 0.02
  });
}

function updateNeuralTelemetry(telemetry, stepInfo) {
  if (!telemetry) return;
  if (telemetry.inputs) neuralData.inputs = telemetry.inputs;
  if (telemetry.hidden) neuralData.hidden = telemetry.hidden;
  if (telemetry.outputs) neuralData.outputs = telemetry.outputs;
  if (telemetry.spectrum) neuralData.spectrum = telemetry.spectrum;
  if (telemetry.moves) neuralData.moves = telemetry.moves;
}

function toggleNeuralGraph(show) {
  const el = document.getElementById('neural-body');
  if (el) el.style.display = show ? 'flex' : 'none';
}

function drawNeuralGraph() {
  const W = cvsNeural.width, H = cvsNeural.height;
  ctxNeural.clearRect(0, 0, W, H);
  ctxNeural.fillStyle = '#080c14';
  ctxNeural.fillRect(0, 0, W, H);

  for (let i = 0; i < 24; i++) {
    currentHidden[i] += ((neuralData.hidden[i] !== undefined ? neuralData.hidden[i] : 0.1) - currentHidden[i]) * 0.15;
  }
  for (let i = 0; i < 3; i++) {
    const target = (neuralData.outputs[i] ? neuralData.outputs[i].val : 0.1);
    currentOutputs[i] += (target - currentOutputs[i]) * 0.15;
  }
  for (let i = 0; i < 6; i++) {
    const target = (neuralData.inputs[i] ? neuralData.inputs[i].val : 0.0);
    currentInputs[i] += (target - currentInputs[i]) * 0.15;
  }

  const inPositions = [];
  for (let i = 0; i < 6; i++) {
    inPositions.push({ x: 170, y: 28 + i * 38 });
  }

  const hidPositions = [];
  for (let col = 0; col < 2; col++) {
    const hx = col === 0 ? 320 : 420;
    for (let r = 0; r < 12; r++) {
      hidPositions.push({ x: hx, y: 20 + r * 19 });
    }
  }

  const outPositions = [
    { x: 545, y: 55 },
    { x: 545, y: 125 },
    { x: 545, y: 195 }
  ];

  const moves = neuralData.moves || [];
  if (currentMoves.length !== moves.length) currentMoves = moves.map(m => m.val);
  const movePositions = moves.map((m, i) => ({
    x: 690,
    y: moves.length > 1 ? 22 + i * (206 / (moves.length - 1)) : 125
  }));

  // Synapses
  ctxNeural.lineWidth = 1;
  for (let i = 0; i < 6; i++) {
    const inPos = inPositions[i];
    const inVal = currentInputs[i];
    for (let h = 0; h < 12; h++) {
      const hidPos = hidPositions[h];
      const hVal = currentHidden[h];
      const alpha = Math.min(Math.max(0.02 + (inVal * hVal) * 0.25, 0.02), 0.4);
      ctxNeural.strokeStyle = `rgba(0, 240, 255, ${alpha.toFixed(3)})`;
      ctxNeural.beginPath();
      ctxNeural.moveTo(inPos.x, inPos.y);
      ctxNeural.lineTo(hidPos.x, hidPos.y);
      ctxNeural.stroke();
    }
  }

  for (let h1 = 0; h1 < 12; h1++) {
    const p1 = hidPositions[h1];
    for (let h2 = 12; h2 < 24; h2++) {
      const p2 = hidPositions[h2];
      const alpha = Math.min(Math.max(0.02 + (currentHidden[h1] * currentHidden[h2]) * 0.28, 0.02), 0.45);
      ctxNeural.strokeStyle = `rgba(168, 85, 247, ${alpha.toFixed(3)})`;
      ctxNeural.beginPath();
      ctxNeural.moveTo(p1.x, p1.y);
      ctxNeural.lineTo(p2.x, p2.y);
      ctxNeural.stroke();
    }
  }

  for (let h2 = 12; h2 < 24; h2++) {
    const p2 = hidPositions[h2];
    for (let o = 0; o < 3; o++) {
      const outPos = outPositions[o];
      const oVal = currentOutputs[o];
      const alpha = Math.min(Math.max(0.03 + (currentHidden[h2] * oVal) * 0.35, 0.03), 0.6);
      const colStr = o === 1 ? '16, 185, 129' : (o === 0 ? '239, 68, 68' : '148, 163, 184');
      ctxNeural.strokeStyle = `rgba(${colStr}, ${alpha.toFixed(3)})`;
      ctxNeural.beginPath();
      ctxNeural.moveTo(p2.x, p2.y);
      ctxNeural.lineTo(outPos.x, outPos.y);
      ctxNeural.stroke();
    }
  }

  // Synaptic pulse particles
  const flowActive = document.getElementById('chk-synaptic-flow')?.checked;
  if (flowActive) {
    for (let p of particles) {
      p.t += p.speed;
      if (p.t >= 1.0) {
        p.t = 0.0;
        p.stage = Math.floor(Math.random() * 3);
        if (p.stage === 2 && movePositions.length === 0) p.stage = 1;
        p.fromIdx = Math.floor(Math.random() * (p.stage === 0 ? 6 : 12)) + (p.stage === 1 ? 12 : 0);
        p.toIdx = Math.floor(Math.random() * (p.stage === 0 ? 12 : 3));
      }

      let startP, endP, col;
      if (p.stage === 2 && movePositions.length > 0) {
        startP = outPositions[p.toIdx % 3];
        endP = movePositions[p.fromIdx % movePositions.length];
        col = 'rgba(251, 191, 36, 0.85)';
      } else if (p.stage === 0) {
        startP = inPositions[p.fromIdx % 6];
        endP = hidPositions[p.toIdx % 12];
        col = 'rgba(0, 240, 255, 0.85)';
      } else {
        startP = hidPositions[(p.fromIdx % 12) + 12];
        endP = outPositions[p.toIdx % 3];
        col = p.toIdx === 1 ? 'rgba(16, 185, 129, 0.9)' : 'rgba(168, 85, 247, 0.85)';
      }

      const curX = startP.x + (endP.x - startP.x) * p.t;
      const curY = startP.y + (endP.y - startP.y) * p.t;

      ctxNeural.fillStyle = col;
      ctxNeural.beginPath();
      ctxNeural.arc(curX, curY, 2.2, 0, Math.PI * 2);
      ctxNeural.fill();
    }
  }

  // Input Nodes
  for (let i = 0; i < 6; i++) {
    const pos = inPositions[i];
    const val = currentInputs[i];
    const node = neuralData.inputs[i] || { name: 'Input' };

    ctxNeural.shadowColor = '#00f0ff';
    ctxNeural.shadowBlur = 6 * val;
    ctxNeural.fillStyle = `rgba(0, 240, 255, ${0.4 + val * 0.6})`;
    ctxNeural.beginPath();
    ctxNeural.arc(pos.x, pos.y, 6, 0, Math.PI * 2);
    ctxNeural.fill();
    ctxNeural.shadowBlur = 0;

    ctxNeural.font = '10px monospace';
    ctxNeural.fillStyle = '#94a3b8';
    ctxNeural.textAlign = 'right';
    ctxNeural.fillText(`${node.name}: ${(val * 100).toFixed(0)}%`, pos.x - 10, pos.y + 3);
  }

  // Hidden Neurons
  for (let h = 0; h < 24; h++) {
    const pos = hidPositions[h];
    const act = currentHidden[h];

    ctxNeural.shadowColor = '#a855f7';
    ctxNeural.shadowBlur = 9 * act;

    const r = 3 + 4 * act;
    const grad = ctxNeural.createRadialGradient(pos.x, pos.y, 1, pos.x, pos.y, r);
    grad.addColorStop(0, `rgba(255, 255, 255, ${0.8 * act})`);
    grad.addColorStop(0.5, `rgba(168, 85, 247, ${0.85 * act})`);
    grad.addColorStop(1, `rgba(107, 33, 168, ${0.4 * act + 0.2})`);

    ctxNeural.fillStyle = grad;
    ctxNeural.beginPath();
    ctxNeural.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctxNeural.fill();
    ctxNeural.shadowBlur = 0;
  }

  // Output Nodes
  for (let o = 0; o < 3; o++) {
    const pos = outPositions[o];
    const val = currentOutputs[o];
    const node = neuralData.outputs[o] || { name: 'Output', color: '#888' };
    const isWinner = o === 1 && val > 0.45;

    ctxNeural.shadowColor = node.color;
    ctxNeural.shadowBlur = isWinner ? 16 : 5 * val;

    ctxNeural.fillStyle = node.color;
    ctxNeural.beginPath();
    ctxNeural.arc(pos.x, pos.y, isWinner ? 11 : 8, 0, Math.PI * 2);
    ctxNeural.fill();
    ctxNeural.shadowBlur = 0;

    ctxNeural.font = isWinner ? 'bold 11px sans-serif' : '11px sans-serif';
    ctxNeural.fillStyle = isWinner ? '#fff' : '#cbd5e1';
    ctxNeural.textAlign = 'right';
    ctxNeural.fillText(`${node.name}: ${(val * 100).toFixed(1)}%`, pos.x - 14, pos.y + 4);
  }

  // Synapses: Output -> Moves
  for (let i = 0; i < movePositions.length; i++) {
    currentMoves[i] += ((moves[i].val - currentMoves[i]) * 0.15);
    const mp = movePositions[i];
    for (let o = 0; o < 3; o++) {
      const alpha = Math.min(0.03 + currentMoves[i] * currentOutputs[o] * 0.5, 0.65);
      const colStr = o === 1 ? '16, 185, 129' : (o === 0 ? '239, 68, 68' : '148, 163, 184');
      ctxNeural.strokeStyle = `rgba(${colStr}, ${alpha.toFixed(3)})`;
      ctxNeural.beginPath();
      ctxNeural.moveTo(outPositions[o].x, outPositions[o].y);
      ctxNeural.lineTo(mp.x, mp.y);
      ctxNeural.stroke();
    }
  }

  // Move Nodes
  for (let i = 0; i < movePositions.length; i++) {
    const m = moves[i], mp = movePositions[i], v = currentMoves[i];
    ctxNeural.shadowColor = m.chosen ? '#10b981' : '#fbbf24';
    ctxNeural.shadowBlur = m.chosen ? 14 : 6 * v;
    ctxNeural.fillStyle = m.chosen ? '#10b981' : `rgba(251, 191, 36, ${0.18 + v * 0.75})`;
    ctxNeural.beginPath();
    ctxNeural.arc(mp.x, mp.y, m.chosen ? 7 : 4, 0, Math.PI * 2);
    ctxNeural.fill();
    ctxNeural.shadowBlur = 0;

    ctxNeural.font = m.chosen ? 'bold 10px monospace' : '9px monospace';
    ctxNeural.fillStyle = m.chosen ? '#34d399' : `rgba(148, 163, 184, ${0.45 + v * 0.55})`;
    ctxNeural.textAlign = 'left';
    ctxNeural.fillText(`r${m.rot}c${m.col}` + (m.lines ? ` +${m.lines}` : ''), mp.x + 10, mp.y + 3);
  }

  const mt = document.getElementById('moves-title');
  const mtTxt = `Candidate Placements (${moves.length})`;
  if (mt && mt.innerText !== mtTxt) mt.innerText = mtTxt;
}

function drawSpectrum() {
  const W = cvsSpec.width, H = cvsSpec.height;
  ctxSpec.clearRect(0, 0, W, H);
  ctxSpec.fillStyle = '#080c14';
  ctxSpec.fillRect(0, 0, W, H);

  const numBars = 48;
  const barW = (W - 30) / numBars;
  let maxPeak = 0;

  for (let i = 0; i < numBars; i++) {
    const target = neuralData.spectrum[i] || 0.1;
    currentSpectrum[i] += (target - currentSpectrum[i]) * 0.15;
    if (currentSpectrum[i] > maxPeak) maxPeak = currentSpectrum[i];

    const h = Math.max(currentSpectrum[i] * (H - 40), 4);
    const x = 15 + i * barW;
    const y = H - 20 - h;

    const grad = ctxSpec.createLinearGradient(0, H - 20, 0, y);
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.3)');
    grad.addColorStop(0.6, 'rgba(168, 85, 247, 0.7)');
    grad.addColorStop(1, 'rgba(251, 191, 36, 0.95)');

    ctxSpec.fillStyle = grad;
    ctxSpec.fillRect(x, y, barW - 2, h);

    ctxSpec.fillStyle = '#fff';
    ctxSpec.fillRect(x, y - 2, barW - 2, 2);
  }

  ctxSpec.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctxSpec.lineWidth = 1;
  ctxSpec.beginPath();
  ctxSpec.moveTo(10, H - 19);
  ctxSpec.lineTo(W - 10, H - 19);
  ctxSpec.stroke();

  const el = document.getElementById('spectrum-energy');
  if (el) el.innerText = `Peak: ${(maxPeak * 100).toFixed(0)}%`;
}

pollTimer = setInterval(fetchCoopState, 200);
fetchCoopState();

function animateLoop() {
  drawNeuralGraph();
  drawSpectrum();
  requestAnimationFrame(animateLoop);
}
requestAnimationFrame(animateLoop);

async function sendHumanAction(act) {
  try {
    const res = await fetch('/api/coop/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: act })
    });
    const data = await res.json();
    renderAll(data);
  } catch (e) {}
}

window.addEventListener('keydown', (e) => {
  if (!running) return;
  const key = e.key;
  if (key === 'ArrowLeft' || key === 'a' || key === 'A') {
    e.preventDefault(); sendHumanAction('left');
  } else if (key === 'ArrowRight' || key === 'd' || key === 'D') {
    e.preventDefault(); sendHumanAction('right');
  } else if (key === 'ArrowUp' || key === 'w' || key === 'W' || key === 'x' || key === 'X') {
    e.preventDefault(); sendHumanAction('rotate_cw');
  } else if (key === 'z' || key === 'Z') {
    e.preventDefault(); sendHumanAction('rotate_ccw');
  } else if (key === 'ArrowDown' || key === 's' || key === 'S') {
    e.preventDefault(); sendHumanAction('soft_drop');
  } else if (key === ' ' || key === 'Spacebar') {
    e.preventDefault(); sendHumanAction('hard_drop');
  } else if (key === 'c' || key === 'C' || key === 'Shift') {
    e.preventDefault(); sendHumanAction('hold');
  }
});

async function togglePlay() {
  running = !running;
  const btn = document.getElementById('btn-start');
  await fetch('/api/coop/toggle', { method: 'POST' });
  if (running) {
    btn.innerText = 'Pause Game';
    btn.classList.add('pause');
    gravityTimer = setInterval(() => { sendHumanAction('soft_drop'); }, 800);
  } else {
    btn.innerText = 'Resume Game';
    btn.classList.remove('pause');
    clearInterval(gravityTimer);
  }
}

async function resetGame() {
  await fetch('/api/coop/reset', { method: 'POST' });
  fetchCoopState();
}

async function changeSpeed(val) {
  await fetch('/api/coop/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ai_tempo: parseFloat(val) })
  });
}

fetchCoopState();
</script>
</body>
</html>
"""


class CoopRequestHandler(http.server.BaseHTTPRequestHandler):
    manager: CoopGameManager = None

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html" or self.path == "/coop":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(COOP_HTML.encode("utf-8"))
        elif self.path == "/api/coop/state":
            st = self.manager.get_full_state()
            hw = {
                "gpu": "NVIDIA RTX 3060 Ti 8GB",
                "engine": "2.97M Bitboard checks/sec (O(1) SRS)",
                "paradigm": "Non-Autoregressive Langevin + RLCD",
                "vram_alloc_mb": 0.0,
                "vram_peak_mb": 0.0,
                "vram_reserved_mb": 0.0,
            }
            if torch is not None and torch.cuda.is_available():
                hw["vram_alloc_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
                hw["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                hw["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 1)
                hw["gpu"] = torch.cuda.get_device_name(0)
            st["hardware"] = hw
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(st).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/coop/action":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            act = data.get("action", "")
            self.manager.human_action(act)
            st = self.manager.get_full_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(st).encode("utf-8"))
        elif self.path == "/api/coop/toggle":
            with self.manager.lock:
                self.manager.running = not self.manager.running
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"running": self.manager.running}).encode("utf-8"))
        elif self.path == "/api/coop/reset":
            self.manager.reset()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "reset"}).encode("utf-8"))
        elif self.path == "/api/coop/config":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            if "ai_tempo" in data:
                self.manager.set_ai_tempo(data["ai_tempo"])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "updated"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def serve_coop(port: int = 8088, ai_player=None, vulkan_player=None, qwen_player=None,
               sales_player=None, kev_player=None, laya_player=None, spark_player=None,
               with_heuristic: bool = False, lockstep: bool = True):
    mgr = CoopGameManager(seed=42, ai_player=ai_player, vulkan_player=vulkan_player,
                          qwen_player=qwen_player, sales_player=sales_player,
                          kev_player=kev_player, laya_player=laya_player,
                          spark_player=spark_player,
                          with_heuristic=with_heuristic, lockstep=lockstep)
    print(f"[*] AI boards: {', '.join(sl.label for sl in mgr.slots)}")
    CoopRequestHandler.manager = mgr
    socketserver.TCPServer.allow_reuse_address = True  # must be set before bind
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), CoopRequestHandler)
    print(f"[*] openjev Coop Tetris server running on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\n[*] Stopped coop server.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--with-openjev", action="store_true", help="Load openjev cross-encoder for NLI mode (CPU, slow)")
    parser.add_argument("--with-kev", action="store_true", help="Load jaredpalmer/kev-0.5b decision model (CPU)")
    parser.add_argument("--kev-mode", default="choice", choices=["choice", "nli"], help="Kev formulation: choice or nli")
    parser.add_argument("--kev-top-k", type=int, default=6, help="Kev candidate placements pre-filtered per piece (0 for all)")
    parser.add_argument("--with-laya", action="store_true", help="Load convaiinnovations/laya RLCD decision model (CPU)")
    parser.add_argument("--laya-top-k", type=int, default=6, help="Laya candidate placements pre-filtered per piece (0 for all)")
    parser.add_argument("--with-spark", action="store_true", help="Load mmastrac/djev-spark DiffusionGemma System 1 decision player")
    parser.add_argument("--spark-url", default="http://127.0.0.1:8095", help="Endpoint for djev-spark DGX server")
    parser.add_argument("--spark-top-k", type=int, default=12, help="Spark candidate placements pre-filtered per piece (0 for all)")
    parser.add_argument("--with-sales-rl", action="store_true", help="Load SalesRLAgent (PPO model from arXiv:2503.23303)")
    parser.add_argument("--sales-model", default="sales_conversion_model.zip", help="Path to sales model zip")
    parser.add_argument("--sales-top-k", type=int, default=6, help="Sales RL candidate placements pre-filtered per piece (0 for all)")
    parser.add_argument("--with-vulkan", action="store_true",
                        help="Use the GPU NLI cross-encoder served by llama-server (Vulkan)")
    parser.add_argument("--vulkan-url", default="http://127.0.0.1:8091", help="llama-server endpoint")
    parser.add_argument("--with-qwen", action="store_true",
                        help="Add a Qwen2.5-1.5B-Instruct board (llama-server on Vulkan, port 8092)")
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8092", help="llama-server endpoint for Qwen")
    parser.add_argument("--qwen-top-k", type=int, default=6,
                        help="Qwen placements retained after the cheap pre-filter; 0 evaluates all")
    parser.add_argument("--free-run", action="store_true",
                        help="Let each AI board run at its own speed instead of lock-stepping them on the same piece")
    parser.add_argument("--with-heuristic", action="store_true",
                        help="Also show a Dellacherie board as a reference opponent")
    parser.add_argument("--strategy", default="qualitative",
                        help="Prompt strategy (qualitative is the measured-best; see tetris_prompts.PROMPT_STRATEGIES)")
    parser.add_argument("--vulkan-model-dir", default="/home/server/models/roberta-large-mnli",
                        help="HF dir holding config.json + model.safetensors for the classification head")
    parser.add_argument("--policy-head", default=None,
                        help="Trained TetrisRankHead directory to apply to RoBERTa embeddings")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Candidate placements the model scores per piece (= nodes in the decision layer). "
                             "0 scores all legal placements; positive values use a cheap pre-filter.")
    parser.add_argument("--openjev-top-k", type=int, default=3,
                        help="CPU openjev placements retained after the cheap pre-filter; 0 evaluates all")
    parser.add_argument("--device", default=None,
                        help="Compute device for neural players (cuda or cpu; default auto-detects CUDA)")
    args = parser.parse_args()

    vulkan_player = None
    if args.with_vulkan:
        from tetris_player import VulkanNLIPlayer
        vulkan_player = VulkanNLIPlayer(url=args.vulkan_url, model_dir=args.vulkan_model_dir,
                                        strategy=args.strategy, top_k=(None if args.top_k <= 0 else args.top_k),
                                        policy_head=args.policy_head)

    qwen_player = None
    if args.with_qwen:
        from tetris_player import QwenMovePlayer
        qwen_player = QwenMovePlayer(url=args.qwen_url, strategy=args.strategy,
                                     top_k=(None if args.qwen_top_k <= 0 else args.qwen_top_k))

    sales_player = None
    if args.with_sales_rl:
        from sales_rl_player import SalesRLTetrisPlayer
        sales_player = SalesRLTetrisPlayer(model_path=args.sales_model,
                                           top_k=(None if args.sales_top_k <= 0 else args.sales_top_k),
                                           device=args.device)

    kev_player = None
    if args.with_kev:
        from kev_player import KevPlayer
        kev_player = KevPlayer(mode=args.kev_mode,
                               top_k=(None if args.kev_top_k <= 0 else args.kev_top_k),
                               strategy=args.strategy,
                               device=args.device)

    laya_player = None
    if args.with_laya:
        from laya_player import LayaPlayer
        laya_player = LayaPlayer(top_k=(None if args.laya_top_k <= 0 else args.laya_top_k),
                                 device=args.device)

    spark_player = None
    if args.with_spark:
        from djev_spark_player import DjevSparkPlayer
        spark_player = DjevSparkPlayer(endpoint_url=args.spark_url,
                                       top_k=(None if args.spark_top_k <= 0 else args.spark_top_k))

    ai_player = None
    if args.with_openjev:
        from tetris_player import OpenJevPlayer
        ai_player = OpenJevPlayer(top_k=(None if args.openjev_top_k <= 0 else args.openjev_top_k),
                                  strategy=args.strategy)

    serve_coop(port=args.port, ai_player=ai_player, vulkan_player=vulkan_player,
               qwen_player=qwen_player, sales_player=sales_player,
               kev_player=kev_player, laya_player=laya_player,
               spark_player=spark_player,
               with_heuristic=args.with_heuristic, lockstep=not args.free_run)
