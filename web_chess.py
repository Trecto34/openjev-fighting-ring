"""Cyberpunk esports browser viewer + HTTP API for the AI-vs-AI Chess Arena.

GET  /                     -> cyberpunk browser client
GET  /api/chess/state      -> full arena telemetry (JSON)
POST /api/chess/step       -> advance one AI move
POST /api/chess/toggle     -> play/pause the auto game loop
POST /api/chess/reset      -> restart the match
POST /api/chess/config     -> swap players / depth / tempo

Two AI models (RandomPlayer, HeuristicPlayer, MinimaxPlayer) play each other
in real time, driven by ChessGameManager's background loop.
"""

import argparse
import http.server
import json
import socketserver

from chess_arena import ChessGameManager

CHESS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OPENJEV // CHESS ARENA</title>
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
    --side-white: #e2f2ff;
    --side-black: #c084fc;
    --font-ui: 'Rajdhani', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 14px 18px 40px;
    background: radial-gradient(circle at 50% -10%, #10192d 0%, #060910 72%);
    min-height: 100vh;
    color: var(--text-main);
    font-family: var(--font-ui);
    user-select: none;
  }
  .wrap { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; }

  header.top { display: flex; align-items: center; justify-content: space-between; }
  .title {
    font-size: 26px; font-weight: 800; letter-spacing: 4px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-neon));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 24px rgba(0, 240, 255, 0.25);
  }
  .title .dim { -webkit-text-fill-color: var(--text-sub); color: var(--text-sub); }
  .live {
    display: flex; align-items: center; gap: 8px;
    font-family: var(--font-mono); font-size: 12px; font-weight: 700;
    letter-spacing: 2px; color: var(--accent-red);
    padding: 6px 12px; border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 6px;
    background: rgba(239, 68, 68, 0.08);
  }
  .live .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent-red); box-shadow: 0 0 10px var(--accent-red); }
  .live.on { color: var(--accent-green); border-color: rgba(16, 185, 129, 0.4); background: rgba(16, 185, 129, 0.08); }
  .live.on .dot { background: var(--accent-green); box-shadow: 0 0 10px var(--accent-green); animation: pulse 1s ease-in-out infinite; }

  .hud { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
  .hud-card {
    background: var(--bg-card); backdrop-filter: blur(16px);
    border: 1px solid var(--border-card); border-radius: 10px;
    padding: 9px 14px; position: relative; overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05);
  }
  .hud-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    opacity: 0.6;
  }
  .hud-label { font-size: 10px; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase; color: var(--text-sub); display: flex; align-items: center; gap: 6px; }
  .hud-val { font-family: var(--font-mono); font-size: 15px; font-weight: 700; color: #fff; }
  .hud-val.small { font-size: 12px; color: var(--accent-cyan); }

  .main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 14px; align-items: start; }

  .panel {
    background: var(--bg-card); backdrop-filter: blur(16px);
    border: 1px solid var(--border-card); border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    padding: 12px;
  }

  .board-panel { display: flex; flex-direction: column; align-items: center; gap: 12px; }
  .board-frame {
    position: relative; width: 100%; max-width: 640px; aspect-ratio: 1;
    background: #0a101c; border-radius: 10px;
    border: 1px solid var(--border-card);
    padding: 20px 28px 28px 20px;
    box-shadow: 0 0 40px rgba(0, 240, 255, 0.08), inset 0 0 40px rgba(0, 0, 0, 0.6);
  }
  .board-grid {
    width: 100%; height: 100%;
    display: grid; grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr);
    gap: 1px; background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 4px; overflow: hidden;
    box-shadow: 0 0 24px rgba(0, 0, 0, 0.5);
  }
  .cell {
    display: flex; align-items: center; justify-content: center;
    font-size: clamp(18px, 4.6vmin, 40px); line-height: 1;
    font-family: var(--font-mono);
    cursor: default; position: relative;
  }
  .cell.light { background: #141d31; }
  .cell .piece, .cell span {
    font-size: clamp(22px, 5.2vmin, 44px); line-height: 1;
    display: inline-block; user-select: none;
    transition: transform 0.15s ease, filter 0.15s ease;
  }
  .cell.pw .piece, .cell.pw span {
    color: #00f0ff !important;
    text-shadow: 0 0 12px rgba(0, 240, 255, 0.8), 0 0 2px #ffffff;
    filter: drop-shadow(0 0 6px rgba(0, 240, 255, 0.85));
  }
  .cell.pb .piece, .cell.pb span {
    color: #d946ef !important;
    text-shadow: 0 0 12px rgba(217, 70, 239, 0.9), 0 0 2px #ffffff;
    filter: drop-shadow(0 0 6px rgba(217, 70, 239, 0.85));
  }
  .cell.lastw { box-shadow: inset 0 0 0 2px rgba(0, 240, 255, 0.85), inset 0 0 14px rgba(0, 240, 255, 0.35); }
  .cell.lastb { box-shadow: inset 0 0 0 2px rgba(192, 132, 252, 0.85), inset 0 0 14px rgba(168, 85, 247, 0.35); }
  .cell.check { animation: checkPulse 1.1s ease-in-out infinite; }
  @keyframes checkPulse {
    0%, 100% { box-shadow: inset 0 0 0 3px rgba(239, 68, 68, 0.9), inset 0 0 20px rgba(239, 68, 68, 0.45); }
    50%      { box-shadow: inset 0 0 0 3px rgba(239, 68, 68, 0.35), inset 0 0 8px rgba(239, 68, 68, 0.2); }
  }
  .coord-file, .coord-rank { position: absolute; font-family: var(--font-mono); font-size: 10px; color: var(--text-sub); z-index: 2; }
  .coord-file { bottom: 6px; }
  .coord-rank { right: 6px; }
  .file-a { left: 3.5%; } .file-b { left: 15.1%; } .file-c { left: 26.7%; } .file-d { left: 38.3%; }
  .file-e { left: 49.9%; } .file-f { left: 61.5%; } .file-g { left: 73.1%; } .file-h { left: 84.7%; }
  .rank-1 { top: 88.5%; } .rank-2 { top: 75.9%; } .rank-3 { top: 63.3%; } .rank-4 { top: 50.7%; }
  .rank-5 { top: 38.1%; } .rank-6 { top: 25.5%; } .rank-7 { top: 12.9%; } .rank-8 { top: 0.3%; }

  .evalbar {
    width: 100%; max-width: 640px; height: 14px; border-radius: 8px; overflow: hidden;
    display: flex; border: 1px solid var(--border-card); background: #0a101c; position: relative;
  }
  .evalbar #fill-w {
    height: 100%; background: linear-gradient(90deg, #0a4a55, var(--accent-cyan));
    transition: width 0.35s ease; box-shadow: 0 0 12px rgba(0, 240, 255, 0.5);
  }
  .evalbar #fill-b { flex: 1; background: linear-gradient(90deg, var(--accent-neon), #3b1d5e); transition: flex 0.35s ease; box-shadow: 0 0 12px rgba(168, 85, 247, 0.5); }
  .evalbar .mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: rgba(248, 250, 252, 0.5); }
  .eval-label { font-family: var(--font-mono); font-size: 11px; color: var(--accent-gold); margin-top: 2px; }

  .side { display: flex; flex-direction: column; gap: 12px; }
  .pcard {
    border-radius: 10px; padding: 12px 14px; border: 1px solid var(--border-card);
    background: var(--bg-card); position: relative; overflow: hidden;
  }
  .pcard .toprow { display: flex; justify-content: space-between; align-items: center; }
  .pcard h3 { margin: 0; font-size: 17px; letter-spacing: 2px; font-weight: 800; }
  .pcard.w h3 { color: var(--accent-cyan); text-shadow: 0 0 12px rgba(0, 240, 255, 0.4); }
  .pcard.b h3 { color: var(--accent-neon); text-shadow: 0 0 12px rgba(168, 85, 247, 0.4); }
  .pcard .chip { font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: var(--text-sub); letter-spacing: 1px; }
  .status { display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-mono); font-size: 11px; font-weight: 700; letter-spacing: 1px; color: var(--text-sub); }
  .status .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-sub); }
  .status.think { color: var(--accent-gold); }
  .status.think .dot { background: var(--accent-gold); box-shadow: 0 0 8px var(--accent-gold); animation: pulse 0.7s ease-in-out infinite; }
  .status.win { color: var(--accent-green); }
  .status.win .dot { background: var(--accent-green); box-shadow: 0 0 8px var(--accent-green); }
  .status.lose { color: var(--accent-red); }
  .status.lose .dot { background: var(--accent-red); }
  .pmeter { height: 4px; border-radius: 2px; background: #0a101c; margin: 8px 0 6px; overflow: hidden; }
  .pmeter > div { height: 100%; width: 0%; transition: width 0.3s ease; }
  .pcard.w .pmeter > div { background: var(--accent-cyan); }
  .pcard.b .pmeter > div { background: var(--accent-neon); }
  .pstats { font-family: var(--font-mono); font-size: 11px; color: var(--text-sub); display: grid; grid-template-columns: 1fr 1fr; gap: 2px 8px; margin-top: 4px; }
  .pstats b { color: #fff; font-weight: 700; }
  .captured { margin-top: 6px; font-family: var(--font-mono); font-size: 13px; min-height: 18px; color: var(--text-sub); }

  .moves-panel { padding: 10px 14px; }
  .section-label { font-size: 10px; font-weight: 700; letter-spacing: 2px; color: var(--accent-cyan); text-transform: uppercase; margin-bottom: 6px; }
  .move-strip {
    display: flex; flex-wrap: wrap; gap: 4px; max-height: 96px; overflow-y: auto;
    font-family: var(--font-mono); font-size: 12px; scrollbar-width: thin;
  }
  .move-row { display: flex; gap: 4px; align-items: center; }
  .mnum { color: var(--text-sub); }
  .mrec { padding: 2px 7px; border-radius: 4px; font-weight: 700; cursor: default; }
  .mrec.w { color: var(--accent-cyan); background: rgba(0, 240, 255, 0.08); border: 1px solid rgba(0, 240, 255, 0.2); }
  .mrec.b { color: var(--accent-neon); background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.2); }
  .mrec:last-child { animation: glowIn 0.4s ease; }
  @keyframes glowIn { from { transform: scale(1.15); opacity: 0.4; } to { transform: scale(1); opacity: 1; } }

  .pgn { margin-top: 8px; }
  .pgn textarea {
    width: 100%; height: 74px; resize: none; border-radius: 6px;
    background: #081020; color: #dbeafe; border: 1px solid var(--border-card);
    font-family: var(--font-mono); font-size: 11px; line-height: 1.5; padding: 8px;
  }

  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end; }
  .btn {
    font-family: var(--font-ui); font-weight: 800; letter-spacing: 2px;
    font-size: 14px; padding: 10px 20px; border-radius: 8px; cursor: pointer;
    color: #04121a; background: linear-gradient(180deg, var(--accent-cyan), #0891b2);
    border: none; box-shadow: 0 0 18px rgba(0, 240, 255, 0.35);
    transition: transform 0.1s ease, box-shadow 0.2s ease;
  }
  .btn:hover { transform: translateY(-1px); box-shadow: 0 0 26px rgba(0, 240, 255, 0.55); }
  .btn:active { transform: translateY(1px); }
  .btn.pause { background: linear-gradient(180deg, var(--accent-gold), #b45309); color: #141000; box-shadow: 0 0 18px rgba(251, 191, 36, 0.35); }
  .btn.ghost { background: rgba(56, 189, 248, 0.08); color: var(--accent-cyan); border: 1px solid rgba(56, 189, 248, 0.35); box-shadow: none; }
  .btn.ghost:hover { box-shadow: 0 0 16px rgba(0, 240, 255, 0.25); }
  .cfg-group { display: flex; flex-direction: column; gap: 4px; }
  .cfg-group label { font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--text-sub); font-weight: 700; }
  .cfg-group select, .cfg-group input[type="number"] {
    font-family: var(--font-mono); font-size: 12px; font-weight: 700;
    background: #081020; color: #fff; border: 1px solid var(--border-card);
    border-radius: 6px; padding: 8px; outline: none;
  }
  .cfg-group input[type="range"] { accent-color: var(--accent-cyan); width: 150px; }
  .tempo-val { color: var(--accent-gold); font-size: 12px; }

  .result-overlay {
    position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
    background: rgba(4, 8, 16, 0.82); backdrop-filter: blur(8px); z-index: 50;
  }
  .result-overlay.show { display: flex; }
  .result-card {
    text-align: center; padding: 34px 60px; border-radius: 16px;
    border: 1px solid var(--accent-gold); background: rgba(13, 19, 32, 0.95);
    box-shadow: 0 0 60px rgba(251, 191, 36, 0.3);
    animation: popIn 0.35s ease;
  }
  .result-card h2 { margin: 0 0 8px; font-size: 34px; letter-spacing: 5px; color: var(--accent-gold); text-shadow: 0 0 20px rgba(251, 191, 36, 0.6); }
  .result-card p { margin: 0; font-family: var(--font-mono); font-size: 13px; color: var(--text-sub); }
  @keyframes popIn { from { transform: scale(0.7); opacity: 0; } to { transform: scale(1); opacity: 1; } }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

  @media (max-width: 860px) {
    .main { grid-template-columns: 1fr; }
    .side { flex-direction: row; flex-wrap: wrap; }
    .pcard { flex: 1 1 240px; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="title">OPENJEV <span class="dim">// CHESS ARENA</span></div>
    <div class="live" id="live"><span class="dot"></span><span id="live-txt">OFFLINE</span></div>
  </header>

  <div class="hud">
    <div class="hud-card"><div class="hud-label">MATCHUP</div><div class="hud-val" id="hud-match">-- vs --</div></div>
    <div class="hud-card"><div class="hud-label">SEARCH</div><div class="hud-val" id="hud-depth">--</div></div>
    <div class="hud-card"><div class="hud-label">TEMPO</div><div class="hud-val" id="hud-tempo">--</div></div>
    <div class="hud-card"><div class="hud-label">EVALUATION</div><div class="hud-val" id="hud-eval">--</div></div>
    <div class="hud-card"><div class="hud-label">MOVES</div><div class="hud-val" id="hud-moves">0</div></div>
    <div class="hud-card"><div class="hud-label">ENGINE</div><div class="hud-val small" id="hud-engine">--</div></div>
  </div>

  <div class="main">
    <section class="panel board-panel">
      <div class="board-frame">
        <div class="board-grid" id="board"></div>
        <span class="coord-file file-a">a</span><span class="coord-file file-b">b</span>
        <span class="coord-file file-c">c</span><span class="coord-file file-d">d</span>
        <span class="coord-file file-e">e</span><span class="coord-file file-f">f</span>
        <span class="coord-file file-g">g</span><span class="coord-file file-h">h</span>
        <span class="coord-rank rank-1">1</span><span class="coord-rank rank-2">2</span>
        <span class="coord-rank rank-3">3</span><span class="coord-rank rank-4">4</span>
        <span class="coord-rank rank-5">5</span><span class="coord-rank rank-6">6</span>
        <span class="coord-rank rank-7">7</span><span class="coord-rank rank-8">8</span>
      </div>
      <div class="evalbar"><div id="fill-w"></div><div id="fill-b"></div><div class="mid"></div></div>
      <div class="eval-label" id="eval-label">--</div>
    </section>

    <aside class="side">
      <div class="pcard w">
        <div class="toprow"><h3>WHITE</h3><span class="status" id="st-w"><span class="dot"></span><span id="st-w-txt">READY</span></span></div>
        <div class="chip" id="chip-w">--</div>
        <div class="pmeter"><div id="pm-w"></div></div>
        <div class="pstats">
          <span>THINK</span><b id="ps-w-ms">0ms</b>
          <span>NODES</span><b id="ps-w-n">0</b>
          <span>LAST MOVE</span><b id="ps-w-mv">--</b>
          <span>STATUS</span><b id="ps-w-st">--</b>
        </div>
        <div class="captured" id="cap-w"></div>
      </div>
      <div class="pcard b">
        <div class="toprow"><h3>BLACK</h3><span class="status" id="st-b"><span class="dot"></span><span id="st-b-txt">READY</span></span></div>
        <div class="chip" id="chip-b">--</div>
        <div class="pmeter"><div id="pm-b"></div></div>
        <div class="pstats">
          <span>THINK</span><b id="ps-b-ms">0ms</b>
          <span>NODES</span><b id="ps-b-n">0</b>
          <span>LAST MOVE</span><b id="ps-b-mv">--</b>
          <span>STATUS</span><b id="ps-b-st">--</b>
        </div>
        <div class="captured" id="cap-b"></div>
      </div>
      <div class="panel moves-panel">
        <div class="section-label">MATCH TAPE</div>
        <div class="move-strip" id="moves"></div>
        <div class="pgn"><div class="section-label">PGN</div><textarea id="pgn" readonly spellcheck="false"></textarea></div>
      </div>
    </aside>
  </div>

  <section class="controls panel">
    <button class="btn" id="btn-run">RUN</button>
    <button class="btn ghost" id="btn-step">STEP</button>
    <button class="btn ghost" id="btn-reset">RESET</button>
    <div class="cfg-group">
      <label>White AI</label>
      <select id="cfg-white">
        <option value="minimax">Minimax // αβ</option>
        <option value="heuristic">Heuristic // static</option>
        <option value="random">Random // chaos</option>
      </select>
    </div>
    <div class="cfg-group">
      <label>Black AI</label>
      <select id="cfg-black">
        <option value="heuristic">Heuristic // static</option>
        <option value="minimax">Minimax // αβ</option>
        <option value="random">Random // chaos</option>
      </select>
    </div>
    <div class="cfg-group">
      <label>Depth</label>
      <input type="number" id="cfg-depth" min="1" max="5" value="3" step="1">
    </div>
    <div class="cfg-group">
      <label>Tempo <span class="tempo-val" id="cfg-tempo-val">0.25s</span></label>
      <input type="range" id="cfg-tempo" min="0.05" max="2.0" step="0.05" value="0.25">
    </div>
    <button class="btn ghost" id="btn-config">DEPLOY CONFIG</button>
  </section>
</div>

<div class="result-overlay" id="overlay" onclick="if(event.target===this) dismissOverlay();">
  <div class="result-card">
    <h2 id="ov-title">CHECKMATE</h2>
    <p id="ov-sub">--</p>
    <div style="display:flex; gap:12px; justify-content:center; margin-top:20px;">
      <button class="btn" style="background:var(--accent-cyan); color:#000;" onclick="doPost('/api/chess/reset').then(() => { overlayDismissed=false; fetchState(); });">PLAY AGAIN</button>
      <button class="btn ghost" onclick="dismissOverlay();">REVIEW BOARD</button>
    </div>
  </div>
</div>

<script>
const GLYPH = {
  P:'♙', N:'♘', B:'♗', R:'♖', Q:'♕', K:'♔',
  p:'♟', n:'♞', b:'♝', r:'♜', q:'♛', k:'♚'
};
let running = false;
let renderedPlies = -1;
let overlayDismissed = false;
const cells = [];

function dismissOverlay() {
  overlayDismissed = true;
  document.getElementById('overlay').classList.remove('show');
}

function buildBoard() {
  const grid = document.getElementById('board');
  grid.innerHTML = '';
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell ' + (((r + c) % 2 === 0) ? 'light' : 'dark');
      cell.dataset.r = r; cell.dataset.c = c;
      const p = document.createElement('span');
      p.className = 'piece';
      cell.appendChild(p);
      grid.appendChild(cell);
      cells.push(cell);
    }
  }
}

function sqToRC(name) {
  if (!name || name.length < 2) return null;
  const f = name.charCodeAt(0) - 97;
  const r = 8 - parseInt(name[1], 10);
  return { r, c: f };
}

function render(state) {
  // board pieces
  const sq = state.squares;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = cells[r * 8 + c];
      const sym = sq[r][c];
      const s = cell.querySelector('span');
      if (sym && sym !== ' ') {
        s.textContent = GLYPH[sym] || sym;
        cell.classList.remove('pw', 'pb');
        cell.classList.add(sym === sym.toUpperCase() ? 'pw' : 'pb');
      } else {
        s.textContent = '';
        cell.classList.remove('pw', 'pb');
      }
      cell.classList.remove('lastw', 'lastb', 'check');
    }
  }

  const lm = state.last_move;
  if (lm && lm.from && lm.to) {
    const a = sqToRC(lm.from), b = sqToRC(lm.to);
    const cls = lm.color === 'w' ? 'lastw' : 'lastb';
    if (a) cells[a.r * 8 + a.c].classList.add(cls);
    if (b) cells[b.r * 8 + b.c].classList.add(cls);
  }
  if (state.check_square) {
    const k = sqToRC(state.check_square);
    if (k) cells[k.r * 8 + k.c].classList.add('check');
  }

  // hud
  const W = (state.players.w.label || '').split(' ')[0];
  const B = (state.players.b.label || '').split(' ')[0];
  document.getElementById('hud-match').innerText = W + ' vs ' + B;
  document.getElementById('hud-depth').innerText = 'd' + state.config.depth +
    ' // ' + state.config.white + ' / ' + state.config.black;
  document.getElementById('hud-tempo').innerText = state.config.tempo.toFixed(2) + 's/move';
  document.getElementById('hud-moves').innerText = state.plies;
  document.getElementById('hud-engine').innerText = state.engine;

  const cp = state.evaluation;
  let pct = 50 + (cp / 100) * 8;
  if (state.evaluation_mate) pct = cp > 0 ? 100 : 0;
  pct = Math.max(2, Math.min(98, pct));
  document.getElementById('fill-w').style.width = pct + '%';
  document.getElementById('hud-eval').innerText = (state.evaluation_mate
    ? (cp > 0 ? 'MATE' : '-MATE') : (cp / 100).toFixed(2)) + ' pawns';

  // side cards
  updateCard('w', state.players.w, state);
  updateCard('b', state.players.b, state);

  // move tape (only rebuild when a new ply arrives)
  const acts = state.moves || state.history;
  const plies = (acts || []).length;
  if (plies !== renderedPlies) {
    renderedPlies = plies;
    const strip = document.getElementById('moves');
    strip.innerHTML = '';
    let html = '';
    for (let i = 0; i < plies; i++) {
      const e = acts[i];
      if (i % 2 === 0) html += '<span class="mnum">' + (i / 2 + 1) + '.</span>' + '';
      html += '<span class="mrec ' + e.color + '">' + e.san + '</span>';
    }
    strip.innerHTML = html;
    strip.scrollTop = strip.scrollHeight;
  }

  document.getElementById('pgn').value = state.pgn || '';
  document.getElementById('eval-label').innerText =
    state.turn.toUpperCase() + ' TO MOVE // ' + state.fen;

  // live / status / overlay
  const live = document.getElementById('live');
  live.classList.toggle('on', !!state.running);
  document.getElementById('live-txt').innerText = state.running ? 'LIVE' : 'PAUSED';

  if (state.result && state.game_over) {
    document.getElementById('ov-title').innerText =
      state.result.winner_name ? (state.result.winner_name + ' VICTORY') : 'DRAW';
    document.getElementById('ov-sub').innerText = state.result.label;
    if (!overlayDismissed) {
      document.getElementById('overlay').classList.add('show');
    }
  } else {
    overlayDismissed = false;
    document.getElementById('overlay').classList.remove('show');
  }
}

function updateCard(key, p, state) {
  const side = key === 'w';
  const self = (state.turn === key);
  const stEl = document.getElementById('st-' + key);
  const stTxt = document.getElementById('st-' + key + '-txt');
  stEl.classList.remove('think', 'win', 'lose');
  if (state.game_over) {
    const winner = state.result && state.result.winner;
    if (winner === key) { stEl.classList.add('win'); stTxt.innerText = 'VICTORY'; }
    else if (winner) { stEl.classList.add('lose'); stTxt.innerText = 'DEFEAT'; }
    else { stTxt.innerText = 'DRAW'; }
  } else if (state.running && self) {
    stEl.classList.add('think'); stTxt.innerText = 'THINKING';
  } else if (self) {
    stTxt.innerText = 'TO MOVE';
  } else {
    stTxt.innerText = 'READY';
  }

  document.getElementById('chip-' + key).innerText = p.label || p.type;
  document.getElementById('ps-' + key + '-ms').innerText = Math.round(p.thinking_ms) + 'ms cum';
  document.getElementById('ps-' + key + '-n').innerText = p.nodes.toLocaleString();
  document.getElementById('pm-' + key).style.width =
    Math.min(100, 8 * Math.log10(1 + p.thinking_ms)) + '%';

  const lastMoveBySide = {};
  (state.history || []).forEach((e) => { if (!lastMoveBySide[e.color]) lastMoveBySide[e.color] = e; });
  const last = lastMoveBySide[key];
  document.getElementById('ps-' + key + '-mv').innerText = last ? last.san : '--';
  document.getElementById('ps-' + key + '-st').innerText =
    state.legal_move_count + ' legal';

  const cap = (state.captured && state.captured[key]) || [];
  document.getElementById('cap-' + key).innerText = cap.length
    ? 'CAPTURED ' + cap.join('') : '';
}

async function fetchState() {
  try {
    const res = await fetch('/api/chess/state');
    const data = await res.json();
    render(data);
  } catch (e) {}
}

async function doPost(path, body) {
  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined
    });
    return await res.json();
  } catch (e) { return null; }
}

function setupButtons() {
  const runBtn = document.getElementById('btn-run');
  runBtn.addEventListener('click', async () => {
    await doPost('/api/chess/toggle');
    running = !running;
    runBtn.innerText = running ? 'PAUSE' : 'RUN';
    runBtn.classList.toggle('pause', running);
    fetchState();
  });
  document.getElementById('btn-step').addEventListener('click', () => doPost('/api/chess/step').then(fetchState));
  document.getElementById('btn-reset').addEventListener('click', () => {
    overlayDismissed = false;
    doPost('/api/chess/reset').then(fetchState);
  });

  const tempo = document.getElementById('cfg-tempo');
  tempo.addEventListener('input', () => {
    document.getElementById('cfg-tempo-val').innerText = parseFloat(tempo.value).toFixed(2) + 's';
  });
  document.getElementById('btn-config').addEventListener('click', () => {
    doPost('/api/chess/config', {
      white: document.getElementById('cfg-white').value,
      black: document.getElementById('cfg-black').value,
      depth: parseInt(document.getElementById('cfg-depth').value, 10),
      tempo: parseFloat(tempo.value)
    }).then(d => { if (d) { renderedPlies = -1; fetchState(); } });
  });
}

buildBoard();
setupButtons();
fetchState();
setInterval(fetchState, 250);
</script>
</body>
</html>
"""


class ChessRequestHandler(http.server.BaseHTTPRequestHandler):
    manager: ChessGameManager = None

    def _send_json(self, obj):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/arena"):
            payload = CHESS_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif path == "/api/chess/state":
            self._send_json(self.manager.get_state())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        data = json.loads(body.decode("utf-8")) if body else {}

        if path == "/api/chess/step":
            self.manager.step()
            self._send_json(self.manager.get_state())
        elif path == "/api/chess/toggle":
            running = self.manager.toggle()
            self._send_json({"running": running})
        elif path == "/api/chess/reset":
            self.manager.reset()
            self._send_json(self.manager.get_state())
        elif path == "/api/chess/config":
            self.manager.set_config(data)
            self._send_json(self.manager.get_state())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def serve_chess(port: int = 8089, white: str = "minimax", black: str = "heuristic",
                depth: int = 3, tempo: float = 0.25, seed=None):
    manager = ChessGameManager(white_type=white, black_type=black,
                               depth=depth, tempo=tempo, seed=seed)
    # Start paused at initial board state. User can click RUN or STEP to begin.
    ChessRequestHandler.manager = manager
    socketserver.TCPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), ChessRequestHandler)
    print(f"[*] AI-vs-AI Chess Arena: {white} (white) vs {black} (black), depth {depth}, tempo {tempo}s")
    print(f"[*] Serving on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopped chess arena server.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="AI-vs-AI Chess Arena web viewer")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--white", default="minimax", choices=["random", "heuristic", "minimax"])
    parser.add_argument("--black", default="heuristic", choices=["random", "heuristic", "minimax"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--tempo", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    serve_chess(port=args.port, white=args.white, black=args.black,
                depth=args.depth, tempo=args.tempo, seed=args.seed)


if __name__ == "__main__":
    main()