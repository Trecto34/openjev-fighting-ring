"""The Chess Instrument — minimalist monochromatic web viewer + HTTP API.

GET  /                     -> minimalist browser client
GET  /api/chess/state      -> full arena telemetry (JSON)
POST /api/chess/step       -> advance one AI move
POST /api/chess/toggle     -> play/pause the auto game loop
POST /api/chess/reset      -> restart the match
POST /api/chess/config     -> swap players / depth / tempo

Registered player kinds from PLAYER_REGISTRY (random / heuristic / minimax, and
later slow external models) play each other in real time, driven by
ChessGameManager's background worker. The server boots paused at Move 0.
"""

import argparse
import http.server
import json
import socketserver

from chess_arena import PLAYER_TYPES, ChessGameManager

CHESS_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OPENJEV — THE CHESS INSTRUMENT</title>
<style>
  :root {
    --ink-0: #0B0C0E;
    --ink-1: #121418;
    --sq-dark: #11141A;
    --sq-light: #1A1E26;
    --hairline: rgba(255, 255, 255, 0.08);
    --hairline-2: rgba(255, 255, 255, 0.14);
    --hairline-cell: rgba(255, 255, 255, 0.04);
    --paper: #F4F5F7;
    --silver: #A0A6B2;
    --slate: #6E747D;
    --coord: #4A505A;
    --crimson: #E5484D;
    --gunmetal: #767D8A;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: "Geist Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html { background: var(--ink-0); }
  body {
    margin: 0; padding: 28px 30px 72px;
    min-height: 100vh;
    background-color: var(--ink-0);
    background-image:
      repeating-linear-gradient(90deg, rgba(244,245,247,0.02) 0 1px, transparent 1px 80px),
      repeating-linear-gradient(0deg, rgba(244,245,247,0.02) 0 1px, transparent 1px 80px);
    color: var(--paper);
    font-family: var(--font-sans);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }

  /* ------------------------------------------------------------- masthead */
  .masthead {
    display: flex; justify-content: space-between; align-items: flex-end;
    padding-bottom: 20px; border-bottom: 1px solid var(--hairline);
    margin-bottom: 26px;
  }
  .title {
    margin: 0; font-size: 30px; font-weight: 650; letter-spacing: -0.015em;
    background: linear-gradient(180deg, #FFFFFF 0%, #C9CDD6 55%, #8B909B 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
  }
  .title em { font-style: normal; letter-spacing: -0.02em;
    background: linear-gradient(180deg, #9BA0AA 0%, #5C626D 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
  .subtitle { margin: 7px 0 0; font-size: 12px; letter-spacing: 0.03em;
    color: var(--slate); font-family: var(--font-mono); }
  .sys {
    display: flex; align-items: center; gap: 9px;
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--silver);
  }
  .sys .led { width: 6px; height: 6px; border-radius: 50%; background: var(--slate); }
  .sys.live .led { background: var(--paper); animation: bk 1.6s ease-in-out infinite; }
  @keyframes bk { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

  /* ------------------------------------------------------------ telemetry */
  .sec-label {
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--silver); margin-bottom: 12px;
  }
  .sec-label b { color: var(--paper); font-weight: 500; }
  .tel { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1px; background: var(--hairline); border: 1px solid var(--hairline);
    margin-bottom: 26px; }
  .tel-cell { background: var(--ink-1); padding: 11px 15px; }
  .tel-label { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--slate); margin-bottom: 7px; }
  .tel-val { font-family: var(--font-mono); font-size: 17px; font-weight: 500; color: var(--paper);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .tel-val dim { color: var(--slate); font-size: 12px; }

  /* ----------------------------------------------------------------- grid */
  .main { display: grid; grid-template-columns: minmax(0, 1fr) 356px; gap: 28px; align-items: start; }

  .panel { background: var(--ink-1); border: 1px solid var(--hairline); padding: 18px; }

  /* ---------------------------------------------------------------- board */
  .board-panel { padding: 20px; }
  .board {
    display: grid; grid-template-columns: repeat(8, 1fr); aspect-ratio: 1;
    gap: 1px; background: var(--hairline-cell);
    border: 1px solid rgba(255, 255, 255, 0.04);
  }
  .cell { position: relative; display: flex; align-items: center; justify-content: center; }
  .cell.light { background: var(--sq-light); }
  .cell.dark { background: var(--sq-dark); }
  .piece { font-family: var(--font-mono); font-size: clamp(22px, 5.2vmin, 44px); line-height: 1; }
  .cell.pw .piece { color: var(--paper); text-shadow: 0 1px 0 rgba(0,0,0,0.45), 0 0 1px rgba(0,0,0,0.35); }
  .cell.pb .piece { color: var(--gunmetal, #767D8A); text-shadow: 0 1px 0 rgba(0,0,0,0.5); }
  .coord { position: absolute; font-family: var(--font-mono); font-size: 9px; line-height: 1;
    color: var(--coord); pointer-events: none; }
  .coord-file { right: 4px; bottom: 4px; }
  .coord-rank { left: 4px; top: 4px; }
  .cell.last { box-shadow: inset 0 0 0 1px rgba(244, 245, 247, 0.35); }
  .cell.check { box-shadow: inset 0 0 0 1px var(--crimson); }

  /* controls ------------------------------------------------------------- */
  .controls { display: flex; flex-wrap: wrap; gap: 28px; margin-top: 22px; }
  .btn {
    font-family: var(--font-sans); font-size: 12px; font-weight: 500;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 10px 20px; border-radius: 3px; cursor: pointer; line-height: 1;
    transition: border-color 0.15s ease, background-color 0.15s ease, color 0.15s ease;
  }
  .btn-run { background: var(--paper); color: var(--ink-0); border: 1px solid var(--paper); }
  .btn-run:hover { background: #DDE0E6; }
  .btn-ghost {
    background: rgba(255, 255, 255, 0.02); color: #C5CBD5;
    border: 1px solid rgba(255, 255, 255, 0.14);
  }
  .btn-ghost:hover { border-color: rgba(255, 255, 255, 0.3); color: var(--paper); background: rgba(255,255,255,0.05); }
  .config { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; margin-left: auto; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field label { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--slate); }
  .field select, .field input[type="number"] {
    background: var(--ink-0); color: var(--paper); border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 3px; font-family: var(--font-mono); font-size: 12px; padding: 8px 10px; outline: none;
  }
  .field select:focus, .field input:focus { border-color: rgba(255, 255, 255, 0.35); }
  .field input[type="range"] { accent-color: var(--paper); width: 130px; margin: 7px 0; }
  .tempo-val { color: var(--paper); font-size: 11px; }

  /* ------------------------------------------------------------------ duel */
  .duel-card { border: 1px solid var(--hairline); background: var(--ink-0); padding: 15px 16px; }
  .duel-card + .duel-card { margin-top: 10px; }
  .duel-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }
  .duel-name { font-size: 13px; letter-spacing: 0.16em; text-transform: uppercase;
    font-weight: 600; color: var(--silver); }
  .duel-name.gn { color: var(--paper); }
  .duel-status { font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--slate); text-align: right; }
  .duel-status.think { color: var(--paper); }
  .duel-status.win { color: var(--paper); }
  .duel-status.lose { color: var(--slate); }
  .duel-meter { height: 2px; background: rgba(255, 255, 255, 0.06); margin: 0 0 12px; }
  .duel-meter > div { height: 100%; width: 0%; transition: width 0.4s ease; }
  .duel-meter.w > div { background: var(--paper); }
  .duel-meter.b > div { background: var(--gunmetal, #767D8A); }
  .duel-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 18px;
    font-family: var(--font-mono); font-size: 11px; color: var(--slate); }
  .duel-stats span { min-width: 0; }
  .duel-stats label { display: block; font-size: 9px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--slate); margin-bottom: 4px; }
  .duel-stats b { display: block; color: var(--paper); font-weight: 500; font-size: 14px;
    font-family: var(--font-mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .captured { margin-top: 11px; padding-top: 9px; border-top: 1px solid var(--hairline);
    font-family: var(--font-mono); font-size: 14px; color: var(--slate); min-height: 20px; }
  .captured label { font-size: 9px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--slate); margin-right: 8px; }

  /* gauge ---------------------------------------------------------------- */
  .gauge { margin-top: 14px; }
  .gauge-ends { display: flex; justify-content: space-between;
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--slate); margin-bottom: 9px; }
  .gauge-track { position: relative; height: 6px; background: var(--gunmetal, #767D8A); }
  .gauge-white { position: absolute; left: 0; top: 0; bottom: 0; width: 50%;
    background: var(--paper); transition: width 0.55s cubic-bezier(0.3, 0.7, 0.2, 1); }
  .gauge-zero { position: absolute; left: 50%; top: -2px; bottom: -2px; width: 1px;
    background: rgba(244, 245, 247, 0.6); }
  .gauge-val { text-align: center; font-family: var(--font-mono); font-size: 13px;
    color: var(--paper); margin-top: 10px; }
  .gauge-val small { color: var(--slate); font-size: 10px; }

  /* tape + fen ----------------------------------------------------------- */
  .tape { max-height: 148px; overflow-y: auto; scrollbar-width: thin;
    font-family: var(--font-mono); font-size: 12px; color: var(--silver); line-height: 1.85; }
  .tape .num { color: var(--slate); margin-right: 2px; }
  .tape .mv { padding: 1px 5px; margin: 0 1px; cursor: default; }
  .tape .mv.w { color: var(--paper); }
  .tape .mv.b { color: var(--silver); }
  .tape .mv.latest { box-shadow: inset 0 -1px 0 rgba(244, 245, 247, 0.4); }
  .fen { margin-top: 14px; padding-top: 11px; border-top: 1px solid var(--hairline);
    font-family: var(--font-mono); font-size: 10px; color: var(--slate); line-height: 1.65;
    word-break: break-all; }
  .fen label { display: block; font-size: 9px; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--slate); margin-bottom: 5px; }

  /* ---------------------------------------------------------------- modal */
  .overlay { position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
    background: rgba(8, 10, 12, 0.55); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    z-index: 50; }
  .overlay.show { display: flex; }
  .modal { background: rgba(11, 14, 18, 0.9); border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 36px 46px; min-width: 340px; text-align: center; }
  .modal h2 { margin: 0 0 7px; font-size: 14px; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--paper); }
  .modal p { margin: 0 0 24px; font-family: var(--font-mono); font-size: 11px;
    color: var(--silver); line-height: 1.7; }
  .modal .row { display: flex; gap: 10px; justify-content: center; }

  @media (max-width: 900px) {
    .main { grid-template-columns: 1fr; }
    body { padding: 20px 16px 48px; }
    .masthead { flex-direction: column; align-items: flex-start; gap: 14px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <header class="masthead">
    <div>
      <h1 class="title">OPENJEV — THE CHESS <em>INSTRUMENT</em></h1>
      <p class="subtitle">Autonomous adversarial intelligence / precision telemetry</p>
    </div>
    <div class="sys" id="sys"><span class="led"></span><span id="sys-txt">STANDBY</span></div>
  </header>

  <div class="sec-label">01 // SYSTEM TELEMETRY</div>
  <div class="tel">
    <div class="tel-cell"><div class="tel-label">ENGINE</div><div class="tel-val" id="tl-engine">—</div></div>
    <div class="tel-cell"><div class="tel-label">MATCHUP</div><div class="tel-val" id="tl-match">—</div></div>
    <div class="tel-cell"><div class="tel-label">PLY</div><div class="tel-val" id="tl-ply">0</div></div>
    <div class="tel-cell"><div class="tel-label">DEPTH</div><div class="tel-val" id="tl-depth">—</div></div>
    <div class="tel-cell"><div class="tel-label">TEMPO</div><div class="tel-val" id="tl-tempo">—</div></div>
    <div class="tel-cell"><div class="tel-label">LEGAL</div><div class="tel-val" id="tl-legal">20</div></div>
  </div>

  <div class="main">
    <section class="panel board-panel">
      <div class="sec-label">02 // BOARD MATRIX</div>
      <div class="board" id="board"></div>
      <div class="controls">
        <button class="btn btn-run" id="btn-run">RUN</button>
        <button class="btn btn-ghost" id="btn-step">STEP</button>
        <button class="btn btn-ghost" id="btn-reset">RESET</button>
        <div class="config">
          <div class="field"><label>White</label>
            <select id="cfg-white"></select>
          </div>
          <div class="field"><label>Black</label>
            <select id="cfg-black"></select>
          </div>
          <div class="field"><label>Depth</label>
            <input type="number" id="cfg-depth" min="1" max="5" value="3" step="1">
          </div>
          <div class="field"><label>Tempo <span class="tempo-val" id="cfg-tempo-val">0.25s</span></label>
            <input type="range" id="cfg-tempo" min="0.05" max="2.0" step="0.05" value="0.25">
          </div>
          <button class="btn btn-ghost" id="btn-config">DEPLOY</button>
        </div>
      </div>
    </section>

    <aside>
      <div class="sec-label">03 // DUAL EVALUATION</div>

      <div class="panel" style="padding:20px;">
        <div class="duel-card">
          <div class="duel-top">
            <div class="duel-name gn" id="dn-w">White</div>
            <div class="duel-status" id="st-w">READY</div>
          </div>
          <div class="duel-meter w"><div id="dm-w"></div></div>
          <div class="duel-stats">
            <span><label>Model</label><b id="ch-w">—</b></span>
            <span><label>Thinking</label><b id="ps-w-ms">0 ms</b></span>
            <span><label>Nodes</label><b id="ps-w-n">0</b></span>
            <span><label>Last move</label><b id="ps-w-mv">—</b></span>
          </div>
          <div class="captured"><label>Taken</label><span id="cap-w"></span></div>
        </div>

        <div class="duel-card">
          <div class="duel-top">
            <div class="duel-name" id="dn-b">Black</div>
            <div class="duel-status" id="st-b">READY</div>
          </div>
          <div class="duel-meter b"><div id="dm-b"></div></div>
          <div class="duel-stats">
            <span><label>Model</label><b id="ch-b">—</b></span>
            <span><label>Thinking</label><b id="ps-b-ms">0 ms</b></span>
            <span><label>Nodes</label><b id="ps-b-n">0</b></span>
            <span><label>Last move</label><b id="ps-b-mv">—</b></span>
          </div>
          <div class="captured"><label>Taken</label><span id="cap-b"></span></div>
        </div>

        <div class="gauge">
          <div class="gauge-ends"><span>White</span><span id="gv">±0.00</span><span>Black</span></div>
          <div class="gauge-track"><div class="gauge-zero"></div><div class="gauge-white" id="gauge-w"></div></div>
          <div class="gauge-val"><span id="gval">0.00</span> <small>pawns — white perspective</small></div>
        </div>
      </div>

      <div class="panel" style="padding:20px; margin-top:12px;">
        <div class="sec-label">PGN TAPE</div>
        <div class="tape" id="tape"></div>
        <div class="fen"><label>FEN</label><span id="fen"></span></div>
      </div>
    </aside>
  </div>
</div>

<div class="overlay" id="overlay">
  <div class="modal">
    <h2 id="ov-title">CHECKMATE</h2>
    <p id="ov-sub">—</p>
    <div class="row">
      <button class="btn btn-run" id="btn-review">REVIEW POSITION</button>
      <button class="btn btn-ghost" id="btn-new">NEW MATCH</button>
    </div>
  </div>
</div>

<script>
const GLYPH = {P:'♟',N:'♞',B:'♝',R:'♜',Q:'♛',K:'♚',p:'♟',n:'♞',b:'♝',r:'♜',q:'♛',k:'♚'};
const cells = [];
let renderedPlies = -1;
let overlayHidden = false;
let cfgInit = false;
const runBtn = document.getElementById('btn-run');

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
      if (r === 7) {
        const f = document.createElement('span');
        f.className = 'coord coord-file';
        f.textContent = String.fromCharCode(97 + c);
        cell.appendChild(f);
      }
      if (c === 0) {
        const rn = document.createElement('span');
        rn.className = 'coord coord-rank';
        rn.textContent = 8 - r;
        cell.appendChild(rn);
      }
      grid.appendChild(cell);
      cells.push(cell);
    }
  }
}

function sqToRC(name) {
  if (!name || name.length < 2) return null;
  return { r: 8 - parseInt(name[1], 10), c: name.charCodeAt(0) - 97 };
}

function render(s) {
  // board
  const sq = s.squares;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = cells[r * 8 + c];
      const span = cell.querySelector('.piece');
      const sym = sq[r][c];
      cell.classList.remove('pw', 'pb', 'last', 'check');
      if (sym && sym !== ' ') {
        span.textContent = GLYPH[sym] || sym;
        cell.classList.add(sym === sym.toUpperCase() ? 'pw' : 'pb');
      } else {
        span.textContent = '';
      }
    }
  }
  const lm = s.last_move;
  if (lm && lm.from && lm.to) {
    const a = sqToRC(lm.from), b = sqToRC(lm.to);
    if (a) cells[a.r * 8 + a.c].classList.add('last');
    if (b) cells[b.r * 8 + b.c].classList.add('last');
  }
  if (s.check_square) {
    const k = sqToRC(s.check_square);
    if (k) cells[k.r * 8 + k.c].classList.add('check');
  }

  // telemetry
  const wt = (s.players.w.label || s.players.w.type).toUpperCase();
  const bt = (s.players.b.label || s.players.b.type).toUpperCase();
  document.getElementById('tl-engine').innerText = s.engine;
  document.getElementById('tl-match').innerText = wt + ' × ' + bt;
  document.getElementById('tl-ply').innerText = s.plies;
  document.getElementById('tl-depth').innerText = 'd' + s.config.depth + ' · ' + s.config.white + ' / ' + s.config.black;
  document.getElementById('tl-tempo').innerText = s.config.tempo.toFixed(2) + 's / move';
  document.getElementById('tl-legal').innerText = s.legal_move_count;

  // system status
  const sys = document.getElementById('sys');
  const sysTxt = document.getElementById('sys-txt');
  sys.classList.toggle('live', !!s.running);
  if (s.game_over) { sysTxt.innerText = 'COMPLETE'; }
  else { sysTxt.innerText = s.running ? 'LIVE // AUTO' : 'PAUSED // MANUAL'; }

  // gauge
  const cp = s.evaluation;
  const cpP = (s.evaluation_mate ? (cp > 0 ? 100 : 0) : Math.max(1.5, Math.min(98.5, 50 + (cp / 100) * 8)));
  document.getElementById('gauge-w').style.width = cpP + '%';
  const gv = s.evaluation_mate ? (cp > 0 ? 'MATE' : '−MATE') : (cp / 100).toFixed(2);
  document.getElementById('gval').innerText = (cp >= 0 ? '+' : '') + gv;
  document.getElementById('gv').innerText = (cp >= 0 ? '+' : '') + gv;

  // duel cards
  const totalThink = s.players.w.thinking_ms + s.players.b.thinking_ms || 1;
  updateCard('w', s.players.w, s, (s.players.w.thinking_ms / totalThink) * 100);
  updateCard('b', s.players.b, s, (s.players.b.thinking_ms / totalThink) * 100);

  // tape
  const acts = s.history || [];
  if (acts.length !== renderedPlies) {
    renderedPlies = acts.length;
    const tape = document.getElementById('tape');
    let html = '';
    for (let i = 0; i < acts.length; i++) {
      if (i % 2 === 0) html += '<span class="num">' + (i / 2 + 1) + '.</span>';
      html += '<span class="mv ' + acts[i].color + (i === acts.length - 1 ? ' latest' : '') + '">' + acts[i].san + '</span>';
    }
    tape.innerHTML = html || '<span style="color:#6E747D">awaiting first move…</span>';
    tape.scrollTop = tape.scrollHeight;
  }
  document.getElementById('fen').innerText = s.fen;
  runBtn.innerText = s.running ? 'PAUSE' : 'RUN';

  if (!cfgInit) {
    cfgInit = true;
    const roster = (s.config.roster || []);
    ['white', 'black'].forEach(function (side) {
      const sel = document.getElementById('cfg-' + side);
      sel.innerHTML = '';
      roster.forEach(function (spec) {
        const o = document.createElement('option');
        o.value = spec.name;
        o.textContent = spec.label + ' · ' + spec.tier;
        o.title = spec.hint || '';
        sel.appendChild(o);
      });
      sel.value = s.config[side];
    });
    document.getElementById('cfg-depth').value = s.config.depth;
    document.getElementById('cfg-tempo').value = s.config.tempo;
    document.getElementById('cfg-tempo-val').innerText = s.config.tempo.toFixed(2) + 's';
  }

  // modal
  if (s.result && !overlayHidden) {
    document.getElementById('ov-title').innerText =
      s.result.winner_name ? (s.result.winner_name + ' VICTORY') : 'DRAW';
    document.getElementById('ov-sub').innerText = s.result.label;
    document.getElementById('overlay').classList.add('show');
  } else if (!s.result) {
    overlayHidden = false;
    document.getElementById('overlay').classList.remove('show');
  }
}

function updateCard(key, p, s, sharePct) {
  const self = s.turn === key;
  const st = document.getElementById('st-' + key);
  st.classList.remove('think', 'win', 'lose');
  let txt;
  if (s.game_over) {
    const winner = s.result && s.result.winner;
    if (winner === key) { st.classList.add('win'); txt = 'VICTORY'; }
    else if (winner) { st.classList.add('lose'); txt = 'DEFEAT'; }
    else { txt = 'DRAW'; }
  } else if (self) {
    st.classList.add('think');
    txt = s.running ? 'THINKING' : 'TO MOVE';
  } else {
    txt = 'READY';
  }
  st.innerText = txt;

  const mn = key === 'w' ? 'dn-w' : 'dn-b';
  document.getElementById(mn).innerText = key === 'w' ? 'White' : 'Black';
  document.getElementById('ch-' + key).innerText = p.label || p.type;
  document.getElementById('ps-' + key + '-ms').innerText = Math.round(p.thinking_ms) + ' ms';
  document.getElementById('ps-' + key + '-n').innerText = p.nodes.toLocaleString();

  let last = '—';
  for (let i = (s.history || []).length - 1; i >= 0; i--) {
    if (s.history[i].color === key) { last = s.history[i].san; break; }
  }
  document.getElementById('ps-' + key + '-mv').innerText = last;
  document.getElementById('dm-' + key).style.width = sharePct + '%';

  const cap = (s.captured && s.captured[key]) || [];
  document.getElementById('cap-' + key).innerText = cap.map(function (g) { return GLYPH[g] || g; }).join(' ');
}

async function fetchState() {
  try {
    const res = await fetch('/api/chess/state');
    render(await res.json());
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

function setup() {
  runBtn.addEventListener('click', function () {
    doPost('/api/chess/toggle').then(function () {
      renderedPlies = -1;
      fetchState();
    });
  });
  document.getElementById('btn-step').addEventListener('click', function () {
    doPost('/api/chess/step').then(function () { renderedPlies = -1; fetchState(); });
  });
  document.getElementById('btn-reset').addEventListener('click', function () {
    overlayHidden = false;
    doPost('/api/chess/reset').then(function () { renderedPlies = -1; fetchState(); });
  });

  const tempo = document.getElementById('cfg-tempo');
  tempo.addEventListener('input', function () {
    document.getElementById('cfg-tempo-val').innerText = parseFloat(tempo.value).toFixed(2) + 's';
  });
  document.getElementById('btn-config').addEventListener('click', function () {
    doPost('/api/chess/config', {
      white: document.getElementById('cfg-white').value,
      black: document.getElementById('cfg-black').value,
      depth: parseInt(document.getElementById('cfg-depth').value, 10),
      tempo: parseFloat(tempo.value)
    }).then(function () { renderedPlies = -1; fetchState(); });
  });

  document.getElementById('btn-review').addEventListener('click', function () {
    overlayHidden = true;
    document.getElementById('overlay').classList.remove('show');
  });
  document.getElementById('btn-new').addEventListener('click', function () {
    overlayHidden = false;
    doPost('/api/chess/reset').then(function () {
      return doPost('/api/chess/toggle');
    }).then(function () { renderedPlies = -1; fetchState(); });
  });
}

buildBoard();
setup();
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
    print(f"[*] Serving on http://0.0.0.0:{port} (paused at move 0 — press RUN)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopped chess arena server.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="The Chess Instrument — minimalist web viewer")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--white", default="minimax", choices=PLAYER_TYPES)
    parser.add_argument("--black", default="heuristic", choices=PLAYER_TYPES)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--tempo", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    serve_chess(port=args.port, white=args.white, black=args.black,
                depth=args.depth, tempo=args.tempo, seed=args.seed)


if __name__ == "__main__":
    main()