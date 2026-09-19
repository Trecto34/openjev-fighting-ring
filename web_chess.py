"""The Chess Instrument — minimalist monochromatic web viewer + HTTP API.

GET  /                     -> minimalist browser client
GET  /api/chess/state      -> full arena telemetry (JSON)
POST /api/chess/step       -> advance one AI move
POST /api/chess/toggle     -> play/pause the auto game loop
POST /api/chess/reset      -> restart the match
POST /api/chess/config     -> swap players / depth / tempo / time control

Registered player kinds from PLAYER_REGISTRY play each other in real time,
driven by ChessGameManager's background worker and orchestrator.
"""

import argparse
import http.server
import json
import socketserver

import chess
import chess.svg

from chess_arena import PLAYER_TYPES, ChessGameManager


def generate_piece_svgs() -> dict:
    svgs = {}
    for sym in "PNBRQK":
        s = chess.svg.piece(chess.Piece.from_symbol(sym))
        # White: crisp ivory/white fill, sharp slate borders
        s = s.replace('fill="#fff"', 'fill="#F8FAFC"')
        s = s.replace('stroke="#000"', 'stroke="#0F172A"')
        svgs[sym] = s
    for sym in "pnbrqk":
        s = chess.svg.piece(chess.Piece.from_symbol(sym))
        # Black: rich dark body with crisp bold white outline
        s = s.replace('fill="#000"', 'fill="#1A202C"')
        s = s.replace('fill:#000000', 'fill:#1A202C')
        s = s.replace('stroke:#000000', 'stroke:#FFFFFF')
        s = s.replace('stroke="#000"', 'stroke:#FFFFFF')
        s = s.replace('stroke-width="1.5"', 'stroke-width="1.8"')
        svgs[sym] = s
    return svgs


CHESS_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OPENJEV — THE CHESS INSTRUMENT</title>
<style>
  :root {
    --ink-0: #0B0C0E;
    --ink-1: #121418;
    --ink-2: #181B22;
    --sq-dark: #141822;
    --sq-light: #2A3344;
    --hairline: rgba(255, 255, 255, 0.08);
    --hairline-2: rgba(255, 255, 255, 0.14);
    --hairline-cell: rgba(255, 255, 255, 0.05);
    --paper: #F4F5F7;
    --silver: #A0A6B2;
    --slate: #6E747D;
    --coord: #737D8F;
    --crimson: #E5484D;
    --amber: #F59E0B;
    --gunmetal: #767D8A;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: "Geist Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html { background: var(--ink-0); }
  body {
    margin: 0; padding: 18px 22px 48px;
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
  .wrap { max-width: 1220px; margin: 0 auto; }

  /* ------------------------------------------------------------- masthead */
  .masthead {
    display: flex; justify-content: space-between; align-items: flex-end;
    padding-bottom: 14px; border-bottom: 1px solid var(--hairline);
    margin-bottom: 18px;
  }
  .title {
    margin: 0; font-size: 26px; font-weight: 650; letter-spacing: -0.015em;
    background: linear-gradient(180deg, #FFFFFF 0%, #C9CDD6 55%, #8B909B 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
  }
  .title em { font-style: normal; letter-spacing: -0.02em;
    background: linear-gradient(180deg, #9BA0AA 0%, #5C626D 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
  .subtitle { margin: 5px 0 0; font-size: 11px; letter-spacing: 0.04em;
    color: var(--slate); font-family: var(--font-mono); }
  .sys {
    display: flex; align-items: center; gap: 9px;
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--silver);
  }
  .sys .led { width: 7px; height: 7px; border-radius: 50%; background: var(--slate); }
  .sys.live .led { background: var(--paper); animation: bk 1.4s ease-in-out infinite; }
  @keyframes bk { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

  /* ------------------------------------------------------------ telemetry */
  .sec-label {
    font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--silver); margin-bottom: 8px;
  }
  .sec-label b { color: var(--paper); font-weight: 500; }
  .tel { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1px; background: var(--hairline); border: 1px solid var(--hairline);
    margin-bottom: 18px; }
  .tel-cell { background: var(--ink-1); padding: 9px 12px; }
  .tel-label { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--slate); margin-bottom: 5px; }
  .tel-val { font-family: var(--font-mono); font-size: 15px; font-weight: 500; color: var(--paper);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* ----------------------------------------------------------------- grid */
  .main { display: grid; grid-template-columns: minmax(340px, 1fr) 350px; gap: 20px; align-items: start; }
  .panel { background: var(--ink-1); border: 1px solid var(--hairline); padding: 16px; }

  /* ---------------------------------------------------------------- board */
  .board-panel {
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
  }
  .board-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .board-tag {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.14em;
    color: var(--slate);
  }
  .board-wrap {
    width: 100%;
    max-width: min(100%, 58vh, 520px);
    margin: 0 auto;
    aspect-ratio: 1 / 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .board {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    grid-template-rows: repeat(8, 1fr);
    aspect-ratio: 1 / 1;
    width: 100%;
    height: 100%;
    gap: 1px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6), inset 0 0 0 1px rgba(255, 255, 255, 0.04);
    border-radius: 4px;
    overflow: hidden;
  }
  .cell {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 1 / 1;
    width: 100%;
    height: 100%;
    overflow: hidden;
    user-select: none;
  }
  .cell.light { background: var(--sq-light); }
  .cell.dark { background: var(--sq-dark); }
  .cell.last { box-shadow: inset 0 0 0 2px rgba(244, 245, 247, 0.6), inset 0 0 16px rgba(255, 255, 255, 0.12); }
  .cell.check { box-shadow: inset 0 0 0 2px var(--crimson), inset 0 0 24px rgba(229, 72, 77, 0.45); }

  /* pieces --------------------------------------------------------------- */
  .piece {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
  }
  .piece svg {
    width: 82%;
    height: 82%;
    display: block;
    pointer-events: none;
    transition: transform 0.12s ease-out;
  }
  .cell.pw .piece svg {
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.65));
  }
  .cell.pb .piece svg {
    filter: drop-shadow(0 0 1.5px rgba(255, 255, 255, 0.75)) drop-shadow(0 2px 5px rgba(0, 0, 0, 0.85));
  }
  .cell:hover .piece svg {
    transform: scale(1.06);
  }
  .coord {
    position: absolute;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    line-height: 1;
    color: var(--coord);
    pointer-events: none;
    z-index: 2;
  }
  .coord-file { right: 4px; bottom: 4px; }
  .coord-rank { left: 4px; top: 4px; }

  /* controls ------------------------------------------------------------- */
  .controls { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 18px; align-items: center; }
  .btn {
    font-family: var(--font-sans); font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 8px 16px; border-radius: 3px; cursor: pointer; line-height: 1;
    transition: all 0.15s ease;
  }
  .btn-run { background: var(--paper); color: var(--ink-0); border: 1px solid var(--paper); }
  .btn-run:hover { background: #E0E4EC; }
  .btn-ghost {
    background: rgba(255, 255, 255, 0.02); color: #C5CBD5;
    border: 1px solid rgba(255, 255, 255, 0.15);
  }
  .btn-ghost:hover { border-color: rgba(255, 255, 255, 0.35); color: var(--paper); background: rgba(255,255,255,0.06); }
  .config { display: flex; gap: 8px; align-items: flex-end; flex-wrap: wrap; margin-left: auto; }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--slate); }
  .field select, .field input[type="number"] {
    background: var(--ink-0); color: var(--paper); border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 3px; font-family: var(--font-mono); font-size: 11px; padding: 6px 8px; outline: none;
  }
  .field select:focus, .field input:focus { border-color: rgba(255, 255, 255, 0.4); }
  .field input[type="range"] { accent-color: var(--paper); width: 90px; margin: 5px 0; }
  .tempo-val { color: var(--paper); font-size: 10px; }

  /* ------------------------------------------------------------------ duel */
  .duel-card { border: 1px solid var(--hairline); background: var(--ink-0); padding: 13px 14px; }
  .duel-card + .duel-card { margin-top: 10px; }
  .duel-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .duel-identity { display: flex; flex-direction: column; gap: 2px; }
  .duel-name { font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase;
    font-weight: 650; color: var(--silver); }
  .duel-name.gn { color: var(--paper); }
  .duel-model-tag { font-family: var(--font-mono); font-size: 10px; color: var(--slate); font-weight: 500; }

  /* digital clock -------------------------------------------------------- */
  .clock-wrap { display: flex; flex-direction: column; align-items: flex-end; gap: 3px; }
  .clock-display {
    font-family: var(--font-mono);
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--silver);
    background: rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 3px 9px;
    line-height: 1.1;
    min-width: 98px;
    text-align: right;
    transition: all 0.15s ease;
  }
  .clock-display.active {
    color: #FFFFFF;
    border-color: rgba(255, 255, 255, 0.5);
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0 0 14px rgba(255, 255, 255, 0.18);
  }
  .clock-display.low {
    color: var(--amber);
    border-color: rgba(245, 158, 11, 0.6);
    box-shadow: 0 0 14px rgba(245, 158, 11, 0.25);
  }
  .clock-display.crit {
    color: var(--crimson);
    border-color: rgba(229, 72, 77, 0.8);
    box-shadow: 0 0 18px rgba(229, 72, 77, 0.4);
    animation: pulse-crit 0.75s infinite alternate ease-in-out;
  }
  @keyframes pulse-crit {
    from { opacity: 1; transform: scale(1); }
    to { opacity: 0.65; transform: scale(1.02); }
  }
  .clock-display.flagged {
    color: var(--crimson);
    border-color: var(--crimson);
    background: rgba(229, 72, 77, 0.18);
  }

  .duel-status { font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--slate); text-align: right; }
  .duel-status.think { color: var(--paper); }
  .duel-status.win { color: var(--paper); }
  .duel-status.lose { color: var(--slate); }

  .duel-meter { height: 2px; background: rgba(255, 255, 255, 0.06); margin: 0 0 10px; }
  .duel-meter > div { height: 100%; width: 0%; transition: width 0.3s ease; }
  .duel-meter.w > div { background: var(--paper); }
  .duel-meter.b > div { background: var(--gunmetal); }

  .duel-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px;
    font-family: var(--font-mono); font-size: 11px; color: var(--slate); }
  .duel-stats span { min-width: 0; }
  .duel-stats label { display: block; font-size: 9px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--slate); margin-bottom: 3px; }
  .duel-stats b { display: block; color: var(--paper); font-weight: 500; font-size: 13px;
    font-family: var(--font-mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  .captured { margin-top: 9px; padding-top: 8px; border-top: 1px solid var(--hairline);
    font-family: var(--font-mono); font-size: 12px; color: var(--slate); min-height: 22px;
    display: flex; align-items: center; flex-wrap: wrap; }
  .captured label { font-size: 9px; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--slate); margin-right: 6px; }
  .cap-glyph { display: inline-block; width: 16px; height: 16px; vertical-align: middle; margin-right: 2px; }
  .cap-glyph svg { width: 100%; height: 100%; display: block; }

  /* gauge ---------------------------------------------------------------- */
  .gauge { margin-top: 12px; }
  .gauge-ends { display: flex; justify-content: space-between;
    font-family: var(--font-mono); font-size: 9px; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--slate); margin-bottom: 7px; }
  .gauge-track { position: relative; height: 5px; background: #222834; border-radius: 3px; overflow: hidden; }
  .gauge-white { position: absolute; left: 0; top: 0; bottom: 0; width: 50%;
    background: var(--paper); transition: width 0.45s cubic-bezier(0.3, 0.7, 0.2, 1); }
  .gauge-zero { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px;
    background: rgba(244, 245, 247, 0.7); z-index: 2; }
  .gauge-val { text-align: center; font-family: var(--font-mono); font-size: 12px;
    color: var(--paper); margin-top: 7px; }
  .gauge-val small { color: var(--slate); font-size: 10px; }

  /* tape + fen ----------------------------------------------------------- */
  .tape { max-height: 120px; overflow-y: auto; scrollbar-width: thin;
    font-family: var(--font-mono); font-size: 11px; color: var(--silver); line-height: 1.8; }
  .tape .num { color: var(--slate); margin-right: 3px; }
  .tape .mv { padding: 1px 4px; margin: 0 1px; border-radius: 2px; }
  .tape .mv.w { color: var(--paper); }
  .tape .mv.b { color: var(--silver); }
  .tape .mv.latest { background: rgba(255, 255, 255, 0.1); box-shadow: inset 0 -1px 0 rgba(244, 245, 247, 0.5); }
  .fen { margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--hairline);
    font-family: var(--font-mono); font-size: 9px; color: var(--slate); line-height: 1.5;
    word-break: break-all; }
  .fen label { display: block; font-size: 8px; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--slate); margin-bottom: 4px; }

  /* ---------------------------------------------------------------- modal */
  .overlay { position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
    background: rgba(8, 10, 12, 0.65); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    z-index: 50; }
  .overlay.show { display: flex; }
  .modal { background: rgba(18, 20, 26, 0.95); border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 32px 42px; min-width: 340px; max-width: 480px; text-align: center; border-radius: 6px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.8); }
  .modal h2 { margin: 0 0 8px; font-size: 16px; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--paper); }
  .modal p { margin: 0 0 24px; font-family: var(--font-mono); font-size: 12px;
    color: var(--silver); line-height: 1.7; }
  .modal .row { display: flex; gap: 12px; justify-content: center; }

  @media (max-width: 768px) {
    .main { grid-template-columns: 1fr; }
    body { padding: 16px 14px 44px; }
    .masthead { flex-direction: column; align-items: flex-start; gap: 12px; }
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
    <div class="tel-cell"><div class="tel-label">TIME CONTROL</div><div class="tel-val" id="tl-tc">5 min</div></div>
    <div class="tel-cell"><div class="tel-label">PLY</div><div class="tel-val" id="tl-ply">0</div></div>
    <div class="tel-cell"><div class="tel-label">TEMPO</div><div class="tel-val" id="tl-tempo">—</div></div>
    <div class="tel-cell"><div class="tel-label">LEGAL</div><div class="tel-val" id="tl-legal">20</div></div>
  </div>

  <div class="main">
    <section class="panel board-panel">
      <div class="board-header">
        <div class="sec-label" style="margin-bottom:0;">02 // BOARD MATRIX</div>
        <div class="board-tag">8 × 8 MONOCHROME MATRIX</div>
      </div>
      <div class="board-wrap">
        <div class="board" id="board"></div>
      </div>
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
          <div class="field"><label>Clock</label>
            <select id="cfg-time">
              <option value="60">1 min (Bullet)</option>
              <option value="180">3 min (Blitz)</option>
              <option value="300" selected>5 min (Rapid)</option>
              <option value="600">10 min (Classical)</option>
              <option value="0">Untimed</option>
            </select>
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

      <div class="panel" style="padding:16px;">
        <div class="duel-card">
          <div class="duel-top">
            <div class="duel-identity">
              <div class="duel-name gn" id="dn-w">White</div>
              <div class="duel-model-tag" id="ch-w">—</div>
            </div>
            <div class="clock-wrap">
              <div class="clock-display" id="clock-w">05:00.0</div>
              <div class="duel-status" id="st-w">READY</div>
            </div>
          </div>
          <div class="duel-meter w"><div id="dm-w"></div></div>
          <div class="duel-stats">
            <span><label>Thinking</label><b id="ps-w-ms">0 ms</b></span>
            <span><label>Nodes</label><b id="ps-w-n">0</b></span>
            <span><label>Last move</label><b id="ps-w-mv">—</b></span>
            <span><label>Material</label><b id="ps-w-mat">0</b></span>
          </div>
          <div class="captured"><label>Taken</label><span id="cap-w"></span></div>
        </div>

        <div class="duel-card">
          <div class="duel-top">
            <div class="duel-identity">
              <div class="duel-name" id="dn-b">Black</div>
              <div class="duel-model-tag" id="ch-b">—</div>
            </div>
            <div class="clock-wrap">
              <div class="clock-display" id="clock-b">05:00.0</div>
              <div class="duel-status" id="st-b">READY</div>
            </div>
          </div>
          <div class="duel-meter b"><div id="dm-b"></div></div>
          <div class="duel-stats">
            <span><label>Thinking</label><b id="ps-b-ms">0 ms</b></span>
            <span><label>Nodes</label><b id="ps-b-n">0</b></span>
            <span><label>Last move</label><b id="ps-b-mv">—</b></span>
            <span><label>Material</label><b id="ps-b-mat">0</b></span>
          </div>
          <div class="captured"><label>Taken</label><span id="cap-b"></span></div>
        </div>

        <div class="gauge">
          <div class="gauge-ends"><span>White</span><span id="gv">±0.00</span><span>Black</span></div>
          <div class="gauge-track"><div class="gauge-zero"></div><div class="gauge-white" id="gauge-w"></div></div>
          <div class="gauge-val"><span id="gval">0.00</span> <small>pawns — white perspective</small></div>
        </div>
      </div>

      <div class="panel" style="padding:16px; margin-top:10px;">
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
const PIECE_SVGS = __PIECE_SVGS__;
const cells = [];
let renderedPlies = -1;
let overlayHidden = false;
let cfgInit = false;
const runBtn = document.getElementById('btn-run');

// Clock telemetry state
let clientClocks = { w: 300, b: 300 };
let activeClock = null;
let isRunning = false;
let isGameOver = false;
let currentTimeCtrl = 300;
let lastTickTime = performance.now();

function formatTime(seconds, isUntimed) {
  if (isUntimed) return '∞ UNTIMED';
  if (seconds <= 0) return '00:00.0';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const tenths = Math.floor((seconds % 1) * 10);
  const mm = (m < 10 ? '0' : '') + m;
  const ss = (s < 10 ? '0' : '') + s;
  return mm + ':' + ss + '.' + tenths;
}

function tickClocks() {
  const now = performance.now();
  const dt = (now - lastTickTime) / 1000.0;
  lastTickTime = now;

  if (isRunning && !isGameOver && currentTimeCtrl > 0 && activeClock) {
    clientClocks[activeClock] = Math.max(0, clientClocks[activeClock] - dt);
  }

  ['w', 'b'].forEach(function (side) {
    const el = document.getElementById('clock-' + side);
    if (!el) return;
    const sec = clientClocks[side];
    el.textContent = formatTime(sec, currentTimeCtrl === 0);
    el.classList.remove('active', 'low', 'crit', 'flagged');

    if (currentTimeCtrl > 0 && sec <= 0) {
      el.classList.add('flagged');
    } else if (activeClock === side && isRunning && !isGameOver) {
      el.classList.add('active');
      if (sec <= 10) el.classList.add('crit');
      else if (sec <= 30) el.classList.add('low');
    }
  });
}
setInterval(tickClocks, 40);

function buildBoard() {
  const grid = document.getElementById('board');
  grid.innerHTML = '';
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell ' + (((r + c) % 2 === 0) ? 'light' : 'dark');
      cell.dataset.r = r; cell.dataset.c = c;
      const p = document.createElement('div');
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
  // Sync clocks with authoritative server state
  currentTimeCtrl = s.time_control !== undefined ? s.time_control : 300;
  if (s.clocks) {
    clientClocks.w = s.clocks.w;
    clientClocks.b = s.clocks.b;
  }
  activeClock = s.active_clock;
  isRunning = !!s.running;
  isGameOver = !!s.game_over;
  lastTickTime = performance.now();

  // board pieces
  const sq = s.squares;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const cell = cells[r * 8 + c];
      const p = cell.querySelector('.piece');
      const sym = sq[r][c];
      cell.classList.remove('pw', 'pb', 'last', 'check');
      if (sym && sym !== ' ') {
        p.innerHTML = PIECE_SVGS[sym] || sym;
        cell.classList.add(sym === sym.toUpperCase() ? 'pw' : 'pb');
      } else {
        p.innerHTML = '';
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
  document.getElementById('tl-tempo').innerText = s.config.tempo.toFixed(2) + 's / move';
  document.getElementById('tl-legal').innerText = s.legal_move_count;

  const tcLabel = currentTimeCtrl === 0 ? 'Untimed' : (currentTimeCtrl / 60) + ' min';
  document.getElementById('tl-tc').innerText = tcLabel;

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
    document.getElementById('cfg-time').value = String(s.config.time_control !== undefined ? s.config.time_control : 300);
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
    else if (winner) {
      txt = (s.result && s.result.type === 'timeout') ? 'FLAGGED' : 'DEFEAT';
      st.classList.add('lose');
    }
    else { txt = 'DRAW'; }
  } else if (self) {
    st.classList.add('think');
    txt = s.running ? 'THINKING' : 'TO MOVE';
  } else {
    txt = 'READY';
  }
  st.innerText = txt;

  document.getElementById('dn-' + key).innerText = key === 'w' ? 'White' : 'Black';
  document.getElementById('ch-' + key).innerText = p.label || p.type;
  document.getElementById('ps-' + key + '-ms').innerText = Math.round(p.thinking_ms) + ' ms';
  document.getElementById('ps-' + key + '-n').innerText = p.nodes.toLocaleString();

  // calculate total piece material count
  const mat = (s.material && s.material[key]) || {};
  let pieceSum = 0;
  const values = { P: 1, N: 3, B: 3, R: 5, Q: 9 };
  for (const k in values) {
    pieceSum += (mat[k] || 0) * values[k];
  }
  document.getElementById('ps-' + key + '-mat').innerText = pieceSum;

  let last = '—';
  for (let i = (s.history || []).length - 1; i >= 0; i--) {
    if (s.history[i].color === key) { last = s.history[i].san; break; }
  }
  document.getElementById('ps-' + key + '-mv').innerText = last;
  document.getElementById('dm-' + key).style.width = sharePct + '%';

  const cap = (s.captured && s.captured[key]) || [];
  const capEl = document.getElementById('cap-' + key);
  if (cap.length === 0) {
    capEl.innerHTML = '<span style="color:var(--slate);font-size:10px;">NONE</span>';
  } else {
    capEl.innerHTML = cap.map(function (sym) {
      const svg = PIECE_SVGS[sym];
      if (svg) return '<span class="cap-glyph">' + svg + '</span>';
      return sym;
    }).join('');
  }
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
      tempo: parseFloat(tempo.value),
      time_control: parseInt(document.getElementById('cfg-time').value, 10)
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

CHESS_HTML = CHESS_HTML_TEMPLATE.replace("__PIECE_SVGS__", json.dumps(generate_piece_svgs()))


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
                depth: int = 3, tempo: float = 0.25, time_control: int = 300, seed=None):
    manager = ChessGameManager(white_type=white, black_type=black,
                               depth=depth, tempo=tempo, time_control=time_control, seed=seed)
    # Start paused at initial board state. User can click RUN or STEP to begin.
    ChessRequestHandler.manager = manager
    socketserver.TCPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), ChessRequestHandler)
    print(f"[*] AI-vs-AI Chess Arena: {white} (white) vs {black} (black), depth {depth}, tempo {tempo}s, time control {time_control}s")
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
    parser.add_argument("--time-control", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    serve_chess(port=args.port, white=args.white, black=args.black,
                depth=args.depth, tempo=args.tempo, time_control=args.time_control, seed=args.seed)


if __name__ == "__main__":
    main()