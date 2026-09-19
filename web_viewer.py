"""Web visualizer for openjev Tetris.
Serves a sleek dark-mode UI with live board display, NLI probability bars, and step-by-step reasoning."""
import http.server
import json
import socketserver
import threading
import time
from typing import Dict, Optional

from tetris_env import Tetris, oracle_policy, compute_board_features
from tetris_prompts import generate_candidate_pairs, build_pair, get_move_descriptors


class TetrisGameManager:
    def __init__(self, player=None):
        self.env = Tetris(seed=42)
        self.player = player
        self.mode = "oracle" if player is None else "openjev"
        self.strategy = "outcome_entailment"
        self.paused = True
        self.fps = 2.0
        self.last_step_info: Dict = {}
        self.lock = threading.Lock()

    def step(self):
        with self.lock:
            if self.env.done:
                self.env.reset()
                self.last_step_info = {}

            legal_moves = self.env.get_legal_moves()
            if not legal_moves:
                self.env.done = True
                return

            t0 = time.perf_counter()
            ranked_moves = []
            best_move = legal_moves[0]

            if self.player is not None and self.mode == "openjev":
                self.player.strategy = self.strategy
                scores, probs = self.player.score_moves(self.env, legal_moves)
                pairs = generate_candidate_pairs(self.env, legal_moves, strategy=self.strategy)

                for idx, (r, c) in enumerate(legal_moves):
                    ranked_moves.append({
                        "rot": r,
                        "col": c,
                        "score": round(scores[idx], 4),
                        "p_ent": round(float(probs[idx, 1]), 4),
                        "p_con": round(float(probs[idx, 0]), 4),
                        "p_neu": round(float(probs[idx, 2]), 4),
                        "premise": pairs[idx][0],
                        "hypothesis": pairs[idx][1]
                    })
                ranked_moves.sort(key=lambda m: m["score"], reverse=True)
                best_move = (ranked_moves[0]["rot"], ranked_moves[0]["col"])
            else:
                # Oracle or random
                best_move = oracle_policy(self.env)
                # Generate sample pairs for top move display
                for r, c in legal_moves:
                    desc = get_move_descriptors(self.env, r, c)
                    premise, hypothesis = build_pair(desc, strategy=self.strategy)
                    ranked_moves.append({
                        "rot": r,
                        "col": c,
                        "score": 1.0 if (r, c) == best_move else 0.0,
                        "p_ent": 0.95 if (r, c) == best_move else 0.05,
                        "p_con": 0.02 if (r, c) == best_move else 0.85,
                        "p_neu": 0.03 if (r, c) == best_move else 0.10,
                        "premise": premise,
                        "hypothesis": hypothesis
                    })
                ranked_moves.sort(key=lambda m: m["score"], reverse=True)

            lat_ms = (time.perf_counter() - t0) * 1000
            rot, col = best_move
            _, reward, done = self.env.step(rot, col)

            self.last_step_info = {
                "chosen_move": [rot, col],
                "lat_ms": round(lat_ms, 1),
                "reward": reward,
                "ranked_candidates": ranked_moves[:5]
            }

    def get_state(self) -> Dict:
        with self.lock:
            st = self.env.state()
            return {
                "grid": st["grid"],
                "current_piece": st["current_piece"],
                "next_piece": st["next_piece"],
                "score": st["score"],
                "lines": st["lines_cleared"],
                "pieces": st["pieces_placed"],
                "done": st["done"],
                "features": st["features"],
                "last_step": self.last_step_info,
                "mode": self.mode,
                "strategy": self.strategy,
                "paused": self.paused,
                "fps": self.fps
            }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>openjev Tetris</title>
<style>
  :root {
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --accent: #58a6ff;
    --ent: #238636;
    --con: #da3633;
    --neu: #8b949e;
  }
  body {
    margin: 0;
    padding: 20px;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  h1 { margin: 0 0 10px 0; font-size: 24px; color: #58a6ff; }
  .tagline { color: var(--neu); font-size: 14px; margin-bottom: 20px; }
  .container {
    display: flex;
    gap: 25px;
    max-width: 1200px;
    width: 100%;
    justify-content: center;
  }
  .panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  .board {
    display: grid;
    grid-template-columns: repeat(10, 24px);
    grid-template-rows: repeat(20, 24px);
    gap: 1px;
    background: #21262d;
    border: 2px solid var(--border);
    border-radius: 4px;
    padding: 2px;
  }
  .cell {
    width: 24px;
    height: 24px;
    background: #0d1117;
    border-radius: 2px;
  }
  .cell.I { background: #00f0f0; box-shadow: inset 0 0 4px #fff; }
  .cell.O { background: #f0f000; box-shadow: inset 0 0 4px #fff; }
  .cell.T { background: #a000f0; box-shadow: inset 0 0 4px #fff; }
  .cell.S { background: #00f000; box-shadow: inset 0 0 4px #fff; }
  .cell.Z { background: #f00000; box-shadow: inset 0 0 4px #fff; }
  .cell.J { background: #0000f0; box-shadow: inset 0 0 4px #fff; }
  .cell.L { background: #f0a000; box-shadow: inset 0 0 4px #fff; }

  .stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 15px;
    font-size: 14px;
  }
  .stat-box {
    background: #0d1117;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .stat-label { color: var(--neu); font-size: 11px; text-transform: uppercase; }
  .stat-val { font-size: 18px; font-weight: bold; color: #fff; margin-top: 2px; }

  .controls {
    display: flex;
    gap: 10px;
    margin-top: 15px;
  }
  button {
    background: #21262d;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 8px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
  }
  button:hover { background: #30363d; }
  button.primary { background: #238636; color: #fff; border: none; }
  button.primary:hover { background: #2ea043; }

  .reasoning-panel {
    flex: 1;
    max-width: 600px;
  }
  .cand-card {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 10px;
    font-size: 13px;
  }
  .cand-card.best {
    border-color: #238636;
    background: rgba(35, 134, 54, 0.08);
  }
  .cand-header {
    display: flex;
    justify-content: space-between;
    font-weight: bold;
    margin-bottom: 6px;
  }
  .prob-bar {
    display: flex;
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
    margin-bottom: 8px;
    background: #30363d;
  }
  .prob-ent { background: #238636; }
  .prob-con { background: #da3633; }
  .prob-neu { background: #8b949e; }
  .premise-box {
    color: #8b949e;
    font-size: 11px;
    line-height: 1.4;
    border-left: 2px solid #30363d;
    padding-left: 8px;
    margin-top: 4px;
  }
  .hyp-box {
    color: #79c0ff;
    font-size: 11px;
    margin-top: 4px;
    font-style: italic;
  }
</style>
</head>
<body>

<h1>openjev Tetris</h1>
<div class="tagline">AlexWortega/openjev NLI Cross-Encoder as Zero-Shot Tetris Controller</div>

<div class="container">
  <div class="panel">
    <div id="board" class="board"></div>
    <div class="stats">
      <div class="stat-box">
        <div class="stat-label">Score</div>
        <div id="stat-score" class="stat-val">0</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Lines Cleared</div>
        <div id="stat-lines" class="stat-val">0</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Piece / Next</div>
        <div id="stat-piece" class="stat-val">- / -</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Decision Latency</div>
        <div id="stat-lat" class="stat-val">0 ms</div>
      </div>
    </div>
    <div class="controls">
      <button id="btn-toggle" class="primary" onclick="togglePlay()">Start</button>
      <button onclick="step()">Step</button>
      <button onclick="resetGame()">Reset</button>
    </div>
  </div>

  <div class="panel reasoning-panel">
    <h3 style="margin-top:0">NLI Reasoning & Candidate Moves</h3>
    <div style="font-size: 12px; color: var(--neu); margin-bottom: 12px;">
      Legend: <span style="color:#238636">■ Entailment</span> | <span style="color:#da3633">■ Contradiction</span> | <span style="color:#8b949e">■ Neutral</span>
    </div>
    <div id="candidates"></div>
  </div>
</div>

<script>
let running = false;
let timer = null;

function renderBoard(grid) {
  const container = document.getElementById('board');
  container.innerHTML = '';
  for (let r = 0; r < 20; r++) {
    for (let c = 0; c < 10; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      if (grid[r][c] !== 0) {
        cell.classList.add(grid[r][c]);
      }
      container.appendChild(cell);
    }
  }
}

function updateUI(data) {
  renderBoard(data.grid);
  document.getElementById('stat-score').innerText = data.score;
  document.getElementById('stat-lines').innerText = data.lines;
  document.getElementById('stat-piece').innerText = data.current_piece + ' / ' + data.next_piece;
  if (data.last_step && data.last_step.lat_ms) {
    document.getElementById('stat-lat').innerText = data.last_step.lat_ms + ' ms';
  }

  const candBox = document.getElementById('candidates');
  candBox.innerHTML = '';
  if (data.last_step && data.last_step.ranked_candidates) {
    data.last_step.ranked_candidates.forEach((c, idx) => {
      const card = document.createElement('div');
      card.className = 'cand-card' + (idx === 0 ? ' best' : '');
      const entPct = (c.p_ent * 100).toFixed(1);
      const conPct = (c.p_con * 100).toFixed(1);
      const neuPct = (c.p_neu * 100).toFixed(1);

      card.innerHTML = `
        <div class="cand-header">
          <span>${idx === 0 ? '★ Chosen: ' : ''}Rot ${c.rot}, Col ${c.col}</span>
          <span>Score: ${c.score.toFixed(3)}</span>
        </div>
        <div class="prob-bar">
          <div class="prob-ent" style="width: ${entPct}%" title="Entailment: ${entPct}%"></div>
          <div class="prob-con" style="width: ${conPct}%" title="Contradiction: ${conPct}%"></div>
          <div class="prob-neu" style="width: ${neuPct}%" title="Neutral: ${neuPct}%"></div>
        </div>
        <div class="premise-box"><strong>Premise:</strong> ${c.premise}</div>
        <div class="hyp-box"><strong>Hypothesis:</strong> ${c.hypothesis}</div>
      `;
      candBox.appendChild(card);
    });
  }
}

async function fetchState() {
  const res = await fetch('/api/state');
  const data = await res.json();
  updateUI(data);
}

async function step() {
  const res = await fetch('/api/step', { method: 'POST' });
  const data = await res.json();
  updateUI(data);
}

async function resetGame() {
  await fetch('/api/reset', { method: 'POST' });
  fetchState();
}

function togglePlay() {
  running = !running;
  const btn = document.getElementById('btn-toggle');
  if (running) {
    btn.innerText = 'Pause';
    btn.classList.remove('primary');
    timer = setInterval(step, 600);
  } else {
    btn.innerText = 'Start';
    btn.classList.add('primary');
    clearInterval(timer);
  }
}

fetchState();
</script>
</body>
</html>
"""


class RequestHandler(http.server.BaseHTTPRequestHandler):
    manager: TetrisGameManager = None

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/api/state":
            st = self.manager.get_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(st).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/step":
            self.manager.step()
            st = self.manager.get_state()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(st).encode("utf-8"))
        elif self.path == "/api/reset":
            with self.manager.lock:
                self.manager.env.reset()
                self.manager.last_step_info = {}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence console log spam for cleaner output
        pass


def start_server(player=None, port: int = 8088):
    manager = TetrisGameManager(player=player)
    RequestHandler.manager = manager
    server = socketserver.TCPServer(("0.0.0.0", port), RequestHandler)
    server.allow_reuse_address = True
    print(f"[*] openjev Tetris visualizer running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--mode", default="oracle", choices=["oracle", "openjev"])
    args = parser.parse_args()

    player = None
    if args.mode == "openjev":
        from tetris_player import OpenJevPlayer
        player = OpenJevPlayer()

    start_server(player=player, port=args.port)
