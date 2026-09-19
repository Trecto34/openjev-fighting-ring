# AI-vs-AI Chess Arena

Real-time bot-vs-bot chess with a cyberpunk esports browser viewer, built on
[`python-chess`](https://python-chess.readthedocs.io/).

Three AI "models" fight it out on a single board: a uniform-random chaos bot, a
greedy single-ply Heuristic, and a full negamax **alpha-beta** Minimax with a
transposition table and quiescence extension.

```
openjev-chess/
├── chess_arena.py     # players + ChessGameManager + headless CLI
├── web_chess.py       # cyberpunk viewer (GET /) + HTTP API
├── run_chess.sh       # launcher (arena / match / bench / test)
└── docs/CHESS_ARENA.md
```

---

## 1. Quickstart

```bash
# Launch the live arena web viewer (default port 8089)
./run_chess.sh arena            # = ./run_chess.sh web 8089
# or directly:
python web_chess.py --port 8089 --white minimax --black heuristic --depth 3 --tempo 0.25
```

Open `http://localhost:8089`. The match starts automatically (RUN), two AIs take
turns in real time, and you can pause, single-step, reset, or re-deploy a
different matchup from the control deck.

The interpreter used throughout is `/home/trecto/openjev-tetris/.venv/bin/python`
(has `python-chess 1.11.2`). `run_chess.sh` prefers a local `.venv` and falls
back to that path.

---

## 2. The Players (`chess_arena.py`)

| Player | Strategy |
|---|---|
| `RandomPlayer` | Uniformly random legal move. The chaos baseline. |
| `HeuristicPlayer` | Single-ply greedy: plays the move maximizing the static evaluation of the resulting position. |
| `MinimaxPlayer` | Negamax alpha-beta search with MVV-LVA move ordering, a per-move transposition table, and an 8-ply capture/promotion quiescence extension. Scores mates by distance so faster mates win. |

### Static evaluation

White-perspective centipawns:

```
score = material + piece-square tables + mobility × 5
```

- Material: P=100, N=320, B=330, R=500, Q=900 (king dominates, mate is scored
  separately at ±MATE with ply distance).
- Piece-square tables are the classic Simplified Evaluation Function values,
  mirrored for Black via `chess.square_mirror`.
- Mobility is legal-move count difference between the sides. The search's leaf
  evaluation skips mobility (material + PST only) for speed — the full
  evaluation is used for the player cards, eval bar, and CLI.

### Minimax details

```
choose_move -> root loop with alpha-beta window (alpha,-beta)
  _negamax(board, depth, alpha, beta, color, ply):
    - checkmate  -> -(MATE - ply)      # prefers fast mates
    - stalemate / insufficient material -> 0
    - depth 0   -> quiescence or static eval
    - TT probe (exact/lower/upper bounds, # of 1M entry slots per move)
    - MVV-LVA ordering + TT best-move first
  _quiesce(board, alpha, beta, color, ply, qdepth):
    - stand pat, then chase captures & promotions (max 8 plies)
```

Nodes per move: ~1.5k–6k at depth 3, ~100–180ms per move on CPU, so full
minimax-vs-minimax matches broadcast at a brisk esports pace.

### Deep API (`chess_arena.py`)

- `evaluate(board, mobility=True) -> int`
- `run_match(white, black, depth, max_plies, seed) -> dict`
- `run_bench(white, black, depth, games, max_plies, seed)`
- `ChessGameManager` — thread-safe game loop and telemetry:
  `step()`, `toggle()`, `reset()`, `set_config(dict)`, `get_state()`.

### Headless CLI

```bash
./run_chess.sh match                    # minimax vs heuristic, PGN to stdout
WHITE=heuristic BLACK=random ./run_chess.sh match
python chess_arena.py --white minimax --black minimax --depth 4 --max-plies 120
./run_chess.sh bench                    # 6-game benchmark series + win stats
```

---

## 3. Web Viewer & HTTP API (`web_chess.py`)

The viewer is a single-page cyberpunk esports broadcast UI: neon board with
last-move/check glows, live evaluation bar, player duel cards (thinking time,
nodes, captured pieces, status LEDs), a scrolling match tape, a PGN dump, and a
fullscreen victory/draw overlay.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Browser client (HTML/CSS/JS) |
| GET | `/api/chess/state` | Full arena telemetry (FEN, board grid, eval, history, PGN, config) |
| POST | `/api/chess/step` | Advance one AI move manually |
| POST | `/api/chess/toggle` | Play/pause the background game loop |
| POST | `/api/chess/reset` | Restart the match |
| POST | `/api/chess/config` | Re-deploy matchup. Body: `{"white","black","depth","tempo"}` |

Changing the player types resets the match; `depth` and `tempo` apply live.
Sample:

```bash
curl -s http://localhost:8089/api/chess/state | python -m json.tool
curl -s -X POST http://localhost:8089/api/chess/toggle
curl -s -X POST http://localhost:8089/api/chess/config \
  -H 'Content-Type: application/json' \
  -d '{"white":"minimax","black":"minimax","depth":4,"tempo":0.4}'
```

### State payload highlights

- `fen`, `squares` (8×8 from rank 8 to rank 1), `turn`, `in_check`, `check_square`
- `evaluation` (centipawns, white-perspective; `evaluation_mate` flag)
- `players.w|b`: type, label, cumulative/current thinking ms, nodes
- `history` (per-move SAN/UCI/color/think-time/eval), `material`, `captured`
- `result` (`checkmate`/`stalemate`/`insufficient_material`/`threefold_repetition`/
  `seventyfive_moves`/`fivefold_repetition`, winner)
- `game_over`, `running`, `config`

### Server flags

```
python web_chess.py --port 8089 --white minimax --black heuristic --depth 3 --tempo 0.25 [--seed N]
```

---

## 4. Tuning Notes

- **Depth 1–2**: fast, blunder-prone; good for a "developmental" arc.
- **Depth 3 (default)**: real-time friendly (~0.15s/move), consistently beats Heuristic.
- **Depth 4–5**: several seconds per move; use `tempo` as the broadcast throttle
  between moves for dramatic effect.
- Against `RandomPlayer`, both Minimax and Heuristic usually win by force within
  a few moves once a hanging piece appears — that is what the eval bar will show.