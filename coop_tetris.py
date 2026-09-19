"""Cooperative Dual Tetris: Human + openjev AI with synchronized shared pieces."""
import copy
import math
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

from tetris_env import (
    BOARD_WIDTH, BOARD_HEIGHT, PIECES, compute_board_features,
    dellacherie_eval, oracle_policy
)
from tetris_prompts import build_pair, get_move_descriptors


class SynchronizedBag:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.sequence: List[str] = []
        self._refill()

    def _refill(self):
        bag = list(PIECES.keys())
        self.rng.shuffle(bag)
        self.sequence.extend(bag)

    def get_piece(self, idx: int) -> str:
        while idx >= len(self.sequence):
            self._refill()
        return self.sequence[idx]


from tetris_engine import (
    BitBoard, SPAWN_POS, get_kick_offsets, grid_to_bitboard,
    SRS_PIECES_COORDS, compute_bitboard_features
)


class HumanBoard:
    def __init__(self, bag: SynchronizedBag, lock_delay: float = 0.5, max_lock_resets: int = 15):
        self.bag = bag
        self.lock_delay = lock_delay
        self.max_lock_resets = max_lock_resets
        self.reset()

    def reset(self):
        self.board = BitBoard()
        self.piece_idx = 0
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.done = False

        self.cur_piece = self.bag.get_piece(self.piece_idx)
        self.cur_rot = 0
        sp_x, sp_y = SPAWN_POS[self.cur_piece]
        self.cur_x = sp_x
        self.cur_y = sp_y
        self.hold_piece: Optional[str] = None
        self.can_hold = True

        self.is_grounded = self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y + 1)
        self.lock_timer = self.lock_delay
        self.lock_resets = 0

        if self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y):
            self.done = True

    @property
    def grid(self) -> List[List[int]]:
        return [
            [self.board.color_grid[r][c] or 0 for c in range(BOARD_WIDTH)]
            for r in range(BOARD_HEIGHT)
        ]

    def move_left(self):
        if self.done: return
        if not self.board.collides(self.cur_piece, self.cur_rot, self.cur_x - 1, self.cur_y):
            self.cur_x -= 1
            self._handle_move_reset()

    def move_right(self):
        if self.done: return
        if not self.board.collides(self.cur_piece, self.cur_rot, self.cur_x + 1, self.cur_y):
            self.cur_x += 1
            self._handle_move_reset()

    def rotate_cw(self):
        if self.done: return
        target_rot = (self.cur_rot + 1) % 4
        kicks = get_kick_offsets(self.cur_piece, self.cur_rot, target_rot)
        for dx, dy in kicks:
            test_x = self.cur_x + dx
            test_y = self.cur_y + dy
            if not self.board.collides(self.cur_piece, target_rot, test_x, test_y):
                self.cur_rot = target_rot
                self.cur_x = test_x
                self.cur_y = test_y
                self._handle_move_reset()
                return

    def rotate_ccw(self):
        if self.done: return
        target_rot = (self.cur_rot - 1) % 4
        kicks = get_kick_offsets(self.cur_piece, self.cur_rot, target_rot)
        for dx, dy in kicks:
            test_x = self.cur_x + dx
            test_y = self.cur_y + dy
            if not self.board.collides(self.cur_piece, target_rot, test_x, test_y):
                self.cur_rot = target_rot
                self.cur_x = test_x
                self.cur_y = test_y
                self._handle_move_reset()
                return

    def _handle_move_reset(self):
        new_grounded = self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y + 1)
        if new_grounded:
            if self.is_grounded:
                if self.lock_resets < self.max_lock_resets:
                    self.lock_resets += 1
                    self.lock_timer = self.lock_delay
            else:
                self.is_grounded = True
                self.lock_timer = self.lock_delay
        else:
            self.is_grounded = False

    def soft_drop(self) -> bool:
        if self.done: return False
        if not self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y + 1):
            self.cur_y += 1
            new_grounded = self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y + 1)
            if new_grounded and not self.is_grounded:
                self.is_grounded = True
                self.lock_timer = self.lock_delay
            return True
        else:
            self.is_grounded = True
            return False

    def hard_drop(self):
        if self.done: return
        self.cur_y = self.board.get_drop_y(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y)
        self.lock_piece()

    def hold(self):
        if self.done or not self.can_hold: return
        if self.hold_piece is None:
            self.hold_piece = self.cur_piece
            self.piece_idx += 1
            self.cur_piece = self.bag.get_piece(self.piece_idx)
        else:
            self.hold_piece, self.cur_piece = self.cur_piece, self.hold_piece
        self.cur_rot = 0
        sp_x, sp_y = SPAWN_POS[self.cur_piece]
        self.cur_x = sp_x
        self.cur_y = sp_y
        self.can_hold = False
        self.is_grounded = self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y + 1)
        self.lock_timer = self.lock_delay
        self.lock_resets = 0
        if self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y):
            self.done = True

    def lock_piece(self):
        if self.done: return
        lines, landing_h = self.board.stamp(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y)
        self.lines_cleared += lines
        self.pieces_placed += 1
        line_pts = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += line_pts.get(lines, 0) + 1

        self.piece_idx += 1
        self.cur_piece = self.bag.get_piece(self.piece_idx)
        self.cur_rot = 0
        sp_x, sp_y = SPAWN_POS[self.cur_piece]
        self.cur_x = sp_x
        self.cur_y = sp_y
        self.can_hold = True
        self.is_grounded = self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y + 1)
        self.lock_timer = self.lock_delay
        self.lock_resets = 0

        if self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y):
            self.done = True

    def tick(self, dt: float):
        if self.done: return
        now_grounded = self.board.collides(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y + 1)
        if now_grounded:
            if not self.is_grounded:
                self.is_grounded = True
                self.lock_timer = self.lock_delay
            self.lock_timer -= dt
            if self.lock_timer <= 0:
                self.lock_piece()
        else:
            self.is_grounded = False

    def get_state(self) -> Dict:
        ghost_y = self.board.get_drop_y(self.cur_piece, self.cur_rot, self.cur_x, self.cur_y)
        return {
            "grid": self.grid,
            "cur_piece": self.cur_piece,
            "cur_x": self.cur_x,
            "cur_y": self.cur_y,
            "cur_rot": self.cur_rot,
            "ghost_y": ghost_y,
            "hold_piece": self.hold_piece,
            "can_hold": self.can_hold,
            "is_grounded": self.is_grounded,
            "lock_timer": round(self.lock_timer, 2),
            "next_piece": self.bag.get_piece(self.piece_idx + 1),
            "score": self.score,
            "lines": self.lines_cleared,
            "piece_idx": self.piece_idx,
            "done": self.done
        }


class AIPlayerBoard:
    def __init__(self, bag: SynchronizedBag):
        self.bag = bag
        self.reset()

    def reset(self):
        self.board = BitBoard()
        self.piece_idx = 0
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.done = False
        self.cur_piece = self.bag.get_piece(self.piece_idx)
        self.cur_rot = 0
        self.cur_x = 3
        self.cur_y = 0
        self.hold_piece = None
        self.can_hold = True
        self.last_step_info: Dict = {}

    @property
    def grid(self) -> List[List[int]]:
        return [
            [self.board.color_grid[r][c] or 0 for c in range(BOARD_WIDTH)]
            for r in range(BOARD_HEIGHT)
        ]

    def get_legal_moves(self, piece: Optional[str] = None) -> List[Tuple[int, int]]:
        orientations = PIECES[piece or self.cur_piece]
        moves = []
        for rot_idx, shape in enumerate(orientations):
            p_width = len(shape[0])
            for col in range(BOARD_WIDTH - p_width + 1):
                if not self._check_collision(shape, col, 0):
                    moves.append((rot_idx, col))
        return moves

    def _check_collision(self, shape: List[List[int]], col: int, row: int) -> bool:
        for r_idx, r in enumerate(shape):
            tr = row + r_idx
            if tr < 0:
                continue
            if tr >= BOARD_HEIGHT:
                return True
            mask = 0
            for c_idx, val in enumerate(r):
                if val:
                    tc = col + c_idx
                    if tc < 0 or tc >= BOARD_WIDTH:
                        return True
                    mask |= (1 << tc)
            if self.board.rows[tr] & mask:
                return True
        return False

    def simulate_placement(self, rot: int, col: int, piece: Optional[str] = None) -> Tuple[List[List[int]], int, int]:
        piece = piece or self.cur_piece
        shape = PIECES[piece][rot]
        h = len(shape)
        w = len(shape[0])
        shape_masks = [0] * h
        for r_idx in range(h):
            m = 0
            for c_idx in range(w):
                if shape[r_idx][c_idx]:
                    m |= (1 << (col + c_idx))
            shape_masks[r_idx] = m

        drop_row = 0
        while drop_row + h < BOARD_HEIGHT:
            collides = False
            for i in range(h):
                if self.board.rows[drop_row + 1 + i] & shape_masks[i]:
                    collides = True
                    break
            if collides:
                break
            drop_row += 1

        landing_h = int(BOARD_HEIGHT - drop_row - (h / 2.0))
        g = [row[:] for row in self.grid]
        for r_idx, r in enumerate(shape):
            for c_idx, val in enumerate(r):
                if val:
                    g[drop_row + r_idx][col + c_idx] = piece
        new_grid = [row for row in g if any(c == 0 for c in row)]
        lines = BOARD_HEIGHT - len(new_grid)
        for _ in range(lines):
            new_grid.insert(0, [0 for _ in range(BOARD_WIDTH)])
        return new_grid, lines, landing_h

    def hold(self) -> bool:
        if self.done or not self.can_hold:
            return False
        if self.hold_piece is None:
            self.hold_piece = self.cur_piece
            self.piece_idx += 1
            self.cur_piece = self.bag.get_piece(self.piece_idx)
        else:
            self.hold_piece, self.cur_piece = self.cur_piece, self.hold_piece
        self.can_hold = False
        self.cur_rot = 0
        self.cur_x = 3
        self.cur_y = 0
        self.done = not self.get_legal_moves()
        return True

    def step(self, rot: int, col: int, info: Optional[Dict] = None):
        if self.done: return
        shape = PIECES[self.cur_piece][rot]
        if self._check_collision(shape, col, 0):
            self.done = True
            return

        new_grid, lines, landing_h = self.simulate_placement(rot, col)
        self.board.rows = grid_to_bitboard(new_grid)
        self.board.color_grid = [
            [new_grid[r][c] if new_grid[r][c] != 0 else None for c in range(BOARD_WIDTH)]
            for r in range(BOARD_HEIGHT)
        ]
        self.lines_cleared += lines
        self.pieces_placed += 1
        line_pts = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += line_pts.get(lines, 0) + 1

        self.piece_idx += 1
        self.cur_piece = self.bag.get_piece(self.piece_idx)
        self.cur_rot = 0
        self.cur_x = 3
        self.cur_y = 0
        self.can_hold = True

        if len(self.get_legal_moves()) == 0:
            self.done = True

        self.last_step_info = info or {
            "chosen_move": [rot, col],
            "lines": lines,
            "landing_h": landing_h
        }

    def get_state(self) -> Dict:
        ghost_y = self.cur_y
        shape = PIECES[self.cur_piece][self.cur_rot % len(PIECES[self.cur_piece])]
        while not self._check_collision(shape, self.cur_x, ghost_y + 1):
            ghost_y += 1
        return {
            "grid": self.grid,
            "cur_piece": self.cur_piece,
            "cur_x": self.cur_x,
            "cur_y": self.cur_y,
            "cur_rot": self.cur_rot,
            "ghost_y": ghost_y,
            "hold_piece": self.hold_piece,
            "can_hold": self.can_hold,
            "next_piece": self.bag.get_piece(self.piece_idx + 1),
            "score": self.score,
            "lines": self.lines_cleared,
            "piece_idx": self.piece_idx,
            "done": self.done,
            "last_step": self.last_step_info
        }


# Fixed projection: when no openjev latents exist, hidden activations are still a
# deterministic function of the 6 real board features, not noise.
_PROJ_RNG = random.Random(7)
_PROJ = [([_PROJ_RNG.uniform(-1.4, 1.4) for _ in range(6)], _PROJ_RNG.uniform(-0.7, 0.7)) for _ in range(48)]


def _project(vals: List[float]) -> List[float]:
    return [round(abs(math.tanh(sum(w * v for w, v in zip(ws, vals)) * 0.9 + b)), 3)
            for ws, b in _PROJ]


def softmax(xs: List[float], temp: float = 3.0) -> List[float]:
    m = max(xs)
    es = [math.exp((x - m) / temp) for x in xs]
    tot = sum(es) or 1.0
    return [e / tot for e in es]


def rank_board_moves(board: AIPlayerBoard, piece: str) -> List[Tuple[float, int, int, int]]:
    ranked = []
    for rot, col in board.get_legal_moves(piece):
        grid, lines, landing_h = board.simulate_placement(rot, col, piece)
        ranked.append((dellacherie_eval(grid, lines, landing_h), rot, col, lines))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def move_nodes(ranked: List[Tuple], best_move: Tuple[int, int], limit: int = 14) -> List[Dict]:
    """Decision layer: the agent's real action space is a (rotation, column) placement,
    one node per candidate the scorer actually evaluated."""
    best = (best_move[0], best_move[1])
    top = ranked[:limit]
    if not any((r[1], r[2]) == best for r in top):
        top = top[:limit - 1] + [r for r in ranked if (r[1], r[2]) == best][:1]
    scs = [r[0] for r in top]
    lo, span = min(scs), (max(scs) - min(scs)) or 1.0
    nodes = [{"rot": r[1], "col": r[2], "lines": r[3],
              "val": round((r[0] - lo) / span, 3),
              "chosen": (r[1], r[2]) == best} for r in top]
    nodes.sort(key=lambda n: (n["col"], n["rot"]))
    return nodes


def generate_neuron_telemetry(feats: Dict, p_ent: float, p_con: float, p_neu: float, latents: Optional[List[float]] = None, moves: Optional[List[Dict]] = None) -> Dict:
    # Input nodes (normalized to 0.0 - 1.0)
    in_nodes = [
        {"name": "Landing Height", "val": round(min(max(feats.get("landing_h", 0) / 20.0, 0.0), 1.0), 2)},
        {"name": "Lines Cleared", "val": round(min(max(feats.get("lines", 0) / 4.0, 0.0), 1.0), 2)},
        {"name": "Holes Added", "val": round(min(max(feats.get("delta_holes", 0) / 3.0, 0.0), 1.0), 2)},
        {"name": "Bumpiness", "val": round(min(max(feats.get("bumpiness", 0) / 15.0, 0.0), 1.0), 2)},
        {"name": "Stack Height", "val": round(min(max(feats.get("max_height", 0) / 20.0, 0.0), 1.0), 2)},
        {"name": "Col Transitions", "val": round(min(max(feats.get("col_transitions", 0) / 20.0, 0.0), 1.0), 2)},
    ]

    # Hidden layer nodes (24 neuron activations)
    hidden_vals = []
    spectrum = []
    if latents is not None and len(latents) >= 24:
        raw = [abs(float(x)) for x in latents[:24]]
        mx = max(raw) or 1.0
        hidden_vals = [round(v / mx, 3) for v in raw]
        spec_raw = [abs(float(x)) for x in latents[:48]]
        spec_mx = max(spec_raw) or 1.0
        spectrum = [round(v / spec_mx, 3) for v in spec_raw]
    else:
        spectrum = _project([n["val"] for n in in_nodes])
        hidden_vals = spectrum[:24]

    # Output nodes
    out_nodes = [
        {"name": "CON", "val": round(p_con, 3), "color": "#f85149"},
        {"name": "ENT", "val": round(p_ent, 3), "color": "#2ea043"},
        {"name": "NEU", "val": round(p_neu, 3), "color": "#8b949e"},
    ]

    return {
        "inputs": in_nodes,
        "hidden": hidden_vals,
        "outputs": out_nodes,
        "spectrum": spectrum,
        "moves": moves or []
    }


class AISlot:
    """One AI board with its own player and its own worker thread: a 20s model
    must not hold up a 1ms one when they play side by side."""

    def __init__(self, bag, player, label: str, mode: str, tempo: float = 0.9):
        self.board = AIPlayerBoard(bag)
        self.player = player
        self.label = label
        self.mode = mode
        self.tempo = tempo
        self.last_step_time = 0.0
        self.board.last_step_info = {
            "chosen_move": [0, 3], "lat_ms": 0.0, "mode": label,
            "p_ent": 0.95, "p_con": 0.02, "p_neu": 0.03,
            "telemetry": generate_neuron_telemetry({"landing_h": 1, "lines": 0, "delta_holes": 0, "bumpiness": 0, "max_height": 0}, 0.95, 0.02, 0.03),
        }


class CoopGameManager:
    def __init__(self, seed: int = 42, ai_player=None, vulkan_player=None, qwen_player=None,
                 sales_player=None, kev_player=None, laya_player=None, spark_player=None,
                 with_heuristic: bool = False, lockstep: bool = True):
        self.seed = seed
        self.bag = SynchronizedBag(seed)
        self.human = HumanBoard(self.bag)
        self.lock = threading.Lock()
        self.running = False
        self.ai_tempo_sec = 0.9
        self.lockstep = lockstep

        self._specs = []
        if with_heuristic or not (ai_player or vulkan_player or qwen_player or sales_player or kev_player or laya_player or spark_player):
            self._specs.append((None, "Heuristic Oracle (Dellacherie)", "fast"))
        if vulkan_player:
            self._specs.append((vulkan_player, vulkan_player.display_name, "vulkan_nli"))
        if ai_player:
            self._specs.append((ai_player, "openjev Zero-Shot NLI (CPU)", "openjev_fast"))
        if kev_player:
            self._specs.append((kev_player, getattr(kev_player, "display_name", "kev-0.5b Decision Model (CPU)"), "kev"))
        if laya_player:
            self._specs.append((laya_player, getattr(laya_player, "display_name", "Laya RLCD Decision Model (CPU)"), "laya"))
        if spark_player:
            self._specs.append((spark_player, getattr(spark_player, "display_name", "djev-spark DiffusionGemma (System 1 Denoising)"), "spark"))
        if qwen_player:
            self._specs.append((qwen_player, qwen_player.display_name, "qwen"))
        if sales_player:
            self._specs.append((sales_player, getattr(sales_player, "display_name", "SalesRLAgent (PPO Conversion Model)"), "sales_rl"))
        self.slots = [AISlot(self.bag, pl, lbl, md, self.ai_tempo_sec) for pl, lbl, md in self._specs]

        for i in range(len(self.slots)):
            threading.Thread(target=self._ai_worker_loop, args=(i,), daemon=True).start()
        threading.Thread(target=self._ai_gravity_loop, daemon=True).start()

    def reset(self, new_seed: Optional[int] = None):
        with self.lock:
            if new_seed is not None:
                self.seed = new_seed
            self.bag = SynchronizedBag(self.seed)
            self.human = HumanBoard(self.bag)
            for slot, (pl, lbl, md) in zip(self.slots, self._specs):
                fresh = AISlot(self.bag, pl, lbl, md, slot.tempo)
                slot.board = fresh.board
                slot.last_step_time = 0.0

    def set_ai_tempo(self, tempo: float):
        with self.lock:
            self.ai_tempo_sec = max(0.0, float(tempo))
            for slot in self.slots:
                slot.tempo = self.ai_tempo_sec

    def _ai_gravity_loop(self):
        while True:
            time.sleep(0.05)
            if not self.running:
                continue
            with self.lock:
                self.human.tick(0.05)

    def _ai_worker_loop(self, idx: int):
        while True:
            time.sleep(0.05)
            slot = self.slots[idx]
            if not self.running or slot.board.done:
                continue

            # Lock-step: every AI board decides on the SAME piece from the shared bag,
            # so the comparison is of decisions, not of inference speed. A board that is
            # ahead waits for the others. --free-run turns this off.
            if self.lockstep and len(self.slots) > 1:
                behind = min(sl.board.piece_idx for sl in self.slots if not sl.board.done)
                if slot.board.piece_idx > behind:
                    continue

            if time.time() - slot.last_step_time < slot.tempo:
                continue

            self._run_single_ai_step(slot)
            slot.last_step_time = time.time()

    def _run_single_ai_step(self, slot):
        with self.lock:
            board = slot.board
            if board.done:
                return
            t0 = time.perf_counter()
            ranked = rank_board_moves(board, board.cur_piece)
            used_hold = False
            if board.can_hold:
                hold_target = board.hold_piece or board.bag.get_piece(board.piece_idx + 1)
                hold_ranked = rank_board_moves(board, hold_target)
                if hold_ranked and (not ranked or hold_ranked[0][0] > ranked[0][0]):
                    board.hold()
                    used_hold = True
                    ranked = hold_ranked
            legal = board.get_legal_moves()
            if not legal:
                slot.board.done = True
                return

            # Fast / Human-Speed Heuristic (instant ~1ms decision)
            player = slot.player
            if player is None:
                best_move = (ranked[0][1], ranked[0][2])
                lat_ms = (time.perf_counter() - t0) * 1000

                sim_grid, lines, landing_h = board.simulate_placement(best_move[0], best_move[1])
                sim_feats = compute_board_features(sim_grid)
                curr_feats = compute_board_features(board.grid)
                delta_holes = sim_feats["holes"] - curr_feats["holes"]

                feats = {
                    "landing_h": landing_h,
                    "lines": lines,
                    "delta_holes": delta_holes,
                    "bumpiness": sim_feats["bumpiness"],
                    "max_height": sim_feats["max_height"],
                    "col_transitions": sim_feats["col_transitions"]
                }
                # Real decision confidence over the candidate set, not a constant.
                # ENT = how decisively the heuristic prefers the chosen move;
                # CON = how much that move still hurts the board.
                p_ent = round(softmax([r[0] for r in ranked])[0], 3)
                p_con = round(min(max(delta_holes, 0) / 3.0 * 0.6 + landing_h / 20.0 * 0.4, 1.0 - p_ent), 3)
                p_neu = round(max(0.0, 1.0 - p_ent - p_con), 3)
                telemetry = generate_neuron_telemetry(feats, p_ent=p_ent, p_con=p_con, p_neu=p_neu,
                                                      moves=move_nodes(ranked, best_move))

                info = {
                    "chosen_move": [best_move[0], best_move[1]],
                    "used_hold": used_hold,
                    "lat_ms": round(lat_ms, 1),
                    "mode": slot.label,
                    "p_ent": p_ent,
                    "p_con": p_con,
                    "p_neu": p_neu,
                    "telemetry": telemetry,
                    "top_candidates": [
                        {"rot": r, "col": c, "score": round(sc, 2), "lines": l}
                        for sc, r, c, l in ranked[:3]
                    ]
                }
                board.step(best_move[0], best_move[1], info)
                return

            # Snapshot the board while holding the lock, then release it for model
            # inference. Rendering and human input must remain responsive while an AI
            # server is working. The board only accepts the result if it is unchanged.
            from tetris_env import Tetris
            decision_piece_idx = board.piece_idx
            decision_piece = board.cur_piece
            dummy_env = Tetris(seed=0, max_steps=1)
            dummy_env.grid = [row[:] for row in board.grid]
            dummy_env.bit_rows = board.board.rows[:]
            dummy_env.current_piece = board.cur_piece
            dummy_env.next_piece = board.bag.get_piece(board.piece_idx + 1)
            dummy_env.hold_piece = board.hold_piece
            dummy_env.can_hold = board.can_hold

        t0 = time.perf_counter()
        try:
            best_move, decision_info = player.choose_move(dummy_env)
        except Exception as exc:
            print(f"[!] {slot.label} inference failed: {exc}", flush=True)
            return

        with self.lock:
            if (slot.board is not board or board.piece_idx != decision_piece_idx or
                    board.cur_piece != decision_piece):
                return  # reset or another accepted move made this result stale
            if best_move not in board.get_legal_moves():
                return

            lat_ms = (time.perf_counter() - t0) * 1000
            sim_grid, lines, landing_h = board.simulate_placement(best_move[0], best_move[1])
            sim_feats = compute_board_features(sim_grid)
            curr_feats = compute_board_features(board.grid)
            delta_holes = sim_feats["holes"] - curr_feats["holes"]
            feats = {
                "landing_h": landing_h, "lines": lines, "delta_holes": delta_holes,
                "bumpiness": sim_feats["bumpiness"], "max_height": sim_feats["max_height"],
                "col_transitions": sim_feats["col_transitions"]
            }
            p_ent = decision_info.get("best_p_ent", 0.8)
            p_con = decision_info.get("best_p_con", 0.1)
            p_neu = decision_info.get("best_p_neu", 0.1)
            cand_ranked = sorted(
                ((sc, m[0], m[1], 0) for sc, m in zip(decision_info.get("cand_scores", []),
                                                      decision_info.get("candidates", []))),
                key=lambda r: r[0], reverse=True)
            telemetry = generate_neuron_telemetry(
                feats, p_ent=p_ent, p_con=p_con, p_neu=p_neu,
                latents=decision_info.get("best_latent"),
                moves=move_nodes(cand_ranked, best_move) if cand_ranked else [])
            info = {
                "chosen_move": [best_move[0], best_move[1]],
                "used_hold": used_hold,
                "lat_ms": round(lat_ms, 1), "mode": slot.label,
                "p_ent": p_ent, "p_con": p_con, "p_neu": p_neu,
                "score": decision_info.get("best_score", 0.0), "telemetry": telemetry,
                "confidence": decision_info.get("confidence"),
                "act_prob": decision_info.get("act_prob"),
                "pitch": decision_info.get("best_pitch", ""),
                "conversion_prob": decision_info.get("conversion_prob"),
                "denoising_trace": decision_info.get("denoising_trace", [])
            }
            board.step(best_move[0], best_move[1], info)

    def human_action(self, action: str):
        with self.lock:
            if action == "left": self.human.move_left()
            elif action == "right": self.human.move_right()
            elif action == "rotate_cw" or action == "up": self.human.rotate_cw()
            elif action == "rotate_ccw": self.human.rotate_ccw()
            elif action == "soft_drop" or action == "down": self.human.soft_drop()
            elif action == "hard_drop" or action == "space": self.human.hard_drop()
            elif action == "hold": self.human.hold()

    def get_full_state(self) -> Dict:
        with self.lock:
            h_st = self.human.get_state()
            ais = []
            for slot in self.slots:
                st = slot.board.get_state()
                st["label"] = slot.label
                st["mode"] = slot.mode
                ais.append(st)
            return {
                "human": h_st,
                "ais": ais,
                "ai": ais[0],  # first AI board, for anything still reading the old shape
                "team_score": h_st["score"] + sum(a["score"] for a in ais),
                "team_lines": h_st["lines"] + sum(a["lines"] for a in ais),
                "running": self.running,
                "ai_tempo": self.ai_tempo_sec,
            }
