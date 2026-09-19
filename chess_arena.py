"""AI-vs-AI Chess Arena: Heuristic, Minimax (alpha-beta), and Random players.

* RandomPlayer      - uniform random legal moves.
* HeuristicPlayer   - single-ply greedy maximizer of a static evaluation.
* MinimaxPlayer     - negamax alpha-beta search with move ordering and an
                      optional capture/promotion quiescence extension.
* ChessGameManager  - thread-safe game loop, config, telemetry, and state that
                      drives both the headless CLI and the web arena.

The static evaluation is white-perspective centipawns: material + piece-square
tables + legal-move mobility.  Mate is scored with distance preference so faster
mates are chosen over slower ones.
"""

import random
import threading
import time
from typing import Dict, List, Optional

import chess

INF = 1_000_000_000
MATE = 1_000_000

PLAYER_TYPES = ["random", "heuristic", "minimax"]

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

PAWN_PST = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_PST = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]

QUEEN_PST = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]

KING_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
    chess.KING: KING_PST,
}

ORDER_SCALE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

MOBILITY_WEIGHT = 5


def _pst_value(piece_type: int, color: bool, square: int) -> int:
    table = PIECE_SQUARE_TABLES[piece_type]
    if color == chess.WHITE:
        return table[chess.square_mirror(square)]
    return table[square]


def _legal_move_count(board: chess.Board, color: bool) -> int:
    tmp = board.copy()
    tmp.turn = color
    return tmp.legal_moves.count()


def _material_counts(board: chess.Board) -> Dict[str, Dict[str, int]]:
    counts = {"w": {}, "b": {}}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        key = "w" if piece.color == chess.WHITE else "b"
        label = piece.symbol().upper()
        counts[key][label] = counts[key].get(label, 0) + 1
    return counts


def evaluate(board: chess.Board, mobility: bool = True) -> int:
    """Static evaluation from White's perspective in centipawns."""
    if board.is_checkmate():
        return -MATE if board.turn == chess.WHITE else MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        value = PIECE_VALUES[piece.piece_type] + _pst_value(
            piece.piece_type, piece.color, square
        )
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value

    if mobility:
        if board.turn == chess.WHITE:
            score += MOBILITY_WEIGHT * (
                board.legal_moves.count() - _legal_move_count(board, chess.BLACK)
            )
        else:
            score += MOBILITY_WEIGHT * (
                _legal_move_count(board, chess.WHITE) - board.legal_moves.count()
            )
    return score


def color_key(color: bool) -> str:
    return "w" if color == chess.WHITE else "b"


def player_label(kind: str, depth: int = 3) -> str:
    kind = (kind or "").lower()
    if kind == "minimax":
        return f"MINIMAX // αβ d{depth}"
    if kind == "heuristic":
        return "HEURISTIC // static"
    return "RANDOM // chaos"


class RandomPlayer:
    """Uniform random legal-move chooser."""

    name = "random"
    label = "RANDOM // chaos"
    last_nodes = 0

    def choose_move(self, board: chess.Board, rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return None
        rng = rng if rng is not None else random
        return rng.choice(moves)


class HeuristicPlayer:
    """Greedy single-ply maximizer of the static evaluation."""

    name = "heuristic"
    label = "HEURISTIC // static"
    last_nodes = 0

    def choose_move(self, board: chess.Board, rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        probe = board.copy()
        moves = list(probe.legal_moves)
        if not moves:
            return None
        best_score, best_moves = -INF, []
        for move in moves:
            probe.push(move)
            score = evaluate(probe)
            probe.pop()
            if score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
        rng = rng if rng is not None else random
        return rng.choice(best_moves)


TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2


class MinimaxPlayer:
    """Negamax alpha-beta search with MVV-LVA ordering and a transposition table.

    An optional quiescence extension (default on) keeps the horizon honest by
    continuing to resolve captures and promotions past the nominal depth.
    """

    name = "minimax"
    label = "MINIMAX // αβ"

    # (key, depth, flag, value, best_move_uci | None)
    def __init__(self, depth: int = 3, seed: Optional[int] = None,
                 move_ordering: bool = True, quiescence: bool = True,
                 qdepth_max: int = 8):
        self.depth = max(1, int(depth))
        self.seed = seed
        self.move_ordering = move_ordering
        self.quiescence = quiescence
        self.qdepth_max = qdepth_max
        self.last_nodes = 0
        self.rng = random.Random(seed)
        self.tt: Dict = {}

    def _probe_tt(self, key: int, depth: int):
        entry = self.tt.get(key)
        if entry is not None and entry[1] >= depth:
            return entry
        return None

    def _store_tt(self, key: int, depth: int, flag: int, value: int,
                  best_uci: Optional[str] = None) -> None:
        if abs(value) >= MATE // 2:
            return
        entry = self.tt.get(key)
        if entry is None or entry[1] <= depth:
            self.tt[key] = (key, depth, flag, value, best_uci)

    @staticmethod
    def _order_score(board: chess.Board, move: chess.Move) -> int:
        score = 0
        if move.promotion:
            score += 900 + ORDER_SCALE[move.promotion]
        if board.is_capture(move):
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim_piece = board.piece_at(move.to_square)
                victim = victim_piece.piece_type if victim_piece else chess.PAWN
            attacker_piece = board.piece_at(move.from_square)
            attacker = attacker_piece.piece_type if attacker_piece else chess.PAWN
            score += 10 * ORDER_SCALE[victim] - ORDER_SCALE[attacker]
        return score

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int,
                 color: int, ply: int) -> int:
        self.last_nodes += 1
        if board.is_checkmate():
            return -(MATE - ply)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if depth <= 0:
            if self.quiescence:
                return self._quiesce(board, alpha, beta, color, ply, 0)
            return color * evaluate(board, mobility=False)

        key = board._transposition_key()
        tt_move = None
        entry = self._probe_tt(key, depth)
        if entry is not None:
            flag, _, value, tt_uci = entry[2], entry[1], entry[3], entry[4]
            if flag == TT_EXACT:
                return value
            if flag == TT_LOWER and value >= beta:
                return value
            if flag == TT_UPPER and value <= alpha:
                return value
            tt_move = tt_uci

        moves = list(board.legal_moves)
        if self.move_ordering:
            moves.sort(key=lambda m: self._order_score(board, m), reverse=True)
            if tt_move is not None:
                moves.sort(key=lambda m: 1 if m.uci() == tt_move else 0,
                           reverse=True)

        best = -2 * MATE
        best_uci = None
        for move in moves:
            board.push(move)
            value = -self._negamax(board, depth - 1, -beta, -alpha, -color, ply + 1)
            board.pop()
            if value > best:
                best = value
                best_uci = move.uci()
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break

        if best <= alpha:
            self._store_tt(key, depth, TT_UPPER, best, best_uci)
        elif best >= beta:
            self._store_tt(key, depth, TT_LOWER, best, best_uci)
        else:
            self._store_tt(key, depth, TT_EXACT, best, best_uci)
        return best

    def _quiesce(self, board: chess.Board, alpha: int, beta: int,
                 color: int, ply: int, qdepth: int) -> int:
        self.last_nodes += 1
        stand = color * evaluate(board, mobility=False)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        if qdepth >= self.qdepth_max:
            return stand

        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        if not moves:
            return stand
        if self.move_ordering:
            moves.sort(key=lambda m: self._order_score(board, m), reverse=True)

        for move in moves:
            board.push(move)
            if board.is_checkmate():
                value = MATE - ply - 1
            else:
                value = -self._quiesce(board, -beta, -alpha, -color, ply + 1, qdepth + 1)
            board.pop()
            if value >= beta:
                return value
            if value > alpha:
                alpha = value
        return alpha

    def choose_move(self, board: chess.Board, rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        self.last_nodes = 0
        self.tt = {}
        probe = board.copy()
        moves = list(probe.legal_moves)
        if not moves:
            return None
        if self.move_ordering:
            moves.sort(key=lambda m: self._order_score(probe, m), reverse=True)

        alpha, beta = -INF, INF
        best_score, best_moves = -2 * MATE, []
        for move in moves:
            probe.push(move)
            value = -self._negamax(probe, self.depth - 1, -beta, -alpha, -1, 1)
            probe.pop()
            if value > best_score:
                best_score, best_moves = value, [move]
                alpha = value
            elif value == best_score:
                best_moves.append(move)

        rng = rng if rng is not None else self.rng
        return rng.choice(best_moves)


class ChessGameManager:
    """Thread-safe match controller shared by the web arena and the CLI."""

    def __init__(self, white_type: str = "minimax", black_type: str = "heuristic",
                 depth: int = 3, tempo: float = 0.25, seed: Optional[int] = None):
        self.lock = threading.Lock()
        self.rng = random.Random(seed)
        self.white_type = white_type.lower()
        self.black_type = black_type.lower()
        self.depth = max(1, min(6, int(depth)))
        self.tempo = max(0.02, min(5.0, float(tempo)))
        self.running = False
        self.game_over = False
        self.result: Optional[Dict] = None
        self.board = chess.Board()
        self.move_log: List[Dict] = []
        self.think_ms = {"w": 0.0, "b": 0.0}
        self.last_think_ms = {"w": 0.0, "b": 0.0}
        self.white = None
        self.black = None
        self._last_step = 0.0
        self._exec_count = 0
        self._rebuild_players()
        threading.Thread(target=self._auto_loop, daemon=True).start()

    # ------------------------------------------------------------------ setup
    def _make_player(self, kind: str):
        kind = kind.lower()
        if kind == "random":
            return RandomPlayer()
        if kind == "heuristic":
            return HeuristicPlayer()
        if kind == "minimax":
            return MinimaxPlayer(depth=self.depth, seed=random.randrange(2 ** 31))
        raise ValueError(f"unknown player type {kind!r}")

    def _rebuild_players(self) -> None:
        self.white = self._make_player(self.white_type)
        self.black = self._make_player(self.black_type)

    # --------------------------------------------------------------- game loop
    def _auto_loop(self) -> None:
        while True:
            time.sleep(0.05)
            with self.lock:
                if not self.running or self.game_over:
                    continue
                now = time.monotonic()
                if now - self._last_step >= self.tempo:
                    self._last_step = now
                    self._do_step()

    def _finish(self, reason: str, winner: Optional[str] = None) -> None:
        color_names = {"w": "WHITE", "b": "BLACK"}
        winner_name = color_names.get(winner)
        label = f"{reason.replace('_', ' ').title()} — {winner_name + ' wins' if winner else 'Draw'}"
        self.result = {
            "type": reason,
            "winner": winner,
            "winner_name": winner_name,
            "label": label,
        }
        self.game_over = True
        self.running = False

    def _check_end(self) -> None:
        b = self.board
        if b.is_checkmate():
            self._finish("checkmate", "b" if b.turn == chess.WHITE else "w")
        elif b.is_stalemate():
            self._finish("stalemate")
        elif b.is_insufficient_material():
            self._finish("insufficient_material")
        elif b.is_seventyfive_moves():
            self._finish("seventyfive_moves")
        elif b.is_fivefold_repetition():
            self._finish("fivefold_repetition")
        elif b.is_repetition(3):
            self._finish("threefold_repetition")

    def _do_step(self) -> None:
        if self.game_over:
            return
        color = self.board.turn
        player = self.white if color == chess.WHITE else self.black
        key = color_key(color)
        self._exec_count += 1

        start = time.monotonic()
        move = player.choose_move(self.board, self.rng)
        thinking_ms = (time.monotonic() - start) * 1000.0
        self.think_ms[key] += thinking_ms
        self.last_think_ms[key] = thinking_ms
        nodes = getattr(player, "last_nodes", 0)

        if move is None:
            self._finish("no_legal_moves", "b" if color == chess.WHITE else "w")
            return

        eval_before = evaluate(self.board)
        san = self.board.san(move)
        self.board.push(move)
        self.move_log.append({
            "n": self._exec_count,
            "san": san,
            "uci": move.uci(),
            "color": key,
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "player": player.name,
            "time_ms": round(thinking_ms, 2),
            "nodes": nodes,
            "eval": eval_before,
        })
        self._check_end()

    def step(self) -> None:
        with self.lock:
            self._do_step()

    def toggle(self) -> bool:
        with self.lock:
            self.running = not self.running
            self._last_step = time.monotonic()
            return self.running

    def reset(self) -> None:
        with self.lock:
            self._reset_locked()

    def _reset_locked(self) -> None:
        self.board = chess.Board()
        self.move_log.clear()
        self.think_ms = {"w": 0.0, "b": 0.0}
        self.last_think_ms = {"w": 0.0, "b": 0.0}
        self._exec_count = 0
        self.game_over = False
        self.result = None

    def set_config(self, data: Dict) -> None:
        with self.lock:
            retype_white = "white" in data and data["white"].lower() != self.white_type
            retype_black = "black" in data and data["black"].lower() != self.black_type
            if "depth" in data:
                self.depth = max(1, min(6, int(data["depth"])))
            if retype_white:
                self.white_type = data["white"].lower()
            if retype_black:
                self.black_type = data["black"].lower()
            if retype_white or retype_black:
                self._rebuild_players()
                self._reset_locked()
            elif "depth" in data:
                for player in (self.white, self.black):
                    if isinstance(player, MinimaxPlayer):
                        player.depth = self.depth
            if "tempo" in data:
                self.tempo = max(0.02, min(5.0, float(data["tempo"])))

    # ------------------------------------------------------------------ state
    def _build_squares(self) -> List[List[str]]:
        rows = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                piece = self.board.piece_at(chess.square(file, rank))
                row.append(piece.symbol() if piece else " ")
            rows.append(row)
        return rows

    def _build_pgn(self) -> str:
        parts = []
        for i, entry in enumerate(self.move_log):
            if i % 2 == 0:
                parts.append(f"{i // 2 + 1}.")
            parts.append(entry["san"])
        return " ".join(parts)

    def get_state(self) -> Dict:
        with self.lock:
            return self._state_locked()

    def _state_locked(self) -> Dict:
        board = self.board
        last_move = None
        if board.move_stack:
            move = board.peek()
            last_move = {
                "uci": move.uci(),
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
                "san": self.move_log[-1]["san"] if self.move_log else move.uci(),
                "color": self.move_log[-1]["color"] if self.move_log else color_key(board.turn),
                "player": self.move_log[-1]["player"] if self.move_log else None,
            }

        check_square = None
        if board.is_check():
            check_square = chess.square_name(board.king(board.turn))

        eval_cp = evaluate(board)
        counts = _material_counts(board)
        start_set = {"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1}
        captured = {}
        for side in ("w", "b"):
            missing = []
            for piece_type, start in start_set.items():
                for _ in range(start - counts[side].get(piece_type, 0)):
                    missing.append(piece_type.lower() if side == "b" else piece_type)
            captured["b" if side == "w" else "w"] = missing

        def player_state(player, key: str) -> Dict:
            return {
                "type": player.name,
                "label": player_label(player.name, self.depth)
                if player.name == "minimax"
                else player.name.upper(),
                "thinking_ms": round(self.think_ms[key], 1),
                "last_think_ms": round(self.last_think_ms[key], 1),
                "nodes": getattr(player, "last_nodes", 0),
            }

        return {
            "engine": f"python-chess {chess.__version__}",
            "fen": board.fen(),
            "turn": color_key(board.turn),
            "turn_color": chess.COLOR_NAMES[board.turn],
            "move_number": len(self.move_log) + 1,
            "plies": len(self.move_log),
            "fullmove_number": board.fullmove_number,
            "halfmove_clock": board.halfmove_clock,
            "in_check": board.is_check(),
            "check_square": check_square,
            "legal_move_count": board.legal_moves.count(),
            "squares": self._build_squares(),
            "last_move": last_move,
            "players": {
                "w": player_state(self.white, "w"),
                "b": player_state(self.black, "b"),
            },
            "config": {
                "white": self.white_type,
                "black": self.black_type,
                "depth": self.depth,
                "tempo": round(self.tempo, 2),
                "available": PLAYER_TYPES,
            },
            "running": self.running,
            "game_over": self.game_over,
            "result": self.result,
            "evaluation": eval_cp,
            "evaluation_mate": abs(eval_cp) > MATE // 2,
            "material": counts,
            "captured": captured,
            "history": list(self.move_log),
            "pgn": self._build_pgn(),
            "total_nodes": sum(getattr(p, "last_nodes", 0) for p in (self.white, self.black)),
        }


def run_match(white_type: str = "minimax", black_type: str = "heuristic",
              depth: int = 3, max_plies: int = 200,
              seed: Optional[int] = None) -> Dict:
    manager = ChessGameManager(white_type=white_type, black_type=black_type,
                               depth=depth, seed=seed)
    manager.running = False
    while not manager.game_over and len(manager.move_log) < max_plies:
        manager.step()
    state = manager.get_state()
    return {
        "pgn": state["pgn"],
        "result": state["result"],
        "plies": state["plies"],
        "fen": state["fen"],
        "white": white_type,
        "black": black_type,
        "depth": depth,
    }


def run_bench(white_type: str, black_type: str, depth: int = 3,
              games: int = 6, max_plies: int = 120, seed: Optional[int] = None):
    wins = {"white": 0, "black": 0, "draw": 0}
    plies = []
    for g in range(games):
        result = run_match(white_type, black_type, depth, max_plies,
                           seed=(seed + g if seed is not None else None))
        winner = result["result"]["winner"] if result["result"] else None
        label = {"w": "white", "b": "black"}.get(winner, "draw")
        wins[label] += 1
        plies.append(result["plies"])
        verdict = result["result"]["label"] if result["result"] else "Max ply cap"
        print(f"[{g + 1}] {white_type:>8} vs {black_type:<8} "
              f"{result['plies']:>3} plies  ->  {verdict}")
    avg = sum(plies) / max(1, len(plies))
    print("-" * 60)
    print(f"{white_type:>10} wins: {wins['white']} | "
          f"{black_type:>10} wins: {wins['black']} | draws: {wins['draw']} | "
          f"avg {avg:.1f} plies")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AI-vs-AI Chess Arena (headless)")
    parser.add_argument("--white", default="minimax", choices=PLAYER_TYPES)
    parser.add_argument("--black", default="heuristic", choices=PLAYER_TYPES)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bench", action="store_true",
                        help="Run a benchmark series instead of a single match")
    parser.add_argument("--games", type=int, default=6)
    args = parser.parse_args()

    if args.bench:
        run_bench(args.white, args.black, args.depth, args.games,
                  args.max_plies, args.seed)
        return

    result = run_match(args.white, args.black, args.depth, args.max_plies, args.seed)
    print(result["pgn"])
    print()
    print(f"Result: {result['result']['label'] if result['result'] else 'Unfinished'}"
          f"  ({result['plies']} plies)")


if __name__ == "__main__":
    main()