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

Inference lifecycle: ChessGameManager NEVER calls choose_move while holding the
game lock.  A dedicated worker serializes inference on a snapshot, the
orchestrator commits the published decision under the lock keyed by a generation
counter, so a 17s external judge cannot freeze the web API (Standard C).  All
player kinds are registered once in PLAYER_REGISTRY, which drives construction,
labels, CLI choices, and the web dropdowns (Standard E).
"""

import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import chess

INF = 1_000_000_000
MATE = 1_000_000

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


class ChessPlayer:
    """Player envelope: choose_move(board, rng) -> Optional[Move], reset().

    Every kind (CPU minimax or an external LLM/judge) implements this surface so
    the manager runs them uniformly and on a snapshot, never under the lock.
    """

    name = "base"
    last_nodes = 0

    def choose_move(self, board: chess.Board, rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        raise NotImplementedError

    def reset(self) -> None:
        """Standard D: scrap cross-position state and warm caches per match."""
        self.last_nodes = 0


class RandomPlayer(ChessPlayer):
    """Uniform random legal-move chooser."""

    name = "random"
    label = "RANDOM // chaos"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board, rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return None
        rng = rng if rng is not None else self.rng
        return rng.choice(moves)


class HeuristicPlayer(ChessPlayer):
    """Greedy single-ply maximizer of the static evaluation."""

    name = "heuristic"
    label = "HEURISTIC // static"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

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
        rng = rng if rng is not None else self.rng
        return rng.choice(best_moves)


TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2


class MinimaxPlayer(ChessPlayer):
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

    def reset(self) -> None:
        """Standard D: drop per-match transposition table, keep rng stream."""
        super().reset()
        self.tt = {}

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


CENTER_BB = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5


def _exposure(board: chess.Board, side: bool) -> int:
    """Piece-value weighted enemy attack load piled on side's pieces."""
    enemy = not side
    load = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != side:
            continue
        for attacker in board.attackers(enemy, sq):
            piece_at = board.piece_at(attacker)
            if piece_at is not None:
                load += ORDER_SCALE[piece_at.piece_type]
    return load


def _aggression(board: chess.Board, side: bool) -> int:
    """Piece-value weighted sum of enemy pieces side currently attacks."""
    enemy = not side
    total = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != enemy:
            continue
        if board.attackers(side, sq):
            total += ORDER_SCALE[piece.piece_type]
    return total


def _center_control(board: chess.Board, side: bool) -> int:
    """Number of central squares side's pieces attack right now."""
    total = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None and piece.color == side:
            total += len(board.attacks(sq) & CENTER_BB)
    return total


def _king_pressure(board: chess.Board, side: bool) -> int:
    """Enemy attack weight aimed at side's king and its adjacent zone."""
    king_sq = board.king(side)
    zone = chess.BB_KING_ATTACKS[king_sq] | chess.BB_SQUARES[king_sq]
    enemy = not side
    pressure = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != enemy:
            continue
        att = board.attacks(sq) & zone
        if att:
            pressure += ORDER_SCALE[piece.piece_type] * len(att)
    return pressure


class DJEVPlayer(ChessPlayer):
    """OU-Langevin score smoother over candidate moves.

    Static-evaluation deltas are z-scored into a prior mean vector mu.  Each
    candidate logit is then denoised through T=8 Ornstein-Uhlenbeck steps
    x_{t+1} = x_t + gamma*(mu - x_t) + sigma_t*noise (gamma=0.2) with the
    gaussian refresh schedule decaying to zero; argmax of the final denoised
    vector is played.
    """

    name = "djev"
    label = "DJEV // diffusion logit"

    def __init__(self, t_steps: int = 8, gamma: float = 0.2,
                 sigma0: float = 1.0, seed: Optional[int] = None):
        self.t_steps = max(1, int(t_steps))
        self.gamma = float(gamma)
        self.sigma0 = float(sigma0)
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board,
                    rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        rng = rng if rng is not None else self.rng
        probe = board.copy()
        moves = list(probe.legal_moves)
        if not moves:
            return None
        sign = 1 if board.turn == chess.WHITE else -1
        prior = evaluate(probe)
        raw = []
        for move in moves:
            probe.push(move)
            raw.append(sign * (evaluate(probe) - prior))
            probe.pop()
        mean = sum(raw) / len(raw)
        std = (sum((v - mean) ** 2 for v in raw) / len(raw)) ** 0.5 or 1.0
        mu = [(v - mean) / std for v in raw]
        x = mu[:]
        for i in range(self.t_steps):
            sigma = self.sigma0 * (1.0 - i / self.t_steps)
            for j in range(len(x)):
                x[j] += self.gamma * (mu[j] - x[j]) + sigma * rng.gauss(0.0, 1.0)
        best = max(range(len(x)), key=lambda j: x[j])
        self.last_nodes = len(moves)
        return moves[best]


class LayaPlayer(ChessPlayer):
    """Continuous dynamics: positional tension, king safety, center stability."""

    name = "laya"
    label = "LAYA // continuous dynamics"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board,
                    rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        rng = rng if rng is not None else self.rng
        probe = board.copy()
        moves = list(probe.legal_moves)
        if not moves:
            return None
        turn = board.turn
        sign = 1 if turn == chess.WHITE else -1
        eval_before = evaluate(probe)
        tension_before = _aggression(probe, turn) - _exposure(probe, turn)
        center_before = _center_control(probe, turn)
        king_before = _king_pressure(probe, turn)
        enemy_king_before = _king_pressure(probe, not turn)
        best_score, best_moves = -INF, []
        for move in moves:
            probe.push(move)
            tension = _aggression(probe, turn) - _exposure(probe, turn)
            score = sign * (evaluate(probe) - eval_before)
            score += 30 * (tension - tension_before)
            score += 10 * (_center_control(probe, turn) - center_before)
            score += 12 * (king_before - _king_pressure(probe, turn))
            score += 4 * (_king_pressure(probe, not turn) - enemy_king_before)
            probe.pop()
            if score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
        self.last_nodes = len(moves)
        return rng.choice(best_moves)


class SalesRLChessPlayer(ChessPlayer):
    """Pitches candidate moves as B2B SaaS deals to maximize P(conversion).

    engagement = mobility delta, effectiveness = material delta plus grabbed
    deals minus hanging exposure, churn risk = king safety pressure.  The
    winning pitch is kept in last_pitch for telemetry.
    """

    name = "sales_rl"
    label = "SALES // B2B SaaS conversion"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.last_pitch = None

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def choose_move(self, board: chess.Board,
                    rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        rng = rng if rng is not None else self.rng
        probe = board.copy()
        moves = list(probe.legal_moves)
        if not moves:
            return None
        turn = board.turn
        sign = 1 if turn == chess.WHITE else -1
        eval_before = evaluate(probe)
        mobility_before = _legal_move_count(probe, turn)
        best_p, best_moves = -INF, []
        best_z = best_eng = best_eff = best_churn = 0.0
        for move in moves:
            is_capture = probe.is_capture(move)
            if is_capture:
                if probe.is_en_passant(move):
                    vic = ORDER_SCALE[chess.PAWN]
                else:
                    victim = probe.piece_at(move.to_square)
                    vic = ORDER_SCALE[victim.piece_type] if victim else 0
            else:
                vic = 0
            probe.push(move)
            material = sign * (evaluate(probe) - eval_before) / 100.0
            engagement = _legal_move_count(probe, turn) - mobility_before
            hanging = _exposure(probe, turn)
            churn = _king_pressure(probe, turn) / 20.0
            effectiveness = material + 0.4 * vic - 0.3 * hanging
            z = 0.6 * engagement + 1.0 * effectiveness - 0.7 * churn
            p = self._sigmoid(z)
            probe.pop()
            if p > best_p:
                best_p, best_moves = p, [move]
                best_z, best_eng, best_eff, best_churn = (
                    z, engagement, effectiveness, churn)
            elif p == best_p:
                best_moves.append(move)

        move = rng.choice(best_moves)
        san = board.san(move)
        self.last_pitch = (
            f"Pitch {san} → eng {best_eng:+.1f} / eff {best_eff:+.2f} / "
            f"churn {best_churn:.2f} → P(convert) {best_p:.1%}"
        )
        self.last_nodes = len(moves)
        return move

    def reset(self) -> None:
        super().reset()
        self.last_pitch = None


class OpenJEVPlayer(ChessPlayer):
    """NLI joint-energy scorer: E(s') = alpha*Phi(s') + beta*theta(s').

    Phi is the resulting-board positional potential (material, center control,
    king-safety denial); theta is the premise -> hypothesis transition
    gradient.  The move minimizing the positional free energy is played.
    """

    name = "openjev"
    label = "OPENJEV // NLI energy"

    def __init__(self, alpha: float = 1.0, beta: float = 0.4,
                 seed: Optional[int] = None):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board,
                    rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        rng = rng if rng is not None else self.rng
        probe = board.copy()
        moves = list(probe.legal_moves)
        if not moves:
            return None
        turn = board.turn
        sign = 1 if turn == chess.WHITE else -1
        base = sign * evaluate(probe)
        base += 6 * _center_control(probe, turn) - 8 * _king_pressure(probe, turn)
        best_e, best_moves = INF, []
        for move in moves:
            probe.push(move)
            potential = sign * evaluate(probe)
            potential += 6 * _center_control(probe, turn)
            potential -= 8 * _king_pressure(probe, turn)
            energy = (self.alpha * -potential
                      + self.beta * -(potential - base))
            probe.pop()
            if energy < best_e:
                best_e, best_moves = energy, [move]
            elif energy == best_e:
                best_moves.append(move)
        self.last_nodes = len(moves)
        return rng.choice(best_moves)


class KEVPlayer(ChessPlayer):
    """Ultra-fast bitboard blitz: MVV-LVA captures, promotions, hangs, checks."""

    name = "kev"
    label = "KEV // bitboard blitz"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: chess.Board,
                    rng: Optional[random.Random] = None) -> Optional[chess.Move]:
        rng = rng if rng is not None else self.rng
        probe = board.copy()
        moves = list(probe.legal_moves)
        if not moves:
            return None
        enemy = not board.turn
        best_score, best_moves = -INF, []
        for move in moves:
            attacker = probe.piece_at(move.from_square)
            att_val = ORDER_SCALE[attacker.piece_type] if attacker else 0
            if probe.is_capture(move):
                if probe.is_en_passant(move):
                    vic_val = ORDER_SCALE[chess.PAWN]
                else:
                    victim = probe.piece_at(move.to_square)
                    vic_val = ORDER_SCALE[victim.piece_type] if victim else 0
            else:
                vic_val = 0
            tact = 0
            if move.promotion:
                tact += 900 + ORDER_SCALE[move.promotion]
            if vic_val:
                tact += 10 * vic_val - att_val
            was_hung = bool(probe.attackers(enemy, move.from_square))
            probe.push(move)
            is_hang = bool(probe.attackers(enemy, move.to_square))
            moved_val = ORDER_SCALE[probe.piece_at(move.to_square).piece_type]
            if is_hang:
                tact -= (2 if vic_val else 4) * moved_val
            if was_hung and not is_hang:
                tact += 2 * moved_val
            if probe.attackers(board.turn, probe.king(enemy)):
                tact += 120
            probe.pop()
            if tact > best_score:
                best_score, best_moves = tact, [move]
            elif tact == best_score:
                best_moves.append(move)
        self.last_nodes = len(moves)
        return rng.choice(best_moves)


@dataclass(frozen=True)
class PlayerContext:
    """Build context handed to player factories at construction time.

    Carrier for whatever a registerable player needs to come up: search depth,
    rng seed, the inference device, a weights/model cache directory, and the
    lane this instance will run in ("live" for the match lanes, "judge" for
    on-demand System-2 arbiters).  Factories keep lifting things from this
    context as LLM players land, so the registry signature does not keep churn.
    """

    depth: int = 3
    seed: Optional[int] = None
    device: str = ""
    model_dir: Optional[str] = None
    lane: str = "live"


@dataclass(frozen=True)
class PlayerSpec:
    """One registry entry: id, display, tier, hint, and a factory for instances.

    This is the single source of truth (Standard E).  Construction, labels,
    CLI argparse choices, and the web dropdown options are all derived from
    PLAYER_REGISTRY so nothing can drift out of sync.  Factories build from a
    PlayerContext (device / model dir / lane), never (depth, seed).
    """

    name: str
    label: str
    tier: str
    hint: str
    factory: Callable[[PlayerContext], ChessPlayer]


def get_player_spec(kind: str) -> PlayerSpec:
    spec = PLAYER_REGISTRY.get((kind or "").lower())
    if spec is None:
        raise ValueError(f"unknown player type {kind!r}; known: {sorted(PLAYER_REGISTRY)}")
    return spec


def default_device() -> str:
    return os.environ.get("OPENJEV_DEVICE", "cpu")


def make_player(kind: str, depth: int = 3, seed: Optional[int] = None,
                ctx: Optional[PlayerContext] = None) -> ChessPlayer:
    if ctx is None:
        ctx = PlayerContext(depth=depth, seed=seed, device=default_device())
    return get_player_spec(kind).factory(ctx)


def player_label(kind: str, depth: int = 3) -> str:
    return get_player_spec(kind).label.format(depth=depth)


PLAYER_REGISTRY: Dict[str, PlayerSpec] = {
    "random": PlayerSpec(
        name="random",
        label="RANDOM // chaos",
        tier="reflex",
        hint="Uniform random legal move",
        factory=lambda ctx: RandomPlayer(seed=ctx.seed),
    ),
    "heuristic": PlayerSpec(
        name="heuristic",
        label="HEURISTIC // static",
        tier="reflex",
        hint="Greedy maximizer of the static evaluation",
        factory=lambda ctx: HeuristicPlayer(seed=ctx.seed),
    ),
    "minimax": PlayerSpec(
        name="minimax",
        label="MINIMAX // αβ d{depth}",
        tier="blitz",
        hint="Alpha-beta with transposition table + quiescence",
        factory=lambda ctx: MinimaxPlayer(depth=ctx.depth, seed=ctx.seed),
    ),
    "djev": PlayerSpec(
        name="djev",
        label="DJEV // diffusion logit",
        tier="reflex",
        hint="OU Langevin score smoother over candidate moves",
        factory=lambda ctx: DJEVPlayer(seed=ctx.seed),
    ),
    "laya": PlayerSpec(
        name="laya",
        label="LAYA // continuous dynamics",
        tier="reflex",
        hint="Continuous positional stability & tactical sharpness",
        factory=lambda ctx: LayaPlayer(seed=ctx.seed),
    ),
    "sales_rl": PlayerSpec(
        name="sales_rl",
        label="SALES // B2B SaaS conversion",
        tier="reflex",
        hint="Pitches moves as enterprise deals to maximize P(conversion)",
        factory=lambda ctx: SalesRLChessPlayer(seed=ctx.seed),
    ),
    "openjev": PlayerSpec(
        name="openjev",
        label="OPENJEV // NLI energy",
        tier="system2",
        hint="Joint energy scoring board premise vs hypothesis transitions",
        factory=lambda ctx: OpenJEVPlayer(seed=ctx.seed),
    ),
    "kev": PlayerSpec(
        name="kev",
        label="KEV // bitboard blitz",
        tier="blitz",
        hint="Ultra-fast bitboard candidate search",
        factory=lambda ctx: KEVPlayer(seed=ctx.seed),
    ),
}

PLAYER_TYPES: List[str] = list(PLAYER_REGISTRY)


@dataclass
class _Decision:
    """A published inference task: snapshot + player + generation stamp.

    The worker fills in move/thinking_ms/nodes off-lock; the orchestrator only
    commits a decision whose gen still matches the live generation (a reset or
    config change invalidates in-flight work).
    """

    snap: chess.Board
    player: ChessPlayer
    color: str
    gen: int
    move: Optional[chess.Move] = None
    thinking_ms: float = 0.0
    nodes: int = field(default=0)
    ready: bool = False
    started: bool = False


class ChessGameManager:
    """Thread-safe match controller shared by the web arena and the CLI.

    Concurrency contract (Standard C): choose_move is NEVER run while holding
    the lock.  A dedicated worker serializes inference against a read-only
    snapshot; the orchestrator commits the published decision under the lock
    only when its generation matches the live one.  A 17s external judge can
    therefore run without freezing the web API.
    """

    def __init__(self, white_type: str = "minimax", black_type: str = "heuristic",
                 depth: int = 3, tempo: float = 0.25, seed: Optional[int] = None):
        self.cond = threading.Condition()
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
        self._pending: Optional[_Decision] = None
        self._gen = 0
        self._force = False
        self._commit_seq = 0
        self._rebuild_players()
        threading.Thread(target=self._orchestrator, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------------ setup
    def _rebuild_players(self) -> None:
        device = default_device()
        self.white = make_player(
            self.white_type,
            ctx=PlayerContext(depth=self.depth, seed=random.randrange(2 ** 31),
                              device=device, lane="live"))
        self.black = make_player(
            self.black_type,
            ctx=PlayerContext(depth=self.depth, seed=random.randrange(2 ** 31),
                              device=device, lane="live"))

    # --------------------------------------------------------------- game loop
    def _worker(self) -> None:
        """Decision thread: performs inference OFF the lock, then publishes.

        A single persistent worker serializes choose_move calls, so the shared
        self.rng stays single-writer and seed reproducibility is preserved.
        """
        while True:
            dec = None
            with self.cond:
                if self._pending is not None and not self._pending.started:
                    dec = self._pending
                    dec.started = True
                if dec is None:
                    self.cond.wait(timeout=0.05)
            if dec is None:
                continue
            start = time.monotonic()
            move = dec.player.choose_move(dec.snap, self.rng)
            thinking_ms = (time.monotonic() - start) * 1000.0
            dec.move = move
            dec.thinking_ms = thinking_ms
            dec.nodes = getattr(dec.player, "last_nodes", 0)
            with self.cond:
                dec.ready = True
                self.cond.notify_all()

    def _orchestrator(self) -> None:
        """Loop thread: periodically pumps the decision queue under the lock."""
        while True:
            time.sleep(0.01)
            with self.cond:
                self._pump_locked()

    def _pump_locked(self) -> None:
        """Commit ready decisions; launch the next inference when due.

        A completed decision is committed only if dec.gen == self._gen, so a
        reset / config change that bumped the generation discards stale output.
        """
        p = self._pending
        if p is not None and p.ready:
            self._pending = None
            if p.gen == self._gen:
                self._commit_locked(p)
            self.cond.notify_all()
        if self._pending is None and not self.game_over:
            due = self._force or (self.running
                                  and time.monotonic() - self._last_step >= self.tempo)
            if due:
                self._force = False
                self._launch_locked()

    def _launch_locked(self) -> None:
        turn = self.board.turn
        player = self.white if turn == chess.WHITE else self.black
        self._pending = _Decision(
            snap=self.board.copy(), player=player, color=color_key(turn),
            gen=self._gen,
        )
        self._last_step = time.monotonic()
        self.cond.notify_all()

    def _commit_locked(self, dec: _Decision) -> None:
        color = dec.color
        self._exec_count += 1
        self.think_ms[color] += dec.thinking_ms
        self.last_think_ms[color] = dec.thinking_ms
        nodes = dec.nodes
        move = dec.move

        if move is None:
            self._finish("no_legal_moves", "b" if color == "w" else "w")
            self.cond.notify_all()
            return

        eval_before = evaluate(self.board)
        san = self.board.san(move)
        self.board.push(move)
        self.move_log.append({
            "n": self._exec_count,
            "san": san,
            "uci": move.uci(),
            "color": color,
            "from": chess.square_name(move.from_square),
            "to": chess.square_name(move.to_square),
            "player": dec.player.name,
            "time_ms": round(dec.thinking_ms, 2),
            "nodes": nodes,
            "eval": eval_before,
        })
        self._gen += 1
        self._commit_seq += 1
        self._check_end()
        self.cond.notify_all()

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

    def step(self) -> None:
        """Force exactly one inference and block until it is committed.

        The API semantics preserved: no lock is held across choose_move, and
        this returns as soon as a move lands (or the game ends, or a reset
        cancels the pending decision).
        """
        with self.cond:
            if self.game_over:
                return
            self._force = True
            self.cond.notify_all()
            start_seq = self._commit_seq
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            with self.cond:
                if (self._commit_seq != start_seq
                        or self.game_over
                        or (self._pending is None and not self._force)):
                    return
                if self._pending is not None and self._pending.ready:
                    self._pump_locked()
                    continue
                self.cond.wait(timeout=0.25)

    def toggle(self) -> bool:
        with self.cond:
            self.running = not self.running
            self._last_step = time.monotonic()
            self.cond.notify_all()
            return self.running

    def reset(self) -> None:
        with self.cond:
            self._reset_locked()
            self.cond.notify_all()

    def _reset_locked(self) -> None:
        self._pending = None
        self._force = False
        self._gen += 1
        self.board = chess.Board()
        self.move_log.clear()
        self.think_ms = {"w": 0.0, "b": 0.0}
        self.last_think_ms = {"w": 0.0, "b": 0.0}
        self._exec_count = 0
        self.game_over = False
        self.result = None
        for player in (self.white, self.black):
            if player is not None:
                player.reset()

    def set_config(self, data: Dict) -> None:
        with self.cond:
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
            self.cond.notify_all()

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
        with self.cond:
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
            st = {
                "type": player.name,
                "label": player_label(player.name, self.depth),
                "tier": PLAYER_REGISTRY[player.name].tier,
                "thinking_ms": round(self.think_ms[key], 1),
                "last_think_ms": round(self.last_think_ms[key], 1),
                "nodes": getattr(player, "last_nodes", 0),
            }
            pitch = getattr(player, "last_pitch", None)
            if pitch is not None:
                st["pitch"] = pitch
            return st

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
                "roster": [
                    {"name": spec.name, "label": spec.label,
                     "tier": spec.tier, "hint": spec.hint}
                    for spec in PLAYER_REGISTRY.values()
                ],
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