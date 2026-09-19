"""Tetris game engine with 7-bag randomizer, collision detection, and Dellacherie features."""
import copy
import random
from typing import Dict, List, Optional, Tuple

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

# Standard 7 tetrominoes: [rotation][row][col]
# Defined in minimal bounding box
PIECES = {
    'I': [
        [[1, 1, 1, 1]],
        [[1], [1], [1], [1]]
    ],
    'O': [
        [[1, 1], [1, 1]]
    ],
    'T': [
        [[0, 1, 0], [1, 1, 1]],
        [[1, 0], [1, 1], [1, 0]],
        [[1, 1, 1], [0, 1, 0]],
        [[0, 1], [1, 1], [0, 1]]
    ],
    'S': [
        [[0, 1, 1], [1, 1, 0]],
        [[1, 0], [1, 1], [0, 1]]
    ],
    'Z': [
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1], [1, 1], [1, 0]]
    ],
    'J': [
        [[1, 0, 0], [1, 1, 1]],
        [[1, 1], [1, 0], [1, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[0, 1], [0, 1], [1, 1]]
    ],
    'L': [
        [[0, 0, 1], [1, 1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]]
    ]
}

from tetris_engine import (
    compute_bitboard_features, grid_to_bitboard, bitboard_to_grid,
    ModernTetris, BitBoard, SRS_PIECES_COORDS, SRS_KICKS_JLSTZ, SRS_KICKS_I
)

COLORS = {
    'I': '\033[96m',  # Cyan
    'O': '\033[93m',  # Yellow
    'T': '\033[95m',  # Magenta
    'S': '\033[92m',  # Green
    'Z': '\033[91m',  # Red
    'J': '\033[94m',  # Blue
    'L': '\033[33m',  # Orange
    'RESET': '\033[0m'
}


class Tetris:
    def __init__(self, seed: Optional[int] = None, max_steps: int = 1000):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        # 0 = empty, string (e.g. 'I', 'O', etc.) = occupied
        self.grid = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.bit_rows = [0] * BOARD_HEIGHT
        self.bag: List[str] = []
        self.current_piece = self._next_piece_from_bag()
        self.next_piece = self._next_piece_from_bag()
        self.hold_piece: Optional[str] = None
        self.can_hold = True
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.done = False
        return self.state()

    def _next_piece_from_bag(self) -> str:
        if not self.bag:
            self.bag = list(PIECES.keys())
            self.rng.shuffle(self.bag)
        return self.bag.pop(0)

    def state(self) -> Dict:
        return {
            "grid": [row[:] for row in self.grid],
            "current_piece": self.current_piece,
            "next_piece": self.next_piece,
            "hold_piece": self.hold_piece,
            "can_hold": self.can_hold,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "pieces_placed": self.pieces_placed,
            "done": self.done,
            "features": compute_board_features(self.grid)
        }

    def hold(self) -> bool:
        """Hold the current piece into the hold queue."""
        if self.done or not self.can_hold:
            return False
        if self.hold_piece is None:
            self.hold_piece = self.current_piece
            self.current_piece = self.next_piece
            self.next_piece = self._next_piece_from_bag()
        else:
            self.hold_piece, self.current_piece = self.current_piece, self.hold_piece
        self.can_hold = False
        if len(self.get_legal_moves()) == 0:
            self.done = True
        return True

    def get_legal_moves(self, piece: Optional[str] = None) -> List[Tuple[int, int]]:
        """Returns all valid (rotation, col) moves for the given piece."""
        p = piece or self.current_piece
        orientations = PIECES[p]
        moves = []
        for rot_idx, shape in enumerate(orientations):
            p_width = len(shape[0])
            for col in range(BOARD_WIDTH - p_width + 1):
                # Verify that the piece can enter the board at (col, 0)
                if not self._check_collision(shape, col, 0):
                    moves.append((rot_idx, col))
        return moves

    def _check_collision(self, shape: List[List[int]], col: int, row: int, grid=None) -> bool:
        if grid is not None:
            for r_idx, r in enumerate(shape):
                for c_idx, val in enumerate(r):
                    if val:
                        target_r = row + r_idx
                        target_c = col + c_idx
                        if target_r < 0 or target_r >= BOARD_HEIGHT or target_c < 0 or target_c >= BOARD_WIDTH:
                            return True
                        if grid[target_r][target_c] != 0:
                            return True
            return False

        # Fast bitboard collision
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
            if self.bit_rows[tr] & mask:
                return True
        return False

    def simulate_placement(self, rot: int, col: int, piece: Optional[str] = None) -> Tuple[List[List[int]], int, int]:
        """Simulates hard-dropping a piece at (rot, col).
        Returns: (new_grid, lines_cleared, landing_height)"""
        p = piece or self.current_piece
        shape = PIECES[p][rot]
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
                if self.bit_rows[drop_row + 1 + i] & shape_masks[i]:
                    collides = True
                    break
            if collides:
                break
            drop_row += 1

        landing_height = int(BOARD_HEIGHT - drop_row - (h / 2.0))

        # Build grid
        g = [row[:] for row in self.grid]
        for r_idx, r in enumerate(shape):
            for c_idx, val in enumerate(r):
                if val:
                    g[drop_row + r_idx][col + c_idx] = p

        # Clear complete rows
        new_grid = [row for row in g if any(c == 0 for c in row)]
        lines = BOARD_HEIGHT - len(new_grid)
        for _ in range(lines):
            new_grid.insert(0, [0 for _ in range(BOARD_WIDTH)])

        return new_grid, lines, landing_height

    def step(self, rot: int, col: int) -> Tuple[Dict, int, bool]:
        """Apply placement of current piece."""
        if self.done:
            return self.state(), 0, True

        shape = PIECES[self.current_piece][rot]
        if self._check_collision(shape, col, 0):
            self.done = True
            return self.state(), 0, True

        new_grid, lines, _ = self.simulate_placement(rot, col, self.current_piece)
        self.grid = new_grid
        self.bit_rows = grid_to_bitboard(new_grid)
        self.lines_cleared += lines
        self.pieces_placed += 1

        # Scoring standard: 1: 100, 2: 300, 3: 500, 4: 800
        line_rewards = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        reward = line_rewards.get(lines, 0) + 1  # survival bonus
        self.score += reward

        self.current_piece = self.next_piece
        self.next_piece = self._next_piece_from_bag()
        self.can_hold = True

        # Check if top row is reached or no legal moves
        if len(self.get_legal_moves()) == 0:
            self.done = True
        elif self.pieces_placed >= self.max_steps:
            self.done = True

        return self.state(), reward, self.done

    def render_ascii(self, highlight_col: Optional[int] = None) -> str:
        lines = []
        lines.append("+" + "--" * BOARD_WIDTH + "+")
        for r in range(BOARD_HEIGHT):
            row_str = "|"
            for c in range(BOARD_WIDTH):
                val = self.grid[r][c]
                if val != 0:
                    color = COLORS.get(val, '')
                    row_str += f"{color}[]{COLORS['RESET']}"
                elif highlight_col is not None and c == highlight_col:
                    row_str += "::"
                else:
                    row_str += " ."
            row_str += "|"
            lines.append(row_str)
        lines.append("+" + "--" * BOARD_WIDTH + "+")
        lines.append(f"Piece: {self.current_piece} | Next: {self.next_piece} | Hold: {self.hold_piece or '-'} | Lines: {self.lines_cleared} | Score: {self.score}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Board feature extraction & Dellacherie heuristic (Bitboard-Accelerated)
# -----------------------------------------------------------------------------
def get_column_heights(grid: List[List[int]]) -> List[int]:
    heights = [0] * BOARD_WIDTH
    for c in range(BOARD_WIDTH):
        for r in range(BOARD_HEIGHT):
            if grid[r][c] != 0:
                heights[c] = BOARD_HEIGHT - r
                break
    return heights


def compute_board_features(grid: List[List[int]]) -> Dict[str, float]:
    bit_rows = grid_to_bitboard(grid)
    return compute_bitboard_features(bit_rows)


def dellacherie_eval(grid: List[List[int]], lines_cleared: int, landing_height: int) -> float:
    """Dellacherie evaluation function."""
    feats = compute_board_features(grid)
    score = (
        -1.0 * landing_height
        + 1.0 * (lines_cleared * 4)
        - 1.0 * feats["row_transitions"]
        - 1.0 * feats["col_transitions"]
        - 4.0 * feats["holes"]
        - 1.0 * feats["wells"]
        - 0.5 * feats["bumpiness"]
    )
    return score


def tetris_policy_eval(grid: List[List[int]], lines_cleared: int, landing_height: int) -> float:
    """A compact board value used as a stronger distillation teacher.

    These weights reward cleared lines while strongly penalizing aggregate height and
    holes. They are intentionally separate from the original Dellacherie reference so
    old benchmark numbers remain comparable.
    """
    feats = compute_board_features(grid)
    return (
        0.760666 * lines_cleared
        - 0.510066 * feats["total_height"]
        - 0.356630 * feats["holes"]
        - 0.184483 * feats["bumpiness"]
        - 0.120000 * feats["wells"]
        - 0.080000 * landing_height
    )


def lookahead_policy(env: Tetris, gamma: float = 0.35) -> Tuple[int, int]:
    """Choose a placement with a one-piece lookahead teacher.

    The current outcome is scored and then augmented with the best outcome available
    for the known next piece. This gives the RoBERTa distillation target a little
    planning signal without making inference depend on a tree search.
    """
    legal_moves = env.get_legal_moves()
    if not legal_moves:
        return 0, 0

    best_move, best_score = legal_moves[0], -1e30
    for rot, col in legal_moves:
        grid, lines, landing_h = env.simulate_placement(rot, col)
        value = tetris_policy_eval(grid, lines, landing_h)

        future = Tetris(seed=0, max_steps=1)
        future.grid = [row[:] for row in grid]
        future.bit_rows = grid_to_bitboard(grid)
        future.current_piece = env.next_piece
        future.next_piece = env.next_piece
        next_best = -1e30
        for next_rot, next_col in future.get_legal_moves():
            next_grid, next_lines, next_height = future.simulate_placement(next_rot, next_col)
            next_best = max(next_best, tetris_policy_eval(next_grid, next_lines, next_height))
        if next_best > -1e29:
            value += gamma * next_best

        if value > best_score:
            best_score, best_move = value, (rot, col)
    return best_move


def lookahead_move_scores(env: Tetris, gamma: float = 0.35) -> List[Tuple[Tuple[int, int], float]]:
    """Return teacher values for every legal move, for listwise policy training."""
    legal_moves = env.get_legal_moves()
    scored = []
    for move in legal_moves:
        rot, col = move
        grid, lines, landing_h = env.simulate_placement(rot, col)
        value = tetris_policy_eval(grid, lines, landing_h)
        future = Tetris(seed=0, max_steps=1)
        future.grid = [row[:] for row in grid]
        future.bit_rows = grid_to_bitboard(grid)
        future.current_piece = env.next_piece
        future.next_piece = env.next_piece
        next_values = []
        for next_rot, next_col in future.get_legal_moves():
            ng, nl, nh = future.simulate_placement(next_rot, next_col)
            next_values.append(tetris_policy_eval(ng, nl, nh))
        if next_values:
            value += gamma * max(next_values)
        scored.append((move, value))
    return scored


def oracle_policy(env: Tetris) -> Tuple[int, int]:
    """Picks the move with highest Dellacherie heuristic score."""
    legal_moves = env.get_legal_moves()
    if not legal_moves:
        return 0, 0
    best_score = -1e9
    best_move = legal_moves[0]
    for rot, col in legal_moves:
        sim_grid, lines, landing_h = env.simulate_placement(rot, col)
        sc = dellacherie_eval(sim_grid, lines, landing_h)
        if sc > best_score:
            best_score = sc
            best_move = (rot, col)
    return best_move
