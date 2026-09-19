"""Comprehensive unit and integration tests for ModernTetris engine:
1. SRS rotation states and all 8 official kick transitions for JLSTZ, I, and O.
2. Wall kicks, floor kicks, and block climbing.
3. HOLD functionality: initial hold, swap hold, lockout until lock, state preservation.
4. Lock-delay state machine: grounded transitions, timer expiration, reset limit, hard drop bypass, soft drop landing.
5. Deterministic replay: same seed + same input sequence = bit-for-bit identical board hash at every step.
"""
import unittest
from tetris_engine import (
    ModernTetris, BitBoard, SRS_PIECES_COORDS, SRS_KICKS_JLSTZ, SRS_KICKS_I,
    compute_bitboard_features, grid_to_bitboard, bitboard_to_grid,
    BOARD_WIDTH, BOARD_HEIGHT
)
from tetris_env import compute_board_features


class TestModernTetrisEngine(unittest.TestCase):

    def test_feature_extraction_parity(self):
        """Bitboard features must match baseline compute_board_features 100%."""
        import random
        rng = random.Random(1337)
        for _ in range(50):
            grid = [[1 if rng.random() > 0.65 else 0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
            base_f = compute_board_features(grid)
            bit_rows = grid_to_bitboard(grid)
            fast_f = compute_bitboard_features(bit_rows)
            for k in base_f:
                self.assertEqual(base_f[k], fast_f[k], f"Feature {k} mismatch: {base_f[k]} vs {fast_f[k]}")

    def test_srs_kick_tables_integrity(self):
        """Verify official SRS kick tables contain 5 tests per transition for JLSTZ and I."""
        expected_transitions = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 0), (0, 3)]
        for trans in expected_transitions:
            self.assertIn(trans, SRS_KICKS_JLSTZ)
            self.assertEqual(len(SRS_KICKS_JLSTZ[trans]), 5)
            self.assertEqual(SRS_KICKS_JLSTZ[trans][0], (0, 0))  # Test 1 is basic rotation

            self.assertIn(trans, SRS_KICKS_I)
            self.assertEqual(len(SRS_KICKS_I[trans]), 5)
            self.assertEqual(SRS_KICKS_I[trans][0], (0, 0))

    def test_srs_wall_kick(self):
        """Verify T-piece rotates near both left and right walls via kicks."""
        # 1. Right wall kick (State L -> 0 kicks left)
        tet = ModernTetris(seed=42)
        tet.current_piece = 'T'
        tet.cur_rot = 3  # State L (uses cols 0 and 1)
        tet.cur_x = 8    # Flush against right wall (cols 8 and 9)
        tet.cur_y = 10
        rotated = tet.rotate_cw()  # 3 -> 0 kicks left by 1 to col 7
        self.assertTrue(rotated)
        self.assertEqual(tet.cur_rot, 0)
        self.assertEqual(tet.cur_x, 7)

        # 2. Left wall kick (State R -> 0 kicks right)
        tet.cur_rot = 1  # State R (uses cols 1 and 2)
        tet.cur_x = -1   # Flush against left wall (cols 0 and 1)
        rotated2 = tet.rotate_ccw()  # 1 -> 0 kicks right by 1 to col 0
        self.assertTrue(rotated2)
        self.assertEqual(tet.cur_rot, 0)
        self.assertEqual(tet.cur_x, 0)

    def test_srs_floor_kick_and_climbing(self):
        """Verify T-piece on the floor kicks upward to rotate."""
        tet = ModernTetris(seed=42)
        tet.current_piece = 'T'
        tet.cur_rot = 0
        tet.cur_x = 4
        tet.cur_y = 18  # Row 19 is floor

        # Rotating 0 -> 1 would put the right block at row 20 (below floor).
        # SRS kicks it up!
        rotated = tet.rotate_cw()
        self.assertTrue(rotated)
        self.assertEqual(tet.cur_rot, 1)
        # Verify no block exceeds floor
        for r, c in SRS_PIECES_COORDS['T'][tet.cur_rot]:
            self.assertLess(tet.cur_y + r, BOARD_HEIGHT)

    def test_srs_i_piece_floor_kick(self):
        """Verify I-piece kicks horizontally and vertically."""
        tet = ModernTetris(seed=42)
        tet.current_piece = 'I'
        tet.cur_rot = 1  # Vertical
        tet.cur_x = 4
        tet.cur_y = 16   # Bottom reaches row 19

        # Rotate 1 -> 0 (horizontal) on floor
        rotated = tet.rotate_ccw()
        self.assertTrue(rotated)
        self.assertEqual(tet.cur_rot, 0)
        for r, c in SRS_PIECES_COORDS['I'][tet.cur_rot]:
            self.assertLess(tet.cur_y + r, BOARD_HEIGHT)
            self.assertGreaterEqual(tet.cur_x + c, 0)
            self.assertLess(tet.cur_x + c, BOARD_WIDTH)

    def test_hold_functionality(self):
        """Test initial hold, swap hold, and single-hold lockout per piece."""
        tet = ModernTetris(seed=100)
        p1 = tet.current_piece
        p2 = tet.next_piece

        # 1. First hold: stores p1, spawns p2
        res = tet.hold()
        self.assertTrue(res)
        self.assertEqual(tet.hold_piece, p1)
        self.assertEqual(tet.current_piece, p2)
        self.assertFalse(tet.can_hold)

        # 2. Cannot hold again before locking
        res2 = tet.hold()
        self.assertFalse(res2)
        self.assertEqual(tet.hold_piece, p1)

        # 3. Lock piece -> hold unlocks
        tet.hard_drop()
        self.assertTrue(tet.can_hold)

        # 4. Swap hold: active piece swapped with held piece
        active_before = tet.current_piece
        held_before = tet.hold_piece
        res3 = tet.hold()
        self.assertTrue(res3)
        self.assertEqual(tet.current_piece, held_before)
        self.assertEqual(tet.hold_piece, active_before)
        self.assertFalse(tet.can_hold)

    def test_lock_delay_state_machine(self):
        """Test grounded detection, timer countdown, move resets, and hard drop bypass."""
        tet = ModernTetris(seed=200, lock_delay=0.5, max_lock_resets=15)
        # Drop until grounded
        tet.cur_y = tet.board.get_drop_y(tet.current_piece, tet.cur_rot, tet.cur_x, tet.cur_y)
        # Tick small dt
        tet.tick(0.1)
        self.assertTrue(tet.is_grounded)
        self.assertAlmostEqual(tet.lock_timer, 0.4, places=2)

        # Lateral move resets timer
        tet.move_left()
        self.assertAlmostEqual(tet.lock_timer, 0.5, places=2)
        self.assertEqual(tet.lock_resets, 1)

        # Exhausting resets (15 resets)
        for _ in range(14):
            tet.move_right() if tet.cur_x < 5 else tet.move_left()
        self.assertEqual(tet.lock_resets, 15)

        # Next move should NOT reset timer
        tet.tick(0.2)
        initial_timer = tet.lock_timer
        tet.move_right() if tet.cur_x < 5 else tet.move_left()
        self.assertEqual(tet.lock_resets, 15)
        self.assertAlmostEqual(tet.lock_timer, initial_timer, places=2)

        # Letting timer expire locks piece
        pieces_before = tet.pieces_placed
        tet.tick(0.4)
        self.assertEqual(tet.pieces_placed, pieces_before + 1)
        self.assertEqual(tet.lock_resets, 0)
        self.assertTrue(tet.can_hold)

    def test_hard_drop_bypasses_lock_delay(self):
        tet = ModernTetris(seed=300)
        pieces_before = tet.pieces_placed
        tet.hard_drop()
        self.assertEqual(tet.pieces_placed, pieces_before + 1)

    def test_soft_drop_respects_lock_delay(self):
        tet = ModernTetris(seed=400, lock_delay=0.5)
        # Soft drop to ground
        while not tet.is_grounded:
            tet.soft_drop()
        pieces_before = tet.pieces_placed
        # Another soft drop on ground does not instantly lock
        tet.soft_drop()
        self.assertEqual(tet.pieces_placed, pieces_before)
        self.assertTrue(tet.is_grounded)

    def test_deterministic_replay(self):
        """Verify identical seed + identical action sequence produces identical state hash at every step."""
        def run_simulation(seed):
            tet = ModernTetris(seed=seed)
            hashes = []
            for _ in range(30):
                tet.move_left()
                tet.rotate_cw()
                tet.move_right()
                tet.soft_drop()
                hashes.append(tet.state_hash())
                tet.hard_drop()
                hashes.append(tet.state_hash())
            return hashes

        run1 = run_simulation(9999)
        run2 = run_simulation(9999)
        self.assertEqual(run1, run2)
        self.assertEqual(len(run1), 60)


if __name__ == "__main__":
    unittest.main()
