"""Regression checks for the live board passed to AI players."""
import unittest

from coop_tetris import AIPlayerBoard, CoopGameManager, SynchronizedBag
from tetris_engine import grid_to_bitboard


class InspectingPlayer:
    def __init__(self):
        self.checked = False

    def choose_move(self, env):
        assert env.bit_rows == grid_to_bitboard(env.grid)
        assert env.bit_rows[-1] != 0
        self.checked = True
        return env.get_legal_moves()[0], {}


class TestCoopLogic(unittest.TestCase):
    def test_ai_hold_empty_and_swap(self):
        bag = SynchronizedBag()
        bag.sequence[:3] = ['S', 'O', 'I']
        board = AIPlayerBoard(bag)
        self.assertTrue(board.hold())
        self.assertEqual((board.hold_piece, board.cur_piece, board.piece_idx), ('S', 'O', 1))
        self.assertFalse(board.hold())
        board.step(*board.get_legal_moves()[0])
        self.assertTrue(board.can_hold)
        self.assertTrue(board.hold())
        self.assertEqual((board.hold_piece, board.cur_piece, board.piece_idx), ('I', 'S', 2))

    def test_heuristic_uses_hold_when_it_scores_better(self):
        manager = CoopGameManager(with_heuristic=True)
        slot = manager.slots[0]
        manager.bag.sequence[:3] = ['S', 'O', 'I']
        slot.board.reset()
        manager._run_single_ai_step(slot)
        self.assertEqual(slot.board.hold_piece, 'S')
        self.assertTrue(slot.board.last_step_info['used_hold'])

    def test_speed_change_updates_existing_slots(self):
        manager = CoopGameManager(with_heuristic=True)
        manager.set_ai_tempo(0.01)
        self.assertEqual(manager.ai_tempo_sec, 0.01)
        self.assertTrue(all(slot.tempo == 0.01 for slot in manager.slots))

    def test_top_row_cell_does_not_end_game_when_spawn_is_open(self):
        board = AIPlayerBoard(SynchronizedBag())
        board.board.rows[0] = 1
        board.board.color_grid[0][0] = 'J'
        move = next(move for move in board.get_legal_moves() if move[1] >= 4)
        board.step(*move)
        self.assertFalse(board.done)

    def test_ai_receives_occupied_collision_board(self):
        player = InspectingPlayer()
        manager = CoopGameManager(ai_player=player)
        slot = manager.slots[0]
        slot.board.board.rows[-1] = 1
        slot.board.board.color_grid[-1][0] = 'I'
        manager._run_single_ai_step(slot)
        self.assertTrue(player.checked)
        self.assertEqual(slot.board.pieces_placed, 1)


if __name__ == '__main__':
    unittest.main()
