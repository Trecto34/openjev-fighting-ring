"""Unit and integration tests for Tetris environment and prompt generation."""
import unittest
from tetris_env import Tetris, compute_board_features, dellacherie_eval, oracle_policy, BOARD_WIDTH, BOARD_HEIGHT
from tetris_prompts import PROMPT_STRATEGIES, generate_candidate_pairs, build_pair, get_move_descriptors


class TestTetrisPipeline(unittest.TestCase):
    def test_environment_initialization(self):
        env = Tetris(seed=123)
        state = env.state()
        self.assertEqual(len(state["grid"]), BOARD_HEIGHT)
        self.assertEqual(len(state["grid"][0]), BOARD_WIDTH)
        self.assertIn(state["current_piece"], ["I", "O", "T", "S", "Z", "J", "L"])
        self.assertFalse(state["done"])

    def test_legal_moves(self):
        env = Tetris(seed=123)
        legal_moves = env.get_legal_moves()
        self.assertGreater(len(legal_moves), 0)
        for rot, col in legal_moves:
            self.assertGreaterEqual(rot, 0)
            self.assertGreaterEqual(col, 0)
            self.assertLess(col, BOARD_WIDTH)

    def test_simulation(self):
        env = Tetris(seed=123)
        legal_moves = env.get_legal_moves()
        rot, col = legal_moves[0]
        sim_grid, lines, landing_h = env.simulate_placement(rot, col)
        self.assertEqual(len(sim_grid), BOARD_HEIGHT)
        self.assertGreaterEqual(lines, 0)
        self.assertGreater(landing_h, 0)

    def test_dellacherie_oracle(self):
        env = Tetris(seed=42)
        move = oracle_policy(env)
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        state, reward, done = env.step(move[0], move[1])
        self.assertFalse(done)
        self.assertEqual(env.pieces_placed, 1)

    def test_prompt_generation(self):
        env = Tetris(seed=42)
        legal_moves = env.get_legal_moves()
        for strat in PROMPT_STRATEGIES:
            pairs = generate_candidate_pairs(env, legal_moves, strategy=strat)
            self.assertEqual(len(pairs), len(legal_moves))
            for premise, hyp in pairs:
                self.assertIsInstance(premise, str)
                self.assertIsInstance(hyp, str)
                self.assertGreater(len(premise), 10)
                # Verify key tokens
                self.assertIn("Tetris", premise)

    def test_kev_player(self):
        from kev_player import KevPlayer
        env = Tetris(seed=42)
        player = KevPlayer(mode="choice", top_k=4)
        move, info = player.choose_move(env)
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        self.assertIn(move, env.get_legal_moves())
        self.assertIn("confidence", info)
        self.assertIn("cand_scores", info)

    def test_laya_player(self):
        from laya_player import LayaPlayer
        env = Tetris(seed=42)
        player = LayaPlayer(top_k=4)
        move, info = player.choose_move(env)
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        self.assertIn(move, env.get_legal_moves())
        self.assertIn("confidence", info)
        self.assertIn("act_prob", info)


if __name__ == "__main__":
    unittest.main()
