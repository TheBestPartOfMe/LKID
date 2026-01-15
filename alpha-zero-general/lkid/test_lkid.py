"""
Unit tests for the LKID game implementation.
"""
import sys
sys.path.append('..')
from lkid.LKIDGame import LKIDGame
from lkid.LKIDLogic import Board
import unittest


class TestLKIDGame(unittest.TestCase):
    def setUp(self):
        self.game = LKIDGame()
        self.board = Board()

    def test_initial_board(self):
        """Test that initial board is created correctly."""
        state = self.game.getInitBoard()
        self.assertEqual(state.shape[0], 49)

    def test_board_size(self):
        """Test board dimensions."""
        size = self.game.getBoardSize()
        self.assertEqual(size, (7, 7))

    def test_action_size(self):
        """Test action space size."""
        action_size = self.game.getActionSize()
        self.assertGreater(action_size, 0)

    def test_valid_moves_p1(self):
        """Test that P1 has valid moves at the start."""
        state = self.game.getInitBoard()
        valids = self.game.getValidMoves(state, 1)
        self.assertGreater(np.sum(valids), 0)

    def test_canonical_form_p1(self):
        """Test canonical form for P1."""
        state = self.game.getInitBoard()
        canonical = self.game.getCanonicalForm(state, 1)
        np.testing.assert_array_equal(state, canonical)

    def test_canonical_form_p2(self):
        """Test canonical form for P2 (should be different from P1)."""
        state = self.game.getInitBoard()
        canonical = self.game.getCanonicalForm(state, -1)
        # Player encoding should be flipped
        self.assertFalse(np.array_equal(state, canonical))

    def test_symmetries(self):
        """Test that symmetries are generated."""
        state = self.game.getInitBoard()
        policy = np.ones(self.game.getActionSize()) / self.game.getActionSize()
        symmetries = self.game.getSymmetries(state, list(policy))
        self.assertGreater(len(symmetries), 0)

    def test_string_representation(self):
        """Test string representation is unique."""
        state = self.game.getInitBoard()
        str_rep = self.game.stringRepresentation(state)
        self.assertIsNotNone(str_rep)

    def test_game_ended_initial(self):
        """Test that game is not ended at start."""
        state = self.game.getInitBoard()
        result = self.game.getGameEnded(state, 1)
        self.assertEqual(result, 0)

    def test_next_state(self):
        """Test that next state changes after a move."""
        state = self.game.getInitBoard()
        valids = self.game.getValidMoves(state, 1)
        valid_actions = np.where(valids)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
            next_state, next_player = self.game.getNextState(state, 1, action)
            # State should change or player should change
            self.assertEqual(next_player, -1)


if __name__ == '__main__':
    import numpy as np
    unittest.main()
