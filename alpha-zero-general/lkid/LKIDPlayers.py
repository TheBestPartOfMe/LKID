"""
Player implementations for LKID game.
Includes random, human, and greedy players.
"""
import sys
sys.path.append('..')
from lkid.LKIDGame import LKIDGame
import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        """Return a random valid move."""
        valids = self.game.getValidMoves(board, 1)
        move = np.random.choice(np.where(valids)[0])
        return move

class HumanLKIDPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        """Allow a human player to input a move."""
        self.game.display(board)
        
        while True:
            try:
                from_x = int(input("From X (0-6): "))
                from_y = int(input("From Y (0-6): "))
                to_x = int(input("To X (0-6): "))
                to_y = int(input("To Y (0-6): "))
                
                move_idx = from_x * 7 * 49 + from_y * 49 + to_x * 7 + to_y
                
                valids = self.game.getValidMoves(board, 1)
                if valids[move_idx]:
                    return move_idx
                else:
                    print("Invalid move, try again.")
            except (ValueError, IndexError):
                print("Invalid input, try again.")
