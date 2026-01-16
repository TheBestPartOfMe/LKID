
import sys
import os
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lkid.LKIDGame import LKIDGame

if __name__ == "__main__":
    game = LKIDGame()
    state = game.getInitBoard()
    player = 1
    move_count = 0

    while True:
        valid_moves = game.getValidMoves(state, player)
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) == 0:
            print(f"No valid moves for player {player}. Game over.")
            break
        action = random.choice(valid_indices)
        state, player = game.getNextState(state, player, action)
        move_count += 1
        ended = game.getGameEnded(state, player)
        if ended != 0:
            print(f"Game ended. Winner: {'Player 1' if ended == 1 else 'Player 2'} after {move_count} moves.")
            break
        if(move_count % 100 == 0):
            print(f"Moves played: {move_count}")
    print(f"Total moves played: {move_count}")
