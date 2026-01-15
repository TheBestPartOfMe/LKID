"""
Pit script for LKID (Lass Die Kirche Im Dorf).

Use this script to play any two agents against each other, or play manually with
any agent.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Arena
from MCTS import MCTS
from lkid.LKIDGame import LKIDGame
from lkid.LKIDPlayers import RandomPlayer, GreedyLKIDPlayer, HumanLKIDPlayer
from lkid.keras.NNet import NNetWrapper as NNet
import numpy as np
from utils import dotdict


# Config
human_vs_cpu = True

# Initialize game
g = LKIDGame()

# All players
rp = RandomPlayer(g).play
gp = GreedyLKIDPlayer(g).play
hp = HumanLKIDPlayer(g).play

# Neural network player 1
n1 = NNet(g)
try:
    n1.load_checkpoint('./temp/', 'best.pth.tar')
    print("Loaded best model for player 1")
except:
    print("Warning: Could not load best model for player 1, using untrained network")

args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    # Neural network player 2
    n2 = NNet(g)
    try:
        n2.load_checkpoint('./temp/', 'best.pth.tar')
        print("Loaded best model for player 2")
    except:
        print("Warning: Could not load best model for player 2, using untrained network")
    
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    player2 = n2p

# Create arena and play games
arena = Arena.Arena(n1p, player2, g, display=LKIDGame.display)

print("\n" + "="*50)
print("LKID Arena - 7x7 Board")
print("="*50)
print(f"Player 1: Neural Network (MCTS)")
print(f"Player 2: {'Human' if human_vs_cpu else 'Neural Network (MCTS)'}")
print("="*50 + "\n")

print(arena.playGames(2, verbose=True))
