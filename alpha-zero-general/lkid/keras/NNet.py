import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet

# Import LKIDNNet - try absolute import first, fallback to relative
try:
    from lkid.keras.LKIDNNet import LKIDNNet as lkid_nnet
except ImportError:
    from .LKIDNNet import LKIDNNet as lkid_nnet

"""
NeuralNet wrapper class for the LKIDNNet.
Implements the NeuralNet interface using Keras/TensorFlow.
"""

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 8,
    'batch_size': 128,
    'cuda': False,
    'num_channels': 64,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.nnet = None
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def _ensure_model(self):
        """Lazily initialize the model on first use."""
        if self.nnet is None:
            self.nnet = lkid_nnet(self.game, args)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        self._ensure_model()
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

    def predict(self, board):
        """
        board: np array with board (flattened 5x5)
        """
        self._ensure_model()
        start = time.time()

        board = board[np.newaxis, :]

        pi, v = self.nnet.model.predict(board, verbose=False)

        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self._ensure_model()

        filename = filename.split(".")[0] + ".weights.h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self._ensure_model()

        filename = filename.split(".")[0] + ".weights.h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path {}".format(filepath))

        self.nnet.model.load_weights(filepath)
