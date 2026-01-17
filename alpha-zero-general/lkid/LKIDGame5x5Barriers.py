"""Barrier-heavy 5x5 variant of Lass Die Kirche Im Dorf (LKID).

This variant uses the same move rules as the standard 5x5 implementation but
adds four permanent barrier cells (0,2), (2,0), (2,4), and (4,2). Barriers have
no owner, cannot be moved, and block pieces movement.
"""
from __future__ import print_function
import random
import sys

sys.path.append('..')

from .LKIDLogic import Board
from .LKIDGame5x5 import LKIDGame as BaseLKID5x5


class LKIDGame5x5Barriers(BaseLKID5x5):
    """Special 5x5 LKID variant that injects four immutable barrier tiles."""

    DEFAULT_BARRIERS = {(0, 2), (2, 0), (2, 4), (4, 2)}

    def __init__(self):
        super().__init__()
        self.barriers = self.DEFAULT_BARRIERS

    def _create_initial_board(self):
        board = Board(n=self.n, barriers=self.barriers)

        start_positions = [
            ([ (0, 0, Board.CHURCH_TOWER, Board.HORIZONTAL),
                (4, 4, Board.CHURCH_SHIP, Board.HORIZONTAL),
                (1, 1, Board.HOUSE, Board.HORIZONTAL),
                (3, 1, Board.HOUSE, Board.VERTICAL),
            ],
             [ (4, 0, Board.CHURCH_TOWER, Board.HORIZONTAL),
                (0, 4, Board.CHURCH_SHIP, Board.HORIZONTAL),
                (1, 0, Board.HOUSE, Board.HORIZONTAL),
                (3, 0, Board.HOUSE, Board.VERTICAL),
            ],
             (2, 2)),
            ([ (0, 0, Board.CHURCH_TOWER, Board.VERTICAL),
                (4, 4, Board.CHURCH_SHIP, Board.HORIZONTAL),
                (1, 3, Board.HOUSE, Board.VERTICAL),
                (3, 3, Board.HOUSE, Board.HORIZONTAL),
            ],
             [ (4, 0, Board.CHURCH_TOWER, Board.VERTICAL),
                (0, 4, Board.CHURCH_SHIP, Board.HORIZONTAL),
                (1, 2, Board.HOUSE, Board.HORIZONTAL),
                (3, 2, Board.HOUSE, Board.VERTICAL),
            ],
             (2, 2)),
            ([ (0, 0, Board.CHURCH_TOWER, Board.VERTICAL),
                (4, 4, Board.CHURCH_SHIP, Board.VERTICAL),
                (1, 1, Board.HOUSE, Board.HORIZONTAL),
                (3, 3, Board.HOUSE, Board.VERTICAL),
            ],
             [ (4, 0, Board.CHURCH_TOWER, Board.VERTICAL),
                (0, 4, Board.CHURCH_SHIP, Board.VERTICAL),
                (1, 3, Board.HOUSE, Board.HORIZONTAL),
                (3, 1, Board.HOUSE, Board.VERTICAL),
            ],
             (2, 2)),
            ([ (0, 0, Board.CHURCH_TOWER, Board.HORIZONTAL),
                (4, 4, Board.CHURCH_SHIP, Board.VERTICAL),
                (0, 1, Board.HOUSE, Board.HORIZONTAL),
                (4, 3, Board.HOUSE, Board.VERTICAL),
            ],
             [ (4, 0, Board.CHURCH_TOWER, Board.HORIZONTAL),
                (0, 4, Board.CHURCH_SHIP, Board.VERTICAL),
                (1, 4, Board.HOUSE, Board.HORIZONTAL),
                (3, 0, Board.HOUSE, Board.VERTICAL),
            ],
             (2, 2)),
            ([ (0, 0, Board.CHURCH_TOWER, Board.VERTICAL),
                (4, 4, Board.CHURCH_SHIP, Board.HORIZONTAL),
                (2, 1, Board.HOUSE, Board.HORIZONTAL),
                (2, 3, Board.HOUSE, Board.VERTICAL),
            ],
             [ (4, 0, Board.CHURCH_TOWER, Board.VERTICAL),
                (0, 4, Board.CHURCH_SHIP, Board.HORIZONTAL),
                (1, 2, Board.HOUSE, Board.HORIZONTAL),
                (3, 2, Board.HOUSE, Board.VERTICAL),
            ],
             (2, 2)),
        ]

        p1_state, p2_state, priest_pos = random.choice(start_positions)
        board.setup_board(p1_state, p2_state, priest_pos)
        return board