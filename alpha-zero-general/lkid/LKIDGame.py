"""
Game class implementation for Lass Die Kirche Im Dorf (LKID).

Board representation:
- Flattened 5x5 board with piece information encoded
- Position encoding: x*5 + y (0-24)

The board state is a numpy array that encodes:
- Piece presence and ownership for each cell
- Orientation information for each piece
"""
from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .LKIDLogic import Board
import numpy as np

# Reference to self for static methods
import lkid.LKIDGame as lkid_module


class LKIDGame(Game):

    def __init__(self):
        self.n = 5
        self.board_size = self.n * self.n
        self.action_space_size = self.board_size * self.board_size

    def getInitBoard(self):
        """
        Return initial board state as a numpy array.
        Board shape: (25,) - flattened 5x5 board
        Each position encodes: owner (3 bits) + piece_type (3 bits) + orientation (1 bit)
        """
        board = self._create_initial_board()
        return self._board_to_state(board)

    def _create_initial_board(self):
        """Create and return the initial board configuration with random start positions."""
        import random
        board = Board()

        # List of (p1_state, p2_state, priest_pos) tuples
        start_positions = [
            # --- Standard ---
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

            # --- Variant 1 ---
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

            # --- Variant 2 ---
            ([ (0, 0, Board.CHURCH_TOWER, Board.HORIZONTAL),
                (4, 4, Board.CHURCH_SHIP, Board.VERTICAL),
                (2, 0, Board.HOUSE, Board.HORIZONTAL),
                (2, 1, Board.HOUSE, Board.VERTICAL),
            ],
             [ (4, 0, Board.CHURCH_TOWER, Board.HORIZONTAL),
                (0, 4, Board.CHURCH_SHIP, Board.VERTICAL),
                (2, 4, Board.HOUSE, Board.HORIZONTAL),
                (2, 3, Board.HOUSE, Board.VERTICAL),
            ],
             (2, 2)),

            # --- Variant 3 ---
            ([ (0, 0, Board.CHURCH_TOWER, Board.HORIZONTAL),
                (4, 4, Board.CHURCH_SHIP, Board.HORIZONTAL),
                (0, 2, Board.HOUSE, Board.VERTICAL),
                (4, 2, Board.HOUSE, Board.VERTICAL),
            ],
             [ (4, 0, Board.CHURCH_TOWER, Board.HORIZONTAL),
                (0, 4, Board.CHURCH_SHIP, Board.HORIZONTAL),
                (2, 0, Board.HOUSE, Board.HORIZONTAL),
                (2, 4, Board.HOUSE, Board.HORIZONTAL),
            ],
             (2, 2)),

            # --- Variant 4 ---
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

            # --- Variant 5 ---
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

            # --- Variant 6 ---
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

        # Randomly select one starting position
        p1_state, p2_state, priest_pos = random.choice(start_positions)
        board.setup_board(p1_state, p2_state, priest_pos)
        return board

    def _board_to_state(self, board):
        """Convert Board object to numpy array state."""
        state = np.zeros(self.board_size, dtype=np.int32)
        
        for x in range(self.n):
            for y in range(self.n):
                owner, piece_type, orientation = board._get_piece(x, y)
                idx = x * self.n + y
                
                # Encode: owner (2 bits) + piece_type (3 bits) + orientation (1 bit)
                # owner: 0=empty, 1=P1, 2=P2, 3=unused
                owner_encoded = 0 if owner == 0 else (1 if owner == 1 else 2)
                value = (owner_encoded << 4) | (piece_type << 1) | (orientation if orientation is not None else 0)
                state[idx] = value
        
        return state

    def _state_to_board(self, state):
        """Convert numpy array state to Board object."""
        board = Board()
        board._init_empty_board()
        
        for x in range(self.n):
            for y in range(self.n):
                idx = x * self.n + y
                value = state[idx]
                
                owner_encoded = (value >> 4) & 0x3
                piece_type = (value >> 1) & 0x7
                orientation = value & 0x1 if piece_type != Board.EMPTY and piece_type != Board.PRIEST else None
                
                owner = 0 if owner_encoded == 0 else (1 if owner_encoded == 1 else -1)
                
                if piece_type != Board.EMPTY:
                    board._set_piece(x, y, owner, piece_type, orientation)
        
        return board

    def getBoardSize(self):
        """Return board dimensions."""
        return (self.n, self.n)

    def getActionSize(self):
        """
        Return the total number of possible actions.
        Actions: from_cell * board_size + to_cell, plus pass action.
        """
        return self.action_space_size

    def _move_to_action(self, move):
        """Convert a move tuple to an action index."""
        from_x, from_y, to_x, to_y, orientation = move
        from_idx = from_x * self.n + from_y
        to_idx = to_x * self.n + to_y
        return from_idx * self.board_size + to_idx

    def _action_to_move(self, action, board, player):
        """Convert an action index to a move tuple."""
        from_idx = action // self.board_size
        to_idx = action % self.board_size
        
        from_x, from_y = from_idx // self.n, from_idx % self.n
        to_x, to_y = to_idx // self.n, to_idx % self.n
        
        owner, piece_type, orientation = board._get_piece(from_x, from_y)
        if owner != player or piece_type == Board.EMPTY:
            return None
        
        return (from_x, from_y, to_x, to_y, orientation)

    def getNextState(self, state, player, action):
        """
        Execute an action and return the next state and player.
        
        Args:
            state: current board state (numpy array)
            player: current player (1 or -1)
            action: action index
        
        Returns:
            (next_state, next_player)
        """
        board = self._state_to_board(state)
        move = self._action_to_move(action, board, player)
        
        if move is not None:
            try:
                board.execute_move(move, player)
            except:
                # Invalid move, return same state
                pass
        
        next_state = self._board_to_state(board)
        return (next_state, -player)

    def getValidMoves(self, state, player):
        """
        Return a binary vector of valid moves.
        
        Args:
            state: board state (numpy array)
            player: current player (1 or -1)
        
        Returns:
            numpy array of length getActionSize() with 1s for valid moves
        """
        board = self._state_to_board(state)
        valid_moves = [0] * self.getActionSize()
        
        legal_moves = board.get_legal_moves(player)
        
        for move in legal_moves:
            action = self._move_to_action(move)
            if 0 <= action < len(valid_moves):
                valid_moves[action] = 1
        
        return np.array(valid_moves)

    def getGameEnded(self, state, player):
        """
        Check if the game has ended.
        
        Args:
            state: board state (numpy array)
            player: current player (1 or -1)
        
        Returns:
            0 if game ongoing
            1 if player has won
            -1 if player has lost
        """
        board = self._state_to_board(state)
        
        # Check win condition for current player
        if board.check_win_condition(player):
            return 1
        
        # Check win condition for opponent
        if board.check_win_condition(-player):
            return -1
        
        # Game is ongoing
        return 0

    def _get_church_positions(self, board, player):
        """Get positions of Church Tower and Church Ship for a player."""
        tower_pos = None
        ship_pos = None
        
        for x in range(self.n):
            for y in range(self.n):
                owner, piece_type, _ = board._get_piece(x, y)
                if owner != player:
                    continue
                if piece_type == Board.CHURCH_TOWER:
                    tower_pos = (x, y)
                elif piece_type == Board.CHURCH_SHIP:
                    ship_pos = (x, y)
        
        return tower_pos, ship_pos

    def getCanonicalForm(self, state, player):
        """
        Return the canonical form of the state.
        For player 1, return state as-is.
        For player -1, flip all player ownership so current player appears as player 1.
        """
        if player == 1:
            return state
        
        canonical = np.copy(state)
        for i in range(len(state)):
            if state[i] != 0:
                owner_encoded = (state[i] >> 4) & 0x3
                piece_type = (state[i] >> 1) & 0x7
                orientation = state[i] & 0x1
                
                if owner_encoded == 1:
                    owner_encoded = 2
                elif owner_encoded == 2:
                    owner_encoded = 1
                
                canonical[i] = (owner_encoded << 4) | (piece_type << 1) | orientation
        
        return canonical

    def getSymmetries(self, state, pi):
        assert len(pi) == self.getActionSize()
        l = []
        l.append((state, pi))
        return l

    def stringRepresentation(self, state):
        """Return a unique string representation of the state."""
        return state.tobytes()

    @staticmethod
    def display(state):
        """Display the board state in a human-readable format."""
        game = LKIDGame()
        board = game._state_to_board(state)
        
        print("  ", end="")
        for y in range(board.n):
            print(f"{y} ", end="")
        print()
        
        for x in range(board.n):
            print(f"{x} ", end="")
            for y in range(board.n):
                owner, piece_type, orientation = board._get_piece(x, y)
                
                if piece_type == Board.EMPTY:
                    print(". ", end="")
                elif piece_type == Board.PRIEST:
                    print("P ", end="")
                elif piece_type == Board.CHURCH_TOWER:
                    print("T" if owner == 1 else "t", end=" ")
                elif piece_type == Board.CHURCH_SHIP:
                    print("S" if owner == 1 else "s", end=" ")
                elif piece_type == Board.HOUSE:
                    print("H" if owner == 1 else "h", end=" ")
                else:
                    print("? ", end="")
            print()
