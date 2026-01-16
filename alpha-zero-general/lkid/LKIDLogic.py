"""
Board class for the game of Lass Die Kirche Im Dorf (LKID).
5x5 board with pieces that have orientation (horizontal/vertical).

Board representation:
- Each cell contains: (owner, piece_type, orientation)
  - owner: 0=empty, 1=P1, -1=P2
  - piece_type: 0=empty, 1=church_tower, 2=church_ship, 3=house, 4=priest
  - orientation: 0=horizontal, 1=vertical
- Priest (owner=0, piece_type=4) has no orientation

"""
import numpy as np
from itertools import product


class Board:
    """Represents the LKID game board."""

    # Piece types
    EMPTY = 0
    CHURCH_TOWER = 1
    CHURCH_SHIP = 2
    HOUSE = 3
    PRIEST = 4

    # Orientations
    HORIZONTAL = 0
    VERTICAL = 1

    def __init__(self):
        """Initialize the board with setup configuration."""
        self.n = 7
        # Board state: array of tuples (owner, piece_type, orientation)
        # owner: 1 for P1, -1 for P2, 0 for neutral/empty
        self.board = np.empty((self.n, self.n), dtype=object)
        self._init_empty_board()

    def _init_empty_board(self):
        """Initialize an empty board."""
        for i in range(self.n):
            for j in range(self.n):
                self.board[i][j] = (0, self.EMPTY, None)

    def _set_piece(self, x, y, owner, piece_type, orientation=None):
        """Set a piece at position (x, y)."""
        self.board[x][y] = (owner, piece_type, orientation)

    def _get_piece(self, x, y):
        """Get the piece at position (x, y)."""
        return self.board[x][y]

    def _is_in_bounds(self, x, y):
        """Check if position is within board bounds."""
        return 0 <= x < self.n and 0 <= y < self.n

    def _is_empty(self, x, y):
        """Check if a cell is empty."""
        if not self._is_in_bounds(x, y):
            return False
        owner, piece_type, _ = self._get_piece(x, y)
        return piece_type == self.EMPTY

    def _is_priest_position(self, x, y):
        """Check if the priest is at this position."""
        if not self._is_in_bounds(x, y):
            return False
        owner, piece_type, _ = self._get_piece(x, y)
        return piece_type == self.PRIEST

    def get_priest_position(self):
        """Return the current position of the priest."""
        for x in range(self.n):
            for y in range(self.n):
                if self._is_priest_position(x, y):
                    return (x, y)
        return None

    def setup_board(self, p1_state, p2_state, priest_pos):
        """
        Set up the board with initial piece placements.
        
        p1_state: list of (x, y, piece_type, orientation) for P1
        p2_state: list of (x, y, piece_type, orientation) for P2
        priest_pos: (x, y) position of the priest
        """
        self._init_empty_board()
        
        for x, y, piece_type, orientation in p1_state:
            self._set_piece(x, y, 1, piece_type, orientation)
        
        for x, y, piece_type, orientation in p2_state:
            self._set_piece(x, y, -1, piece_type, orientation)
        
        px, py = priest_pos
        self._set_piece(px, py, 0, self.PRIEST, None)

    def get_legal_moves(self, player):
        """
        Returns all legal moves for the given player.
        A move is represented as (from_x, from_y, to_x, to_y, new_orientation).
        
        Legal moves:
        1. Move a player's piece in its orientation direction (any distance)
        2. Swap with the priest if the piece is blocked
        """
        moves = []
        
        for x in range(self.n):
            for y in range(self.n):
                owner, piece_type, orientation = self._get_piece(x, y)
                if owner != player or piece_type == self.EMPTY:
                    continue
                
                # For each piece, generate possible moves
                moves.extend(self._get_piece_moves(x, y, piece_type, orientation))
        
        return moves

    def _get_piece_moves(self, x, y, piece_type, orientation):
        """Get all valid moves for a specific piece."""
        moves = []
        
        # If priest, it can only swap positions
        if piece_type == self.PRIEST:
            return moves
        
        if orientation == self.HORIZONTAL:
            directions = [(1, 0), (-1, 0)]
        else:  # VERTICAL
            directions = [(0, 1), (0, -1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while self._is_in_bounds(nx, ny):
                if self._is_empty(nx, ny):
                    moves.append((x, y, nx, ny, orientation))
                    nx += dx
                    ny += dy
                else:
                    break
        
        # Always allow swapping with priest as a fallback option
        px, py = self.get_priest_position()
        if (px, py) != (x, y):  # Make sure we're not at the same position
            moves.append((x, y, px, py, orientation))
        
        return moves

    def has_legal_moves(self, player):
        """Check if the player has at least one legal move."""
        return len(self.get_legal_moves(player)) > 0

    def execute_move(self, move, player):
        """
        Execute a move on the board.
        move: (from_x, from_y, to_x, to_y, new_orientation)
        Pieces rotate 90 degrees after moving (horizontal -> vertical, vertical -> horizontal)
        """
        from_x, from_y, to_x, to_y, orientation = move
        
        owner, piece_type, old_orientation = self._get_piece(from_x, from_y)
        
        # Rotate orientation 90 degrees after moving
        new_orientation = self.VERTICAL if orientation == self.HORIZONTAL else self.HORIZONTAL
        
        # Check if this is a swap with the priest
        to_owner, to_piece_type, to_orientation = self._get_piece(to_x, to_y)
        
        if to_piece_type == self.PRIEST:
            # Swap: player's piece goes to priest position, priest goes to players old position
            self._set_piece(to_x, to_y, owner, piece_type, new_orientation)
            self._set_piece(from_x, from_y, 0, self.PRIEST, None)
        else:
            # Regular move
            self._set_piece(to_x, to_y, owner, piece_type, new_orientation)
            self._set_piece(from_x, from_y, 0, self.EMPTY, None)

    def is_adjacent(self, x1, y1, x2, y2):
        """Check if two positions are adjacent (4-connected)."""
        return abs(x1 - x2) + abs(y1 - y2) == 1

    def is_on_edge(self, x, y):
        """Check if a position is on the board edge."""
        return x == 0 or x == self.n - 1 or y == 0 or y == self.n - 1

    def check_church_placement(self, player):
        """
        Check if the player has a valid church placement.
        Church Ship and Church Tower must be adjacent to each other.
        """
        tower_pos = None
        ship_pos = None
        
        for x in range(self.n):
            for y in range(self.n):
                owner, piece_type, _ = self._get_piece(x, y)
                if owner != player:
                    continue
                if piece_type == self.CHURCH_TOWER:
                    tower_pos = (x, y)
                elif piece_type == self.CHURCH_SHIP:
                    ship_pos = (x, y)
        
        if not tower_pos or not ship_pos:
            return False
        
        return self.is_adjacent(*tower_pos, *ship_pos)

    def is_connected_to_church(self, x, y, player):
        """Check if a house at (x, y) is connected to the church or other connected houses."""
        # BFS
        church_pos = None
        
        for cx in range(self.n):
            for cy in range(self.n):
                owner, piece_type, _ = self._get_piece(cx, cy)
                if owner == player and piece_type == self.CHURCH_TOWER:
                    church_pos = (cx, cy)
                    break
        
        if not church_pos:
            return False
        
        visited = set()
        queue = [church_pos]
        visited.add(church_pos)
        
        while queue:
            cx, cy = queue.pop(0)
            
            # Check all adjacent cells
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if not self._is_in_bounds(nx, ny) or (nx, ny) in visited:
                    continue
                
                owner, piece_type, _ = self._get_piece(nx, ny)
                
                # Add player's buildings to the connected component
                if owner == player and piece_type in [self.CHURCH_TOWER, self.CHURCH_SHIP, self.HOUSE]:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return (x, y) in visited

    def check_win_condition(self, player):
        """
        Check if the player has won.
        Win condition: church is properly placed AND all houses are connected to the church.
        """
        if not self.check_church_placement(player):
            return False
        
        for x in range(self.n):
            for y in range(self.n):
                owner, piece_type, _ = self._get_piece(x, y)
                if owner == player and piece_type == self.HOUSE:
                    if not self.is_connected_to_church(x, y, player):
                        return False
        
        return True

    def __copy__(self):
        """Create a deep copy of the board."""
        new_board = Board()
        new_board.board = np.copy(self.board)
        return new_board
