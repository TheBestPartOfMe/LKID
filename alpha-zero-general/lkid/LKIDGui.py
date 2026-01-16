"""
LKID Game Frontend

A graphical interface for playingLKID
"""


# Dirty Keras fix for TF
import tensorflow as tf
import sys
import types

keras_api = types.ModuleType("keras.api")
keras_api_v2 = types.ModuleType("keras.api._v2")

keras_api_v2.keras = tf.keras
keras_api._v2 = keras_api_v2

sys.modules["keras.api"] = keras_api
sys.modules["keras.api._v2"] = keras_api_v2
sys.modules["keras.api._v2.keras"] = tf.keras

import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QGridLayout, QTextEdit)
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QTimer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from lkid.LKIDGame import LKIDGame
from lkid.LKIDLogic import Board
from lkid.keras.NNet import NNetWrapper
from lkid.LKIDPlayers import RandomPlayer
from MCTS import MCTS
from utils import dotdict


class LKIDBoardWidget(QWidget):
    def __init__(self, game, board, selected_piece=None, parent=None):
        super().__init__(parent)
        self.game = game
        self.board = board
        self.selected_piece = selected_piece
        self.setMinimumSize(560, 560)
        self.cell_size = 70

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        self.draw_board(qp)

    def draw_board(self, qp):
        
        for i in range(8):
            qp.drawLine(40, 40 + i * self.cell_size, 40 + 7 * self.cell_size, 40 + i * self.cell_size)
            qp.drawLine(40 + i * self.cell_size, 40, 40 + i * self.cell_size, 40 + 7 * self.cell_size)
        
        font = QFont('Arial', 10)
        qp.setFont(font)
        for y in range(7):
            qp.drawText(40 + y * self.cell_size + self.cell_size//2 - 8, 30, str(y))
        for x in range(7):
            qp.drawText(20, 40 + x * self.cell_size + self.cell_size//2 + 5, str(x))
        
        board_obj = self.game._state_to_board(self.board)
        for x in range(7):
            for y in range(7):
                owner, piece_type, orientation = board_obj._get_piece(x, y)
                if piece_type != Board.EMPTY:
                    color = QColor('black')
                    if owner == 1:
                        color = QColor('blue')
                    elif owner == -1:
                        color = QColor('red')
                    piece_char = self.get_piece_char(piece_type, owner)
                    qp.setFont(QFont('Arial', 22, QFont.Bold))
                    qp.setPen(color)
                    qp.drawText(40 + y * self.cell_size + self.cell_size//2 - 12,
                                40 + x * self.cell_size + self.cell_size//2 + 12,
                                piece_char)
                    
                    if piece_type != Board.PRIEST and orientation is not None:
                        self.draw_orientation_indicator(qp, x, y, orientation, color)
        
        if self.selected_piece:
            x, y = self.selected_piece
            qp.setPen(QColor('red'))
            if owner == 1:
                qp.setPen(QColor('blue'))
            qp.setBrush(Qt.NoBrush)
            qp.drawRect(40 + y * self.cell_size, 40 + x * self.cell_size,
                        self.cell_size, self.cell_size)

    def get_piece_char(self, piece_type, owner):
        if piece_type == Board.PRIEST:
            return 'P'
        elif piece_type == Board.CHURCH_TOWER:
            return 'T' if owner == 1 else 't'
        elif piece_type == Board.CHURCH_SHIP:
            return 'S' if owner == 1 else 's'
        elif piece_type == Board.HOUSE:
            return 'H' if owner == 1 else 'h'
        return '?'

    def draw_orientation_indicator(self, qp, x, y, orientation, color):
        center_x = 40 + y * self.cell_size + self.cell_size//2
        center_y = 40 + x * self.cell_size + self.cell_size//2
        qp.setPen(color)
        if orientation == Board.VERTICAL:
            qp.drawLine(center_x - 15, center_y + 20, center_x + 15, center_y + 20)
        elif orientation == Board.HORIZONTAL:
            qp.drawLine(center_x + 20, center_y - 15, center_x + 20, center_y + 15)

class LKIDFrontend(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LKID - Lass Die Kirche Im Dorf")
        self.setGeometry(100, 100, 700, 900)
        self.game = LKIDGame()
        self.board = self.game.getInitBoard()
        self.current_player = 1
        self.selected_piece = None
        self.game_over = False
        self.ai_player = None
        self.mcts = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        vbox = QVBoxLayout()
        self.board_widget = LKIDBoardWidget(self.game, self.board)
        vbox.addWidget(self.board_widget)
        
        hbox = QHBoxLayout()
        self.new_game_btn = QPushButton("New Game")
        self.new_game_btn.clicked.connect(self.new_game)
        hbox.addWidget(self.new_game_btn)
        self.vs_ai_btn = QPushButton("Play vs AI (best)")
        self.vs_ai_btn.clicked.connect(self.setup_vs_ai)
        hbox.addWidget(self.vs_ai_btn)
        self.vs_random_btn = QPushButton("Play vs Random")
        self.vs_random_btn.clicked.connect(self.setup_vs_random)
        hbox.addWidget(self.vs_random_btn)
        self.vs_human_btn = QPushButton("Play vs Human")
        self.vs_human_btn.clicked.connect(self.setup_vs_human)
        hbox.addWidget(self.vs_human_btn)
        vbox.addLayout(hbox)
        self.status_label = QLabel("Select game mode")
        vbox.addWidget(self.status_label)
        self.instructions_label = QLabel(
            """
Instructians:\n1. Click 'New Game' to start\n2. Choose 'Play vs Human' or 'Play vs AI'\n3. Click your piece, then destination\n4. Goal: Connect all houses to church\nT/t=Tower, S/s=Ship, H/h=House, P=Priest\nBlue=Player 1, Red=Player 2\n↔ horizontal, ↕ vertical\n"""
        )
        vbox.addWidget(self.instructions_label)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        vbox.addWidget(self.history_text)
        central.setLayout(vbox)
        self.setCentralWidget(central)
        self.board_widget.mousePressEvent = self.on_board_click
        self.update_board()

    def update_board(self):
        self.board_widget.board = self.board
        self.board_widget.selected_piece = self.selected_piece
        self.board_widget.update()

    def on_board_click(self, event):
        if self.game_over:
            return
        pos = event.pos()
        x = (pos.y() - 40) // self.board_widget.cell_size
        y = (pos.x() - 40) // self.board_widget.cell_size
        if 0 <= x < 7 and 0 <= y < 7:
            self.handle_cell_click(x, y)

    def handle_cell_click(self, x, y):
        if self.selected_piece is None:
            owner, piece_type, _ = self.game._state_to_board(self.board)._get_piece(x, y)
            if owner == self.current_player and piece_type != Board.EMPTY:
                self.selected_piece = (x, y)
                self.status_label.setText(f"Selected piece at ({x},{y}). Click destination.")
                self.update_board()
        else:
            from_x, from_y = self.selected_piece
            from_idx = from_x * 7 + from_y
            to_idx = x * 7 + y
            move_idx = from_idx * 49 + to_idx
            valids = self.game.getValidMoves(self.board, self.current_player)
            if valids[move_idx]:
                self.make_move(move_idx, self.current_player)
            else:
                self.status_label.setText("Invalid move! Select a different destination")
                self.selected_piece = None
                self.update_board()

    def make_move(self, move_idx, acting_player):
        self.board, self.current_player = self.game.getNextState(self.board, acting_player, move_idx)
        from_idx = move_idx // 49
        to_idx = move_idx % 49
        from_x, from_y = from_idx // 7, from_idx % 7
        to_x, to_y = to_idx // 7, to_idx % 7
        move_text = f"Player {acting_player} moved from ({from_x},{from_y}) to ({to_x},{to_y})\n"
        self.history_text.append(move_text)
        self.selected_piece = None
        self.update_board()
        winner = self.game.getGameEnded(self.board, self.current_player)
        if winner != 0:
            self.game_over = True
            winner_text = "Player -1" if winner == 1 else "Player 1" if winner == -1 else "Draw"
            self.status_label.setText(f"Game Over! Winner: {winner_text}")
            QMessageBox.information(self, "Game Over", f"Winner: {winner_text}")
        else:
            self.status_label.setText(f"Player {self.current_player}'s turn")
            self.check_ai_move()

    def check_ai_move(self):
        if self.ai_player and self.current_player == -1:
            QTimer.singleShot(500, self.make_ai_move)

    def make_ai_move(self):
        if not self.ai_player:
            return
        acting_player = self.current_player
        action = self.ai_player.play(self.board, acting_player)
        self.make_move(action, acting_player)

    def new_game(self):
        self.board = self.game.getInitBoard()
        self.current_player = 1
        self.selected_piece = None
        self.game_over = False
        self.ai_player = None
        self.status_label.setText("Select game mode")
        self.history_text.clear()
        self.update_board()

    def setup_vs_human(self):
        self.ai_player = None
        self.status_label.setText("Player 1's turn (Human vs Human)")

    def setup_vs_ai(self):
        try:
            nnet = NNetWrapper(self.game)
            nnet.load_checkpoint('./temp/', 'best')
            args = dotdict({'numMCTSSims': 25, 'cpuct': 1})
            self.mcts = MCTS(self.game, nnet, args)
            self.ai_player = AIPlayer(self.game, self.mcts)
            self.status_label.setText("Player 1's turn (Human vs AI)")
            QMessageBox.information(self, "Success", "Loaded best AI model from ./temp/best")
        except Exception as e:
            self.ai_player = None
            QMessageBox.critical(self, "Error", f"Failed to load AI model: {str(e)}")

    def setup_vs_random(self):
        self.ai_player = GUIRandomPlayer(self.game)
        self.status_label.setText("Player 1's turn (Human vs Random)")
        self.history_text.append("Random opponent ready.\n")
        
class AIPlayer:
    def __init__(self, game, mcts):
        self.game = game
        self.mcts = mcts
    def play(self, board):
        probs = self.mcts.getActionProb(board, temp=0)
        action = np.argmax(probs)
        return action


class GUIRandomPlayer(RandomPlayer):
    def play(self, board, player=-1):
        valids = self.game.getValidMoves(board, player)
        candidates = np.where(valids)[0]
        if len(candidates) == 0:
            return 0
        return int(np.random.choice(candidates))

def main():
    app = QApplication(sys.argv)
    window = LKIDFrontend()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
