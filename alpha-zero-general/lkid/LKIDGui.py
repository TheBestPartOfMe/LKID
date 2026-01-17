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
import textwrap
import numpy as np
import pygame

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from lkid.LKIDGame import LKIDGame
from lkid.LKIDGame5x5 import LKIDGame as LKIDGame5x5
from lkid.LKIDGame5x5Barriers import LKIDGame5x5Barriers
from lkid.LKIDLogic import Board
from lkid.keras.NNet import NNetWrapper
from lkid.LKIDPlayers import RandomPlayer
from MCTS import MCTS
from utils import dotdict


class UIButton:
    """Lightweight button helper for pygame UI."""

    def __init__(self, rect, text, callback):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.hover = False

    def set_text(self, text):
        self.text = text

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False

    def draw(self, surface, font):
        base_color = (52, 88, 187) if self.hover else (36, 62, 132)
        pygame.draw.rect(surface, base_color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (16, 24, 48), self.rect, 2, border_radius=6)
        label = font.render(self.text, True, (235, 240, 255))
        label_rect = label.get_rect(center=self.rect.center)
        surface.blit(label, label_rect)


class LKIDPygameFrontend:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("LKID - Lass Die Kirche Im Dorf")

        self.width = 1100
        self.height = 780
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.board_origin = (40, 40)
        self.cell_size = 70
        self.max_board_size = self.cell_size * 7

        self.game_variants = [
            ("Classic 7x7", LKIDGame),
            ("Classic 5x5", LKIDGame5x5),
            ("5x5 + Barriers", LKIDGame5x5Barriers),
        ]
        self.variant_index = 0
        self.game_class = self.game_variants[self.variant_index][1]
        self.game = self.game_class()
        self.board = self.game.getInitBoard()
        self._update_board_metrics()
        self.current_player = 1
        self.selected_piece = None
        self.game_over = False
        self.ai_player = None
        self.mcts = None
        self.pending_ai_move_time = None

        self.history = []
        self.max_history = 4
        self.status_text = f"{self.current_variant_name}: Select game mode"

        self.font_small = pygame.font.SysFont("arial", 18)
        self.font_medium = pygame.font.SysFont("arial", 22)
        self.font_large = pygame.font.SysFont("arial", 28, bold=True)

        self.instructions = (
            "1. Use 'Variant' to switch board types\n"
            "2. Click 'New Game' then pick an opponent\n"
            "3. Select your piece, then its destination\n"
            "Goal: Connect every house to your church\n"
            "Blue=P1, Red=P2; '-' horizontal, '|' vertical\n"
            "Barrier tiles (if present) block movement"
        )

        self.buttons = []
        self.variant_button = None
        self._create_buttons()

    def _create_buttons(self):
        panel_x = self._panel_origin_x()
        btn_width = 360
        btn_height = 48
        spacing = 16
        top = 70

        def add_button(label, callback):
            rect = (panel_x, top + len(self.buttons) * (btn_height + spacing), btn_width, btn_height)
            button = UIButton(rect, label, callback)
            self.buttons.append(button)
            return button

        self.variant_button = add_button(self._variant_button_label(), self.cycle_variant)
        add_button("New Game", self.new_game)
        add_button("Play vs Human", self.setup_vs_human)
        add_button("Play vs Random", self.setup_vs_random)
        add_button("Play vs AI (best)", self.setup_vs_ai)

    def _panel_origin_x(self):
        return self.board_origin[0] + self.max_board_size + 40

    @property
    def current_variant_name(self):
        return self.game_variants[self.variant_index][0]

    def _variant_button_label(self):
        return f"Variant: {self.current_variant_name}"

    def _update_board_metrics(self):
        self.board_size = self.cell_size * self.game.n

    def cycle_variant(self):
        self.variant_index = (self.variant_index + 1) % len(self.game_variants)
        self.game_class = self.game_variants[self.variant_index][1]
        self.game = self.game_class()
        self.board = self.game.getInitBoard()
        self._update_board_metrics()
        self.current_player = 1
        self.selected_piece = None
        self.game_over = False
        self.ai_player = None
        self.mcts = None
        self.pending_ai_move_time = None
        self.history.clear()
        variant_name = self.current_variant_name
        self.status_text = f"{variant_name}: Select game mode"
        self.add_history_entry(f"Variant changed to {variant_name}")
        if self.variant_button:
            self.variant_button.set_text(self._variant_button_label())

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.type == pygame.MOUSEMOTION:
                    for button in self.buttons:
                        button.handle_event(event)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if any(button.handle_event(event) for button in self.buttons):
                        continue
                    self.handle_board_click(event.pos)

            if not running:
                break

            self.update_ai()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit(0)

    def update_ai(self):
        if self.pending_ai_move_time is None:
            return
        if pygame.time.get_ticks() >= self.pending_ai_move_time:
            self.pending_ai_move_time = None
            self.make_ai_move()

    def draw(self):
        self.screen.fill((12, 16, 30))
        self._draw_board()
        self._draw_panel()

    def _draw_board(self):
        size = self.game.n
        ox, oy = self.board_origin
        board_px = self.board_size
        pygame.draw.rect(
            self.screen, (24, 30, 52), (ox - 10, oy - 10, board_px + 20, board_px + 20), 0, border_radius=8
        )

        for i in range(size + 1):
            start_h = (ox, oy + i * self.cell_size)
            end_h = (ox + board_px, oy + i * self.cell_size)
            start_v = (ox + i * self.cell_size, oy)
            end_v = (ox + i * self.cell_size, oy + board_px)
            pygame.draw.line(self.screen, (70, 90, 130), start_h, end_h, 2)
            pygame.draw.line(self.screen, (70, 90, 130), start_v, end_v, 2)

        axis_color = (180, 190, 210)
        for idx in range(size):
            label = self.font_small.render(str(idx), True, axis_color)
            x_pos = ox + idx * self.cell_size + self.cell_size // 2 - label.get_width() // 2
            self.screen.blit(label, (x_pos, oy - 30))

        for idx in range(size):
            label = self.font_small.render(str(idx), True, axis_color)
            y_pos = oy + idx * self.cell_size + self.cell_size // 2 - label.get_height() // 2
            self.screen.blit(label, (ox - 30, y_pos))

        board_obj = self.game._state_to_board(self.board)
        for x in range(size):
            for y in range(size):
                owner, piece_type, orientation = board_obj._get_piece(x, y)
                if piece_type == Board.EMPTY:
                    continue
                if piece_type == Board.BARRIER:
                    rect = pygame.Rect(
                        ox + y * self.cell_size + 6,
                        oy + x * self.cell_size + 6,
                        self.cell_size - 12,
                        self.cell_size - 12,
                    )
                    pygame.draw.rect(self.screen, (80, 90, 120), rect, border_radius=6)
                    pygame.draw.line(self.screen, (25, 30, 45), rect.topleft, rect.bottomright, 3)
                    pygame.draw.line(self.screen, (25, 30, 45), rect.topright, rect.bottomleft, 3)
                    continue
                color = (0, 120, 255) if owner == 1 else (220, 65, 65)
                if owner not in (1, -1):
                    color = (220, 220, 220)
                piece_char = self.get_piece_char(piece_type, owner)
                label = self.font_large.render(piece_char, True, color)
                center = (
                    ox + y * self.cell_size + self.cell_size // 2 - label.get_width() // 2,
                    oy + x * self.cell_size + self.cell_size // 2 - label.get_height() // 2,
                )
                self.screen.blit(label, center)

                if piece_type not in (Board.PRIEST, Board.BARRIER) and orientation is not None:
                    self.draw_orientation_indicator(x, y, orientation, color)

        if self.selected_piece:
            sx, sy = self.selected_piece
            rect = pygame.Rect(
                ox + sy * self.cell_size,
                oy + sx * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(self.screen, (255, 210, 0), rect, 4)

    def _draw_panel(self):
        panel_x = self._panel_origin_x()
        panel_width = self.width - panel_x - 30
        panel_rect = pygame.Rect(panel_x, 30, panel_width, self.height - 60)
        pygame.draw.rect(self.screen, (20, 28, 54), panel_rect, 0, border_radius=10)
        pygame.draw.rect(self.screen, (50, 70, 120), panel_rect, 2, border_radius=10)

        status_label = self.font_medium.render(self.status_text, True, (235, 240, 255))
        self.screen.blit(status_label, (panel_x + 16, 32))

        for button in self.buttons:
            button.draw(self.screen, self.font_small)

        instructions_y = self.buttons[-1].rect.bottom + 30
        title = self.font_medium.render("Instructions", True, (210, 220, 240))
        self.screen.blit(title, (panel_x + 16, instructions_y))
        instructions_y += 32

        for line in self.wrap_text(self.instructions, 48):
            label = self.font_small.render(line, True, (200, 205, 220))
            self.screen.blit(label, (panel_x + 16, instructions_y))
            instructions_y += 22

        history_y = instructions_y + 24
        history_title = self.font_medium.render("History", True, (210, 220, 240))
        self.screen.blit(history_title, (panel_x + 16, history_y))
        history_y += 32
        for line in self._get_history_lines(max_chars=48):
            label = self.font_small.render(line, True, (190, 200, 215))
            self.screen.blit(label, (panel_x + 16, history_y))
            history_y += 20

    def wrap_text(self, text, width):
        return textwrap.wrap(text.replace("\n", " "), width=width)

    def _get_history_lines(self, max_chars):
        lines = []
        for entry in self.history[-self.max_history :]:
            wrapped = self.wrap_text(entry, max_chars)
            if not wrapped:
                continue
            lines.extend(wrapped)
        return lines[-self.max_history :]

    def handle_board_click(self, pos):
        if self.game_over:
            return
        coords = self.pos_to_coord(pos)
        if coords:
            self.handle_cell_click(*coords)

    def pos_to_coord(self, pos):
        px, py = pos
        ox, oy = self.board_origin
        if ox <= px < ox + self.board_size and oy <= py < oy + self.board_size:
            x = (py - oy) // self.cell_size
            y = (px - ox) // self.cell_size
            return int(x), int(y)
        return None

    def handle_cell_click(self, x, y):
        board_obj = self.game._state_to_board(self.board)
        owner, piece_type, _ = board_obj._get_piece(x, y)
        if self.selected_piece is None:
            if owner == self.current_player and piece_type != Board.EMPTY:
                self.selected_piece = (x, y)
                self.status_text = f"Selected ({x},{y}). Choose destination"
        else:
            from_x, from_y = self.selected_piece
            move_idx = self.coords_to_move_idx(from_x, from_y, x, y)
            valids = self.game.getValidMoves(self.board, self.current_player)
            if valids[move_idx]:
                self.make_move(move_idx, self.current_player)
            else:
                self.status_text = "Invalid move. Select again"
                self.selected_piece = None

    def coords_to_move_idx(self, from_x, from_y, to_x, to_y):
        side = self.game.n
        board_area = side * side
        from_idx = from_x * side + from_y
        to_idx = to_x * side + to_y
        return from_idx * board_area + to_idx

    def make_move(self, move_idx, acting_player):
        self.board, self.current_player = self.game.getNextState(self.board, acting_player, move_idx)
        side = self.game.n
        board_area = side * side
        from_idx = move_idx // board_area
        to_idx = move_idx % board_area
        from_x, from_y = divmod(from_idx, side)
        to_x, to_y = divmod(to_idx, side)
        self.add_history_entry(f"P{acting_player}: ({from_x},{from_y}) → ({to_x},{to_y})")
        self.selected_piece = None

        winner = self.game.getGameEnded(self.board, self.current_player)
        if winner != 0:
            self.game_over = True
            if winner == 1:
                winner_text = "Player -1"
            elif winner == -1:
                winner_text = "Player 1"
            else:
                winner_text = "Draw"
            self.status_text = f"Game over. Winner: {winner_text}"
            self.add_history_entry(f"Result: {winner_text}")
        else:
            self.status_text = f"Player {self.current_player}'s turn"
            self.check_ai_move()

    def check_ai_move(self):
        if self.ai_player and self.current_player == -1 and self.pending_ai_move_time is None:
            self.pending_ai_move_time = pygame.time.get_ticks() + 500

    def make_ai_move(self):
        if not self.ai_player:
            return
        acting_player = self.current_player
        action = self.ai_player.play(self.board, acting_player)
        self.make_move(action, acting_player)

    def add_history_entry(self, text):
        self.history.append(text)
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history :]

    def new_game(self):
        self.board = self.game.getInitBoard()
        self.current_player = 1
        self.selected_piece = None
        self.game_over = False
        self.ai_player = None
        self.pending_ai_move_time = None
        self.status_text = f"{self.current_variant_name}: Select game mode"
        self.history.clear()

    def setup_vs_human(self):
        self.ai_player = None
        self.pending_ai_move_time = None
        self.status_text = "Player 1's turn (Human vs Human)"
        self.add_history_entry("Mode: Human vs Human")

    def setup_vs_random(self):
        self.ai_player = GUIRandomPlayer(self.game)
        self.pending_ai_move_time = None
        self.status_text = "Player 1's turn (Human vs Random)"
        self.add_history_entry("Random opponent ready.")

    def setup_vs_ai(self):
        try:
            nnet = NNetWrapper(self.game)
            nnet.load_checkpoint("./temp/", "best")
            args = dotdict({"numMCTSSims": 25, "cpuct": 1})
            self.mcts = MCTS(self.game, nnet, args)
            self.ai_player = AIPlayer(self.game, self.mcts)
            self.pending_ai_move_time = None
            self.status_text = "Player 1's turn (Human vs AI)"
            self.add_history_entry("Loaded best AI from ./temp/best")
        except Exception as exc:
            self.ai_player = None
            self.mcts = None
            self.status_text = "AI load failed. Check console."
            self.add_history_entry(f"AI error: {exc}")

    def get_piece_char(self, piece_type, owner):
        if piece_type == Board.PRIEST:
            return "P"
        if piece_type == Board.CHURCH_TOWER:
            return "T" if owner == 1 else "t"
        if piece_type == Board.CHURCH_SHIP:
            return "S" if owner == 1 else "s"
        if piece_type == Board.HOUSE:
            return "H" if owner == 1 else "h"
        return "?"

    def draw_orientation_indicator(self, x, y, orientation, color):
        ox, oy = self.board_origin
        center_x = ox + y * self.cell_size + self.cell_size // 2
        center_y = oy + x * self.cell_size + self.cell_size // 2
        if orientation == Board.VERTICAL:
            pygame.draw.line(self.screen, color, (center_x - 15, center_y + 20), (center_x + 15, center_y + 20), 3)
        elif orientation == Board.HORIZONTAL:
            pygame.draw.line(self.screen, color, (center_x + 20, center_y - 15), (center_x + 20, center_y + 15), 3)


class AIPlayer:
    def __init__(self, game, mcts):
        self.game = game
        self.mcts = mcts

    def play(self, board, player=-1):
        probs = self.mcts.getActionProb(board, temp=0)
        return int(np.argmax(probs))


class GUIRandomPlayer(RandomPlayer):
    def __init__(self, game):
        super().__init__(game)

    def play(self, board, player=-1):
        valids = self.game.getValidMoves(board, player)
        candidates = np.where(valids)[0]
        if len(candidates) == 0:
            return 0
        return int(np.random.choice(candidates))


def main():
    frontend = LKIDPygameFrontend()
    frontend.run()


if __name__ == "__main__":
    main()
