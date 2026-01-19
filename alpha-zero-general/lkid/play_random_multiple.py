
from __future__ import annotations

import argparse
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lkid.LKIDGame import LKIDGame

class ProgressTracker:
    def __init__(self, total_games: int) -> None:
        self.total_games = total_games
        self.total_moves = 0
        self.completed_games = 0
        self.move_counts: list[int] = []
        self.lock = threading.Lock()
        self.done_event = threading.Event()

    def record_move(self) -> None:
        with self.lock:
            self.total_moves += 1

    def record_game_result(self, move_count: int) -> None:
        with self.lock:
            self.completed_games += 1
            self.move_counts.append(move_count)
            if self.completed_games == self.total_games:
                self.done_event.set()

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        return self.done_event.wait(timeout)

    def snapshot(self) -> tuple[int, int, int]:
        with self.lock:
            return self.total_moves, self.completed_games, self.total_games

    def summary(self) -> tuple[float, int, int]:
        with self.lock:
            if not self.move_counts:
                return 0.0, 0, 0
            total_moves = sum(self.move_counts)
            avg_moves = total_moves / len(self.move_counts)
            return avg_moves, min(self.move_counts), max(self.move_counts)


def progress_reporter(tracker: ProgressTracker, interval_seconds: int) -> None:
    while True:
        finished = tracker.wait_for_completion(timeout=interval_seconds)
        total_moves, completed_games, total_games = tracker.snapshot()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{timestamp}] Fortschritt: {completed_games}/{total_games} Spiele fertig, "
            f"bisherige Züge: {total_moves}"
        )
        if finished:
            break


def play_random_game(tracker: ProgressTracker) -> int:
    game = LKIDGame()
    state = game.getInitBoard()
    player = 1
    move_count = 0
    rng = random.Random()

    while True:
        valid_moves = game.getValidMoves(state, player)
        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) == 0:
            break
        action = rng.choice(valid_indices)
        state, player = game.getNextState(state, player, action)
        move_count += 1
        tracker.record_move()
        ended = game.getGameEnded(state, player)
        if ended != 0:
            break

    tracker.record_game_result(move_count)
    return move_count


def run_multithreaded_games(num_games: int, report_interval: int, max_workers: int | None) -> None:
    tracker = ProgressTracker(num_games)
    reporter_thread = threading.Thread(
        target=progress_reporter,
        args=(tracker, report_interval),
        daemon=True,
    )
    reporter_thread.start()

    workers = max_workers or min(num_games, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(play_random_game, tracker) for _ in range(num_games)]
        for future in futures:
            future.result()

    tracker.wait_for_completion()
    reporter_thread.join()

    avg_moves, min_moves, max_moves = tracker.summary()
    total_moves = sum(tracker.move_counts)
    print("Alle Spiele abgeschlossen.")
    print(f"Gesamte Züge: {total_moves}")
    print(f"Durchschnittliche Züge pro Spiel: {avg_moves:.2f}")
    print(f"Minimale Züge in einer Partie: {min_moves}")
    print(f"Maximale Züge in einer Partie: {max_moves}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spielt mehrere LKID-Partien zufällig in Threads.")
    parser.add_argument("--games", type=int, default=30, help="Anzahl der Partien")
    parser.add_argument(
        "--report-interval",
        type=int,
        default=30,
        help="Intervall für Fortschrittsmeldungen in Sekunden",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximale Anzahl Threads",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multithreaded_games(args.games, args.report_interval, args.max_workers)
