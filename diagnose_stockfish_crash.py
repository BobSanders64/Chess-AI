# diagnose_stockfish_crash2.py

import traceback
import time
import os
from NathanEngine import NathanChessEngine
import chess

def diagnose_stockfish_crash2(
    num_iterations=5000,
    log_path="crash_log2.txt",
    delay_between=0.0
):
    """
    Runs `num_iterations` random-board evaluations to catch any Stockfish subprocess
    crashes. Between calls, checks engine.stockfish._stockfish.poll() to see if the process
    died. If it did, logs FEN, ASCII board, traceback, and restarts the engine.
    """
    # Open (or create) the log file and write a header
    header = f"\n=== Diagnostic run started at {time.ctime()} ===\n"
    with open(log_path, "a") as logfile:
        logfile.write(header)

    # Initialize one engine instance for the entire diagnostic session
    engine = NathanChessEngine()

    for i in range(num_iterations):
        # 1) Generate a random but legal board
        board = engine.GenerateRandomBoard()
        fen = board.fen()

        # 2) Before calling evaluation, check if the engine process is alive
        proc = engine.stockfish._stockfish
        was_alive = (proc is not None and proc.poll() is None)

        # 3) Try to get an evaluation
        try:
            eval_score, mate_in = engine.get_stockfish_evaluation(board, depth=15)
        except Exception as e:
            # In case the wrapper suddenly raises something unexpected
            _log_crash(
                log_path,
                i,
                fen,
                board,
                "Exception in get_stockfish_evaluation()",
                traceback.format_exc()
            )
            # Restart the engine
            engine.cleanup_stockfish()
            engine = NathanChessEngine()
            continue

        # 4) After evaluation, check if the process died
        proc = engine.stockfish._stockfish
        is_alive_after_eval = (proc is not None and proc.poll() is None)

        if was_alive and not is_alive_after_eval:
            # Stockfish died during get_stockfish_evaluation
            _log_crash(
                log_path,
                i,
                fen,
                board,
                "Crash during get_stockfish_evaluation()",
                "Process exited unexpectedly (poll() != None)."
            )
            engine.cleanup_stockfish()
            engine = NathanChessEngine()
            continue

        # 5) Now attempt to get the best move
        proc = engine.stockfish._stockfish
        was_alive = (proc is not None and proc.poll() is None)

        try:
            best_move = engine.get_stockfish_best_move(board, depth=15)
        except Exception as e:
            _log_crash(
                log_path,
                i,
                fen,
                board,
                "Exception in get_stockfish_best_move()",
                traceback.format_exc()
            )
            engine.cleanup_stockfish()
            engine = NathanChessEngine()
            continue

        # 6) After best move, check if process died
        proc = engine.stockfish._stockfish
        is_alive_after_bm = (proc is not None and proc.poll() is None)

        if was_alive and not is_alive_after_bm:
            # Stockfish died during get_stockfish_best_move
            _log_crash(
                log_path,
                i,
                fen,
                board,
                "Crash during get_stockfish_best_move()",
                "Process exited unexpectedly (poll() != None)."
            )
            engine.cleanup_stockfish()
            engine = NathanChessEngine()
            continue

        # 7) (Optional) Add a short delay between iterations if needed
        if delay_between > 0:
            time.sleep(delay_between)

    # Write a footer when finished
    footer = f"\n=== Diagnostic run ended at {time.ctime()} ===\n"
    with open(log_path, "a") as logfile:
        logfile.write(footer)


def _log_crash(log_path, iteration, fen, board, context, tb_str):
    """
    Append a structured entry to the log file whenever a crash is detected.
    """
    timestamp = time.ctime()
    header = f"\n[{timestamp}] Iteration #{iteration} - {context}\n"
    fen_line = f"FEN: {fen}\n"
    board_ascii = board.unicode()  # ASCII/Unicode diagram of the board
    separator = "-" * 80 + "\n"

    with open(log_path, "a", encoding="utf-8") as logfile:
        logfile.write(header)
        logfile.write(fen_line)
        logfile.write("Board Diagram:\n")
        logfile.write(board_ascii + "\n")
        logfile.write("Traceback or Info:\n")
        logfile.write(tb_str + "\n")
        logfile.write(separator)


if __name__ == "__main__":
    # Adjust num_iterations if you want more or fewer tests
    diagnose_stockfish_crash2(
        num_iterations=15,
        log_path="crash_log2.txt",
        delay_between=0.0
    )
