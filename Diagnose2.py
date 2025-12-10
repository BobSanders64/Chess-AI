import chess
from NathanEngine import NathanChessEngine

engine = NathanChessEngine()
board = chess.Board("2Q5/3kP3/P1rPpqp1/p1N1PB1P/BPPP1rPN/1N3bP1/P3p1p1/6K1 w - - 0 1")

# Before patch, this would crash. After patch:
# 1) GenerateRandomBoard would never produce that exact side-to-move, but
# 2) if you want to test directly:
engine.WhoseTurnIsIt = "chess.WHITE"
engine.board = board
engine.board.turn = chess.WHITE

# Now check what happens:
print("Eval:", engine.get_stockfish_evaluation(engine.board))
print("Best move:", engine.get_stockfish_best_move(engine.board))
