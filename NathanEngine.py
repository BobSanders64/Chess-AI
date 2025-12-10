import chess
import chess.engine
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import deque
from stockfish import Stockfish
from sympy.codegen.ast import continue_
import atexit
import sys
import os
import copy
import time
import secrets
import concurrent.futures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Monkey patch to fix Stockfish cleanup issue, this prevents the encoding error at shutdown
def stockfish_del_override(self):
    """Override Stockfish's problematic __del__ method"""
    try:
        if hasattr(self, '_stockfish') and self._stockfish:
            try:
                self._stockfish.stdin.write('quit\n'.encode())
                self._stockfish.stdin.flush()
            except:
                pass
            try:
                self._stockfish.terminate()
            except:
                pass
    except:
        pass

Stockfish.__del__ = stockfish_del_override

class NathanChessEngine(nn.Module):
    def __init__(self, training=False):
        super(NathanChessEngine, self).__init__()
        self.depth = 23;self.skill = 20

        #Stockfish Engine
        try:
            self.stockfish = Stockfish(r"C:\Users\iwana\OneDrive\Documents\stockfish\stockfish-windows-x86-64-avx2.exe")
            # Set some basic parameters to ensure Stockfish is working
            self.stockfish.set_depth(self.depth)
            self.stockfish.set_skill_level(self.skill)
            if training==True:
                print("Stockfish initialized successfully")
        except Exception as e:
            if training==True:
                print(f"Failed to initialize Stockfish: {e}")
            self.stockfish = None

        atexit.register(self.cleanup_stockfish)

        #Tracking variables
        self.result = None
        self.move_history = []
        self.board = chess.Board()
        self.WhoseTurnIsIt = "Not yours lol"
        self.promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

        #CNN Architecture
        #Input: 12 channels (6 piece types × 2 colors) on 8×8 board
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        #Fully connected layers
        self.fc1 = nn.Linear(256*8*8, 1024)
        self.fc2 = nn.Linear(1024, 4096+32)  #All possible moves+the 32 possible promotions (queen, rook, bishop, knight)

        #Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #####CONTROL ZONE
        #Learning Perameters
        self.η = 1E-6
        self.optimizer = optim.Adam(self.parameters(), lr=self.η)

        #Loss Function
        self.LossFunction = nn.SmoothL1Loss()
        self.lossBox=np.array([])

        #Memory
        self.memory = deque(maxlen=7500)  #Experience replay buffer
        self.timeout=6 #second(s)
        self.youcomehereoften=1 #How how depth should it reduce

        #Q learning parameters
        self.γ = 0.95  #Discount factor for future rewards #.75 for this test

        #Sizes
        self.batch_size = 64
        self.MaxEpisodesToTrain=1001 #1 more than you really want

        #Exploration parameters
        self.epsilon = 1.0  #Starting exploration rate
        self.epsilon_min = 0.1 #Min exploration rate
        self.epsilon_decay_rate = 6E-4 #Decay rate per episode
        self.ExplorThreshold=0.98765432101234567890

        self.to(device)

        #Target Netowrk Properties
        self.target_update_counter = 0
        self.target_update_freq = 20
        #####

        #Target Network
        #Target network layers (separate from main network)
        self.target_conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1).to(device)
        self.target_bn1 = nn.BatchNorm2d(64).to(device)
        self.target_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
        self.target_bn2 = nn.BatchNorm2d(128).to(device)
        self.target_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1).to(device)
        self.target_bn3 = nn.BatchNorm2d(256).to(device)
        self.target_fc1 = nn.Linear(256 * 8 * 8, 1024).to(device)
        self.target_fc2 = nn.Linear(1024, 4096 + 32).to(device)

        #Copy weights from main network to target network
        self.update_target_network()
        self.target_update_counter = 0

    def forward(self, x):
        push1 = F.elu(self.bn1(self.conv1(x)))
        push2 = F.elu(self.bn2(self.conv2(push1)))
        push3 = F.elu(self.bn3(self.conv3(push2)))
        push4 = push3.view(-1, 256 * 8 * 8) #Flattens output to 1D vector
        #Fully connected layers
        push5 = F.elu(self.fc1(push4))
        QvaluEs = self.fc2(push5)
        return QvaluEs

    # ============== SELF PLAY ==============
    def copy(self):
        """Create a copy of the current model for self-play opponent"""
        new_bot = NathanChessEngine(training=self.training)
        new_bot.load_state_dict(self.state_dict())
        new_bot.epsilon = self.epsilon
        return new_bot

    def self_play_game(self, opponent, max_moves=200, verbose=False):
        """
        Play a complete game against an opponent.
        Returns game history and outcome.
        """
        board = chess.Board()
        history = []

        # Randomly assign colors
        if np.random.random() < 0.5:
            white_player, black_player = self, opponent
            self_is_white = True
        else:
            white_player, black_player = opponent, self
            self_is_white = False

        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            # Determine current player
            current_player = white_player if board.turn == chess.WHITE else black_player
            is_self = (current_player == self)

            # Make move
            move, state = current_player.play_move(board)

            if move is None:
                break

            # Store in history
            if is_self:
                history.append({
                    'state': state,
                    'action': self.move_to_index(move),
                    'color': board.turn
                })

            board.push(move)
            move_count += 1

            if verbose and move_count % 10 == 0:
                print(f"Move {move_count}: {move.uci()}")

        # Determine outcome
        outcome = board.outcome()
        if outcome is None:
            reward = 0.0  # Draw
        elif outcome.winner == chess.WHITE:
            reward = 1.0 if self_is_white else -1.0
        else:  # Black wins
            reward = -1.0 if self_is_white else 1.0

        return history, reward, move_count

    # ============== BOARD REPRESENTATION ==============

    def board_to_tensor(self, board=None):
        """Convert a chess.Board to a tensor representation."""
        if board is None:
            board = self.board

        #Create a 12-channel 8x8 tensor (zeros)
        tensor = torch.zeros((12, 8, 8), device=device)

        #Fill the tensor based on piece positions
        piece_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  #White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  #Black pieces
        }
        for sq in range(64):
            piece = board.piece_at(sq) #Basicaly a complex version of enemruate
            if piece!=None:
                #Calculate row and column (0-7)
                row, col = divmod(sq, 8)
                tensor[piece_idx[piece.symbol()], row, col] = 1 #Put a marker there to say "I see you"
        return tensor.unsqueeze(0)  #Add batch dimension

    def tensor_to_board(self, tensor, display=False, validate=False):
        """Convert a tensor representation back to a chess.Board object.

        Args:
            tensor: A tensor of shape (B, 12, 8, 8), (1, 12, 8, 8) or (12, 8, 8) representing the board(s)
            display: If True, returns string representation of the board instead of board object
            validate: If True, validates the conversion by comparing with original

        Returns:
            - chess.Board object (default) for single board
            - String representation if display=True (formatted for batches)
            - Validation result (True/False) if validate=True
        """
        # Ensure tensor is on CPU for processing
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # Create reverse mapping from channel index to piece
        idx_to_piece = {
            0: chess.Piece(chess.PAWN, chess.WHITE),  # P
            1: chess.Piece(chess.KNIGHT, chess.WHITE),  # N
            2: chess.Piece(chess.BISHOP, chess.WHITE),  # B
            3: chess.Piece(chess.ROOK, chess.WHITE),  # R
            4: chess.Piece(chess.QUEEN, chess.WHITE),  # Q
            5: chess.Piece(chess.KING, chess.WHITE),  # K
            6: chess.Piece(chess.PAWN, chess.BLACK),  # p
            7: chess.Piece(chess.KNIGHT, chess.BLACK),  # n
            8: chess.Piece(chess.BISHOP, chess.BLACK),  # b
            9: chess.Piece(chess.ROOK, chess.BLACK),  # r
            10: chess.Piece(chess.QUEEN, chess.BLACK),  # q
            11: chess.Piece(chess.KING, chess.BLACK),  # k
        }

        # Check if this is a batch of boards
        is_batch = tensor.dim() == 4 and tensor.shape[0] > 1

        if is_batch:
            # Handle batch of boards - always return string for printing
            board_strings = []
            for i in range(min(3, tensor.shape[0])):  # Show first 3 boards
                board = chess.Board(None)
                piece_map = {}

                for row in range(8):
                    for col in range(8):
                        for channel in range(12):
                            if tensor[i, channel, row, col].item() > 0.5:
                                square = row * 8 + col
                                piece_map[square] = idx_to_piece[channel]
                                break

                board.set_piece_map(piece_map)
                board_strings.append(f"Board {i + 1}:\n{str(board)}")

            if tensor.shape[0] > 3:
                board_strings.append(f"... and {tensor.shape[0] - 3} more boards")

            return "\n".join(board_strings)
        else:
            # Handle single board
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)

            # Convert tensor to board
            board = chess.Board(None)
            piece_map = {}

            for row in range(8):
                for col in range(8):
                    for channel in range(12):
                        if tensor[channel, row, col].item() > 0.5:
                            square = row * 8 + col
                            piece_map[square] = idx_to_piece[channel]
                            break

            board.set_piece_map(piece_map)

            # Handle return options
            if validate:
                original_fen = self.board.fen().split()[0]
                reconstructed_fen = board.fen().split()[0]

                if original_fen == reconstructed_fen:
                    print("✓ Tensor conversion validated successfully!")
                    return True
                else:
                    print("✗ Tensor conversion failed!")
                    print(f"Original:      {original_fen}")
                    print(f"Reconstructed: {reconstructed_fen}")
                    print("\nOriginal board:")
                    print(self.board)
                    print("\nReconstructed board:")
                    print(board)
                    return False
            elif display:
                return str(board)
            else:
                return board

    # ============== MOVE TYPE TRANSFERING ==============

    def move_to_index(self, move):
        """Convert a chess move to an index in the 4096+32 output space."""
        if move.promotion is not None:
            #Get the file (column) of the destination square (0-7)
            promo_file = move.to_square%8
            #Get the piece type index (0-3 for queen, rook, bishop, knight)
            piece_idx = self.promotion_pieces.index(move.promotion)
            #4096 + (8 * piece_idx + promo_file) gives index in range 4096-4127
            return (4096 + (8 * piece_idx + promo_file))
        else: #It's a regular move
            from_sq = move.from_square
            to_sq = move.to_square
            return from_sq * 64 + to_sq

    def index_to_move(self, index):
        """Convert an index back to a chess move."""
        #Check if this is a promotion move
        if index >= 4096:
            #Calculate which promotion piece and file
            promo_index = index - 4096
            piece_idx = promo_index // 8
            promo_file = promo_index % 8

            #Find the appropriate pawn that can promote on this file
            promotion_piece = self.promotion_pieces[piece_idx]

            #Check legal moves for a pawn that can promote on this file
            for move in self.board.legal_moves:
                if move.promotion == promotion_piece and move.to_square % 8 == promo_file:
                    return move
            return None #Error prevention: should never happen.
        else:
            from_sq = index // 64
            to_sq = index % 64
            return chess.Move(from_sq, to_sq)

    # ============== BOARD SET-UP AND VALIDATION ==============

    def BasicChecks(self):
        # Detects checkmates, stalemates, if kings are next to eachother, opposite color pawns on back row, and draws by insufficient material
        BlackKingLoc = None
        WhiteKingLoc = None

        for sq, piece in self.board.piece_map().items():
            if piece.symbol() == 'k':
                BlackKingLoc = int(sq)
            elif piece.symbol() == 'K':
                WhiteKingLoc = int(sq)
            elif (piece.symbol() == 'p' and sq > 55) or (piece.symbol() == 'P' and sq > 55):
                EndResult = "Pawns on backrow"
                return EndResult
            elif (piece.symbol() == 'P' and sq <= 7) or (piece.symbol() == 'p' and sq <= 7):
                EndResult = "Pawns on backrow"
                return EndResult

        if BlackKingLoc is not None and WhiteKingLoc is not None:
            adjacent_squares = [BlackKingLoc + 1, BlackKingLoc - 1, BlackKingLoc + 8, BlackKingLoc - 8, BlackKingLoc + 9, BlackKingLoc - 9, BlackKingLoc + 7, BlackKingLoc - 7]

            if WhiteKingLoc in adjacent_squares:
                EndResult = "Got a King problem"
                return EndResult

        if self.board.is_game_over() or self.board.is_insufficient_material() or self.board.is_stalemate() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            EndResult = self.board.outcome(claim_draw=True)
        else:
            EndResult = None
        return EndResult

    @staticmethod
    def BetterRandomZeroOrOne():
        '''This will do a better job than np.random.randint(0,2)'''
        if secrets.randbits(1)==0:
            return 0
        else:
            return 1

    def AltGenerateRandomBoard(self):
        '''More realistic generation of Random Board'''
        RealisticBoards=["r2qkb1r/ppp2p1p/2n2np1/3ppb2/2P4P/P5P1/1PQPPP2/RNB1KBNR w KQkq - 0 1",
                         "rnbq1rk1/pp2bppp/4pn2/2pp4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 0 8",
                         "3r1rk1/ppq2p2/2n3pb/2pRp2p/P1N1P2P/2P2BP1/1PQ2P2/R5K1 b - - 4 22",
                         "3r2k1/p1q1np2/1p4pb/2p1p2p/P1N1P2P/2P2BP1/1PQ2P2/3R2K1 w - - 0 25",
                         "3r1rk1/p1q2p2/1pn3pb/2pRp2p/P1N1P2P/2P2BP1/1PQ2P2/R5K1 w - - 0 23",
                         "r1bq1rk1/pppp1ppp/2n2n2/4p3/1b2P3/2N1BN2/PPP2PPP/R2QKB1R w KQ - 2 6",
                         "r2q1rk1/pp2bppp/2ppbn2/4p3/2P1P3/2N2QNP/PPP2PP1/R1B2RK1 w - - 1 11",
                         "8/8/8/8/8/6k1/7p/7K b - - 1 73",
                         "3R1rk1/p1q1np2/1p4pb/2p1p2p/P1N1P2P/2P2BP1/1PQ2P2/3R2K1 b - - 0 24",
                         "2rq1rk1/pp1nbppp/2p1pn2/8/3P4/2N1BN2/PPQ1BPPP/2RR2K1 w - - 8 12",
                         "r4rk1/pp1n1ppp/2pb1q2/3p4/3P4/2N1PN2/PP3PPP/R2Q1RK1 w - - 0 11",
                         "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
                         "r3kbnr/pppn1ppp/8/q3p3/8/2N2N2/PPPP1PPP/R1BQ1RK1 b kq - 1 7",
                         "r3k1nr/pppn1ppp/8/q3p3/8/P1P2N2/2PP1PPP/R1BQ1RK1 b kq - 0 9",
                         "2kr2nr/pppn1ppp/8/4p3/8/P1qP1N2/2PB1PPP/R2Q1RK1 b - - 1 11",
                         "r2q1rk1/pp2bppp/2ppbn2/4p3/2P1P3/1PN2QNP/P1P2PP1/R1B2RK1 b - - 0 11",
                         "r2q1rk1/pp2bppp/2p1bn2/3pp3/2P1P3/1PN2QNP/P1P2PP1/R1B2RK1 w - - 0 12",
                         "2n4r/1k5p/Qnq5/4p3/5pp1/3P4/3N1PPP/R5K1 b - - 1 27",
                         "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
                         "r3k1nr/pppn1ppp/8/q3p3/8/P1b2N2/1PPP1PPP/R1BQ1RK1 w kq - 0 9",
                         "2kr2nr/pppn1ppp/8/4p3/8/P1qP1N2/2P2PPP/R1BQ1RK1 w - - 0 11",
                         "2kr2nr/pppn1ppp/8/4p3/8/P1qP1N2/2PB1PPP/R2Q1RK1 b - - 1 11",
                         "2kr2nr/Bppn2pp/2q2p2/4p3/8/P2P1N2/2P2PPP/R2Q1RK1 w - - 0 14",
                         "2kr2nr/B1pn3p/1pq2p2/P3p1p1/8/3P1N2/2P2PPP/R2Q1RK1 w - - 0 16",
                         "2kr2nr/B1pn3p/1Pq2p2/4p1p1/8/3P1N2/2P2PPP/R2Q1RK1 b - - 0 16",
                         "2n4r/1k1n3p/1pq2p2/2P1p3/1Q4p1/3P4/3N1PPP/R5K1 b - - 1 24",
                         "2n4r/1k1n3p/1Pq5/4pp2/1Q4p1/3P4/3N1PPP/R5K1 b - - 0 25",
                         "2n4r/1k5p/1nq5/4pp2/1Q4p1/3P4/3N1PPP/R5K1 w - - 0 26",
                         "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
                         "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3",
                         "2n4r/1k5p/1nq5/Q3p3/5pp1/3P4/3N1PPP/R5K1 w - - 0 27",
                         "rnbqk2r/ppp1bppp/4pn2/3p4/2PP1B2/2N5/PP2PPPP/R2QKBNR w KQkq - 4 5",
                         "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3",
                         "rnbqkbnr/pppp1ppp/8/4p3/8/4P3/PPPP1PPP/RNBQKBNR w KQkq e6 0 1",
                         "rnbqkbnr/pp3ppp/2p1p3/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
                         "rnbqkbnr/pp4pp/2p1p3/3p1p2/2PP4/2N1P3/PP3PPP/R1BQKBNR w KQkq f6 0 5",
                         "rnbqkbnr/pp4pp/2p1p3/3p1p2/2PP4/2N1PN2/PP3PPP/R1BQKB1R b KQkq - 1 5",
                         "rnbqkbnr/pp3ppp/2p1p3/3p4/2PP4/2N1P3/PP3PPP/R1BQKBNR b KQkq - 0 4",
                         "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/4P3/PPP2PPP/RNBQKBNR w KQkq - 0 1",
                         "r1bqkbnr/pppp1ppp/2n5/3Pp3/8/4P3/PPP2PPP/RNBQKBNR b KQkq - 0 1",
                         "rnbqkbnr/p1pppppp/1p6/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
                         "r1bqkbnr/ppppnppp/8/3Pp3/8/4P3/PPP2PPP/RNBQKBNR w KQkq - 0 1",
                         "2n4r/2k4p/R7/2P1p3/5pp1/8/5PPP/6K1 w - - 1 32",
                         "rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
                         "rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                         "rnbqkbnr/pppp1ppp/8/4p3/3P4/4P3/PPP2PPP/RNBQKBNR b KQkq d3 0 1",
                         "rnbqkbnr/p1pppp1p/6p1/1p6/1P6/8/P1PPPPPP/RNBQKBNR w KQkq b6 0 1",
                         "2n5/2k4p/4R3/2P1p3/5pp1/6P1/5PKP/3r4 b - - 2 34",
                         "rnbqkbnr/pppppp1p/6p1/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq b3 0 1",
                         "rnbqkbnr/pp1ppppp/2p5/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 1",
                         "rn1qkbnr/pbpppp1p/1p4p1/8/3P4/2P2P2/PP2P1PP/RNBQKBNR w KQkq - 0 1",
                         "rnbqkbnr/p1pppppp/1p6/8/8/2P2P2/PP1PP1PP/RNBQKBNR b KQkq - 0 1",
                         "rnbqk1nr/pppp1ppp/8/4p3/1b2P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
                         "rnbqk1nr/pppp1ppp/8/4p3/1b2P3/2N4P/PPPP1PP1/R1BQKBNR b KQkq - 2 3",
                         "rnbqk1nr/pppp1ppp/8/4p3/1b2PP2/2N5/PPPP2PP/R1BQKBNR b KQkq f3 2 3",
                         "rnbqk1nr/pppp1ppp/8/8/1b2Pp2/2N5/PPPP2PP/R1BQKBNR w KQkq - 0 4",
                         "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
                         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                         "7R/1p2b1p1/p3pk2/4N2p/1P1BP3/P4PPn/3r4/5K2 w - - 5 32",
                         "6k1/1p2bppp/p3p3/3nN3/1P2P3/PR3P2/1Br3PP/5K2 b - - 0 24",
                         "2rq1rk1/pp2b1p1/2R4p/5p1P/3P4/4BQP1/PP3P2/1B2R1K1 b - - 0 24",
                         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                         "rnbqk1nr/pp1p1ppp/2p5/8/1b1PPB2/2N5/PPP3PP/R2QKBNR b KQkq - 0 5",
                         "rnbqk1nr/pp1p1ppp/2p5/8/1b1PPp2/2N5/PPP3PP/R1BQKBNR w KQkq - 0 5",
                         "r4rk1/pp3pbp/2p1b1p1/q2p4/2PPpN2/1P2P1P1/P4PBP/2RQ1RK1 w - - 0 15",
                         "rnb1k1nr/pp2qppp/2p5/3p4/3PPB2/2PB2P1/P1P4P/R2QK1NR w KQkq d6 0 9",
                         "rnb1k1nr/pp2q1pp/2p2p2/3pP3/3P1B2/2PB1NP1/P1P4P/R2QK2R b KQkq - 1 10",
                         "r3k1nr/pp1nq1pp/2p2P2/3p4/3P1B2/2PB1bP1/P1P4P/R2QR1K1 b kq - 0 13",
                         "r3k1nr/pp2q1pp/2p2n2/3p4/3P1B2/2PB1bP1/P1P4P/R2QR1K1 w kq - 0 14",
                         "2kr1r2/pp2n1pp/2p5/3p2PQ/3P1B2/2PB4/P1P4P/R5K1 b - - 0 18",
                         "8/ppk3pp/2p5/2Qp2P1/3P1r2/2PB4/P1P4P/6K1 w - - 3 24",
                         "8/pp2Q1pp/1k6/2pp1BP1/3P3r/2P4P/P1P5/6K1 w - - 0 27",
                         "8/p5pp/k7/1pQp2P1/3P3r/2PB3P/P1P5/6K1 w - b6 0 29",
                         "r1bqkbnr/pp1p1ppp/2n1p3/1Bp5/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 1 4",
                         "r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                         "8/pp4pp/k7/2Qp2P1/3P3r/2PB3P/P1P5/6K1 b - - 2 28",
                         "rnbqkbnr/ppp1pp1p/3p2p1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
                         "rnbq1rk1/ppp1ppbp/3p1np1/8/2PP4/2N1PN2/PP2BPPP/R1BQK2R b KQ - 4 6",
                         "r3k1nr/pp3ppp/2p1b2q/1Bbp4/5B2/2N2NP1/PPP1Q3/R3K2R w KQkq - 4 14",
                         "r1b2rk1/pp2qpb1/2p3pp/2n1p3/2P1Pn2/2N1BN1P/PPQ2PP1/R2R1BK1 w - - 6 15",
                         "r1b2rk1/pp2bppp/2n1p3/q6n/2B2B2/P1N1PN2/1PQ2PPP/3RK2R w K - 1 13",
                         "r5k1/p5b1/2p4p/4p2P/2Pn4/2N1B2r/P4PK1/1R1R4 b - - 1 28",
                         "r1bqk2r/pppn1pb1/5npp/4p3/2P1P3/2N1BN2/PP2BPPP/R2QK2R b KQkq - 1 9",
                         "3rb2k/1p2bp2/p1n4Q/5B2/3q4/P5N1/5PPP/3R2K1 b - - 3 31",
                         "r5k1/p5b1/2p4p/4p2P/2Pn4/2N1B2r/P4P2/1R1R2K1 w - - 0 28",
                         "r1bqk1nr/pppn1pbp/3p2p1/4p1B1/2PPP3/2N2N2/PP3PPP/R2QKB1R b KQkq - 1 6",
                         "rnb1kbnr/ppp1pppp/8/3q4/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
                         "rnbqkbnr/ppp1pp1p/3p2p1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3",
                         "r1b1kbnr/ppp1pppp/2n5/q7/8/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 5",
                         "r1b1kbnr/ppp1pppp/2n5/q7/3P4/2N2N2/PPP2PPP/R1BQKB1R b KQkq d3 0 5",
                         "5r1k/2R1nBpp/4P3/p1Q5/8/4P3/5PKP/3q4 w - - 4 36",
                         "r1b1kbnr/ppp2ppp/2n5/qB6/3p4/2N2N2/PPP2PPP/R1BQK2R w KQkq - 0 7",
                         "r1b1kb1r/ppp1nppp/2n5/qB6/1P1Q4/2N2N2/P1P2PPP/R1B1K2R b KQkq b3 0 8",
                         "rnb1kbnr/ppp1pppp/8/q7/8/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 3 4",
                         "rnbqkbnr/ppp1pppp/3p4/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
                         "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
                         "rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR b KQkq - 1 3",
                         "r1b1k1nr/pp3ppp/2p2q2/1Bbp4/8/2N2NP1/PPP5/R1BQK2R w KQkq - 0 12",
                         "r2q1k2/1p1bbp2/p1n1p1pp/8/5P2/P4NN1/1PQ2PPP/1B1R2K1 w - - 0 21",
                         "5rk1/p4pp1/1nR4p/1P2Q3/8/1q2PP2/5P1P/2R3K1 w - - 0 30",
                         "rnbqkbnr/ppp2ppp/8/3p4/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq - 1 4",
                         "r1bqkbnr/ppp2ppp/2n5/3p4/3P4/2P2N2/PP3PPP/RNBQKB1R b KQkq - 0 5",
                         "4rrk1/p1p5/1pnb3p/3p1np1/2PP1p2/1PB2N2/P1N2PPP/R3R1K1 w - - 0 19",
                         "r4rk1/ppq2pb1/2n3p1/2p1p2p/P1N1P2P/2P2BP1/1PQ2P2/R3R1K1 b - a3 0 20",
                         "r4rk1/ppq2p2/2n3pb/2p1p2p/P1N1P2P/2P2BP1/1PQ2P2/R3R1K1 w - - 1 21",
                         "r4rk1/ppq2p2/2n2bp1/2p1p2p/2N1P3/2P2BPP/PPQ2P2/R3R1K1 w - - 0 19",
                         "rn1qk2r/pp2bppp/3ppn2/2p4b/4P3/2P2N1P/PP1PBPP1/RNBQ1RK1 w kq - 2 8",
                         "rn1qk2r/pp2bppp/3ppn2/2p4b/4P3/2PP1N1P/PP2BPP1/RNBQ1RK1 b kq - 0 8",
                         "r3k2r/pppqnppp/2nb4/3p4/3P4/2PQNN2/PP3PPP/R1B1K2R b KQkq - 2 10",
                         "rnbqkb1r/pp2pppp/3p1n2/2p5/4P3/2P2N2/PP1P1PPP/RNBQKB1R w KQkq - 1 4",
                         "rn1qkb1r/pp3ppp/3ppn2/2p4b/4P3/2P2N1P/PP1PBPP1/RNBQK2R w KQkq - 0 7",

                         "rn1qkb1r/pp2pppp/3p1n2/2p4b/4P3/2P2N1P/PP1PBPP1/RNBQK2R b KQkq - 2 6",
                         "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
                         "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
                         "r4rk1/pppqnppp/2nb4/3p4/3P4/2PQNN2/PP3PPP/R1B1K2R w KQ - 3 11",
                         "r1bqk2r/ppp1nppp/2nb4/3p4/3P4/2PB1N2/PP3PPP/RNBQK2R w KQkq - 3 7",
                         "4rrk1/ppp3pp/2nb4/3p1n2/3P1p2/2P2N2/PPNB1PPP/R3R1K1 w - - 2 16",
                         "5rk1/p4ppp/1nR2b2/1P2p3/4Q3/qP2PPB1/5P1P/2R3K1 b - - 0 27",
                         "5rk1/p4ppp/1nR2b2/1P2p3/4Q3/1q2PPB1/5P1P/2R3K1 w - - 0 28",
                         "r2qk2r/ppp1nppp/2nb4/3p4/3P4/2Pb1N2/PP3PPP/R1BQKN1R w KQkq - 0 9",
                         "r1b2rk1/pp2bppp/2n1pn2/q7/2B2B2/P1N1PN2/1PQ2PPP/3RK2R b K - 0 12",
                         "r1b1k1nr/pp3ppp/2p2q2/1Bbp4/8/2N2NP1/PPP1Q3/R1B1K2R b KQkq - 1 12"]
        chosenFEN=np.random.choice(RealisticBoards)

        #Load into the board
        self.board = chess.Board(chosenFEN)
        self.WhoseTurnIsIt = "chess.WHITE" if self.board.turn == chess.WHITE else "chess.BLACK"

        ChangeIt=self.BetterRandomZeroOrOne()
        if len(self.board.piece_map())>30:
            ChangeIt=0
        if ChangeIt == 1:
            AmtOfPeices=np.random.randint(1, 3)
            PieceMap = self.board.piece_map()
            square_indices = set(PieceMap.keys())  #Use set for O(1) lookup

            PossiblePieces = {
                'Q': chess.Piece(chess.QUEEN, chess.WHITE),
                'R': chess.Piece(chess.ROOK, chess.WHITE),
                'B': chess.Piece(chess.BISHOP, chess.WHITE),
                'N': chess.Piece(chess.KNIGHT, chess.WHITE),
                'P': chess.Piece(chess.PAWN, chess.WHITE),
                'q': chess.Piece(chess.QUEEN, chess.BLACK),
                'r': chess.Piece(chess.ROOK, chess.BLACK),
                'b': chess.Piece(chess.BISHOP, chess.BLACK),
                'n': chess.Piece(chess.KNIGHT, chess.BLACK),
                'p': chess.Piece(chess.PAWN, chess.BLACK),
            }

            back_rank_squares = set(range(0, 8)).union(range(56, 64))  #ranks 1 and 8

            for _ in range(AmtOfPeices):
                max_tries = 10  #Prevent infinite loops
                for __ in range(max_tries):
                    sq = np.random.randint(0, 64)
                    if sq in square_indices:
                        continue

                    new_piece_symbol = np.random.choice(list(PossiblePieces.keys()))
                    if new_piece_symbol.lower() == 'p' and sq in back_rank_squares:
                        continue  #Skip illegal pawn placement

                    PieceMap[sq] = PossiblePieces[new_piece_symbol]
                    self.board.set_piece_map(PieceMap)

                    if self.BasicChecks() is None:
                        square_indices.add(sq)
                        break  #valid piece added
                    else:
                        del PieceMap[sq]  #revert
                        self.board.set_piece_map(PieceMap)
        else:
            pass

        return self.board

    def GenerateRandomBoard(self, PossiblePiecesSymbols=['K', 'k', 'Q', 'R', 'R', 'B', 'B', 'N', 'N', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'q', 'r', 'r', 'b', 'b', 'n', 'n', 'p', 'p', 'p', 'p', 'p', 'p','p', 'p']):
        '''
        This became so frusterating it became its own method. It generates a random board that doesn't violate the chess laws.
        '''

        self.board = chess.Board(None)

        AmtOfPeices = np.random.randint(3, 33)  # Will generate the amount of random peices for that specific board configuration
        SquaresTheirAt = np.random.randint(0, 64, size=AmtOfPeices)  # Will randomly generate the square the peice is on, with number of squares (AmtOfPeices) being the number of peices on board
        PieceMap = {}

        PossiblePieces = {'K': chess.Piece(chess.KING, chess.WHITE), 'Q': chess.Piece(chess.QUEEN, chess.WHITE), 'R': chess.Piece(chess.ROOK, chess.WHITE), 'B': chess.Piece(chess.BISHOP, chess.WHITE),
                          'N': chess.Piece(chess.KNIGHT, chess.WHITE), 'P': chess.Piece(chess.PAWN, chess.WHITE), 'k': chess.Piece(chess.KING, chess.BLACK), 'q': chess.Piece(chess.QUEEN, chess.BLACK),
                          'r': chess.Piece(chess.ROOK, chess.BLACK), 'b': chess.Piece(chess.BISHOP, chess.BLACK), 'n': chess.Piece(chess.KNIGHT, chess.BLACK),
                          'p': chess.Piece(chess.PAWN, chess.BLACK)}

        used_squares = set()  #Keeps track of where the peices are to avoid putting multiple on the same one

        for i in range(AmtOfPeices):
            #Finding unused square
            square = SquaresTheirAt[i]
            while square in used_squares:
                square = np.random.randint(0, 64)
            used_squares.add(square)

            if i == 0:
                PieceMap[square] = PossiblePieces['K']  #White King
            elif i == 1:
                PieceMap[square] = PossiblePieces['k']  #Black King
            else:
                choice = np.random.choice([t for t in PossiblePiecesSymbols if t not in ['k', 'K']])
                PieceMap[square] = PossiblePieces[choice]  #Any other piece

        self.board.set_piece_map(PieceMap)
        IsGameOver = self.BasicChecks()
        while (IsGameOver is not None):
            #print(f"Change it\n{IsGameOver}\n{self.board}\n")
            square = np.random.randint(0, 64)
            while square in used_squares:
                square = np.random.randint(0, 64)

            #If pawn problem
            if IsGameOver == "Pawns on backrow":
                PieceMap = self.board.piece_map().copy()
                OldPawnsSquares = []
                for sq, piece in PieceMap.items():
                    if (piece.symbol() == 'p' or piece.symbol() == 'P') and sq > 55 or (piece.symbol() == 'P' or piece.symbol() == 'p') and sq <= 7:
                        OldPawnsSquares.append((sq, piece))

                for sq, piece in OldPawnsSquares:
                    #Remove from PieceMap
                    PieceMap.pop(sq, None)

                    #Find a new square
                    squarespawn = np.random.randint(8, 55)
                    while (squarespawn in PieceMap) or (squarespawn < 8) or (squarespawn > 55):
                        squarespawn = np.random.randint(8, 55)

                    #Place the piece in the new position
                    PieceMap[squarespawn] = piece

                #Update the board with the modified PieceMap
                self.board.set_piece_map(PieceMap)

            elif IsGameOver == "Got a King problem":
                # 1) Resync PieceMap from the current board
                PieceMap = self.board.piece_map().copy()

                # 2) Locate the Black king
                OldBlackKingSquare = next((sq for sq, p in PieceMap.items() if p.symbol() == 'k'), None)

                # 3) Safely remove it and place it at `square`
                if OldBlackKingSquare is not None:
                    PieceMap.pop(OldBlackKingSquare, None)
                    PieceMap[square] = PossiblePieces['k']

                # 4) Write the updated map back to the board
                self.board.set_piece_map(PieceMap)

            #If white won by the default set up
            elif hasattr(IsGameOver, 'winner'):
                if IsGameOver.winner == chess.WHITE:
                    PieceMap = self.board.piece_map().copy()
                    OldBlackKingSquare = next((sq for sq, p in PieceMap.items() if p.symbol() == 'k'), None)
                    if OldBlackKingSquare is not None:
                        PieceMap.pop(OldBlackKingSquare, None)
                        PieceMap[square] = PossiblePieces['k']
                    self.board.set_piece_map(PieceMap)

                #If Black Won by the default set up
                elif hasattr(IsGameOver, 'winner') and IsGameOver.winner == chess.BLACK:
                    # 1) Re‐sync PieceMap from the actual board
                    PieceMap = self.board.piece_map().copy()

                    # 2) Find the White King’s square
                    OldWhiteKingSquare = next((sq for sq, p in PieceMap.items() if p.symbol() == 'K'), None)

                    # 3) Safely remove it and place it on the new random square
                    if OldWhiteKingSquare is not None:
                        PieceMap.pop(OldWhiteKingSquare, None)
                        PieceMap[square] = PossiblePieces['K']

                    # 4) Push the updated map back into the board
                    self.board.set_piece_map(PieceMap)

                else:
                    #winner is None → add a bishop to avoid insufficient material
                    PieceMap = self.board.piece_map().copy()

                    newPiece = np.random.choice(['B', 'b'])
                    #Find an empty square
                    squarespawn = np.random.randint(0, 64)
                    while squarespawn in PieceMap:
                        squarespawn = np.random.randint(0, 64)

                    PieceMap[squarespawn] = PossiblePieces[newPiece]
                    self.board.set_piece_map(PieceMap)

            IsGameOver = self.BasicChecks()

        #Temporarily check “if White is in check?”
        self.board.turn = chess.WHITE
        white_in_check = self.board.is_check()

        #Then check “if Black is in check?”
        self.board.turn = chess.BLACK
        black_in_check = self.board.is_check()

        if white_in_check and not black_in_check:
            #White king is attacked, so it MUST be White's move
            self.WhoseTurnIsIt = "chess.WHITE"
            self.board.turn = chess.WHITE
        elif black_in_check and not white_in_check:
            #Black king is attacked, so it MUST be Black's move
            self.WhoseTurnIsIt = "chess.BLACK"
            self.board.turn = chess.BLACK
        elif white_in_check and black_in_check:
            #Both kings in check is an impossible/illegal position → regenerate
            return self.GenerateRandomBoard(PossiblePiecesSymbols)
        else:
            #Neither side is in check → pick a side at random
            if self.BetterRandomZeroOrOne() == 0:
                self.WhoseTurnIsIt = "chess.WHITE"
                self.board.turn = chess.WHITE
            else:
                self.WhoseTurnIsIt = "chess.BLACK"
                self.board.turn = chess.BLACK

        return self.board

    # ============== STOCKFISH ==============

    def IsStockFishALive(self):
        """Check if Stockfish is alive"""
        output=self.stockfish.is_fen_valid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        if output == True:
            return True #Its alive
        else:
            return False #Not alive, restart

    def reset_stockfish(self):
        """Reset Stockfish engine - useful if it crashes"""
        print("Resetting Stockfish...")

        #First, cleanup existing instance
        if self.stockfish is not None:
            try:
                #Try to send quit command
                self.stockfish._put("quit")
                time.sleep(0.1)
            except:
                pass  #Ignore errors

            try:
                #Try to terminate the process
                if hasattr(self.stockfish, '_stockfish') and self.stockfish._stockfish:
                    self.stockfish._stockfish.terminate()
                    #Wait briefly for termination
                    try:
                        self.stockfish._stockfish.wait(timeout=0.5)
                    except:
                        #Force kill if terminate didn't work
                        try:
                            self.stockfish._stockfish.kill()
                        except:
                            pass
            except:
                pass

            self.stockfish = None

        #Try to reinitialize
        try:
            self.stockfish = Stockfish(r"C:\Users\iwana\OneDrive\Documents\stockfish\stockfish-windows-x86-64-avx2.exe")
            self.stockfish.set_depth(self.depth)
            self.stockfish.set_skill_level(self.skill)
            print("Stockfish reset successfully")
            return True
        except Exception as e:
            print(f"Failed to reset Stockfish: {e}")
            self.stockfish = None
            return False

    def cleanup_stockfish(self):
        """Properly close Stockfish when program exits"""
        try:
            if self.stockfish is not None:
                #Send quit command to Stockfish
                self.stockfish._put("quit")
                #Give it a moment to process
                time.sleep(0.1)
                #Terminate the process if it exists
                if hasattr(self.stockfish, '_stockfish') and self.stockfish._stockfish:
                    self.stockfish._stockfish.terminate()
                    try:
                        self.stockfish._stockfish.wait(timeout=1)
                    except:
                        pass
                self.stockfish = None
                print("Stockfish cleaned up successfully")
        except Exception as e:
            print(f"Error during Stockfish cleanup: {e}")

    def get_stockfish_evaluation(self, board):
        """Get Stockfish's evaluation of current position, if it takes more than 1 second, it will restart the worker thread"""
        def GoEval():
            """The actual Evaluation process"""
            try:
                if self.IsStockFishALive()==True:
                    pass
            except Exception as e:
                print(f"Dead Stockfish in GoEval: {e}")
                output=self.reset_stockfish()
                if output==True:
                    print("Stockfish died but we brought it back to life")
                else:
                    print("Failed to reset Stockfish, look into this")
                return (0, False)

            try:
                #Set the turn based on whose turn it is
                if self.WhoseTurnIsIt == "chess.WHITE":
                    board.turn = chess.WHITE
                else:
                    board.turn = chess.BLACK

                mate_in=False
                EvalValue=0.0

                self.stockfish.set_fen_position(board.fen())
                self.stockfish.set_depth(self.depth)
                evaluation = self.stockfish.get_evaluation()

                if evaluation['type'] == 'cp':
                    #Centipawn evaluation
                    EvalValue = evaluation['value'] / 100.0
                elif evaluation['type'] == 'mate':
                    #Mate in X moves
                    mate_in = evaluation['value']
                    if mate_in>0:
                        EvalValue=100/mate_in
                    elif mate_in<0:
                        EvalValue=-100/abs(mate_in)
                    else:
                        EvalValue=100 #It checkmated! Aka mate_in=0
                else:
                    EvalValue=0.0

                EvalValue=EvalValue/100
                return (EvalValue, mate_in)

            except Exception as e:
                print(f"Dead Stockfish in GoEval: {e}")
                output = self.reset_stockfish()
                if output == True:
                    print("Stockfish died but we brought it back to life")
                else:
                    print("Failed to reset Stockfish, look into this")
                return (0, False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(GoEval)
            try:
                result = future.result(timeout=self.timeout)
                return result
            except concurrent.futures.TimeoutError:
                print(f"Stockfish evaluation timed out after {self.timeout} seconds")
                # Cancel the future
                future.cancel()
                # Shutdown executor with wait=False to avoid hanging
                executor.shutdown(wait=False, cancel_futures=True)
                # self.youcomehereoften+=1
                #
                # if self.youcomehereoften <= 1:
                #     self.youcomehereoften = 3

                # Reset Stockfish and try once more with a simpler depth
                self.olddepth=self.depth
                self.depth = self.depth-3*self.youcomehereoften  #Reduce depth for retry
                print(f"Attempting evaluation with reduced depth... {self.depth}")

                try:
                    if self.reset_stockfish():
                        #Try again with reduced depth
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor2:
                            future2 = executor2.submit(GoEval)
                            try:
                                result = future2.result(timeout=self.timeout*3)
                                return result
                            except concurrent.futures.TimeoutError:
                                self.youcomehereoften=self.youcomehereoften+1
                                if self.youcomehereoften<=1:
                                    self.youcomehereoften=3
                                future2.cancel()
                                executor2.shutdown(wait=False, cancel_futures=True)
                                print("Second attempt also timed out")
                                self.youcomehereoften=1 #Resets counter
                finally:
                    self.depth=self.olddepth

                #If all attempts fail, return a default value
                self.youcomehereoften = 1  #Resets counter
                return (0.0, False)

    def get_stockfish_best_move(self, board):
        """Get Stockfish's evaluation of current position, if it takes more than 1 second, it will restart the worker thread"""
        def GoEvalBestMove():
            """The actual Best Move Evaluation process"""
            try:
                if self.IsStockFishALive() == True:
                    pass
            except Exception as e:
                print(f"Dead Stockfish in GoEval: {e}")
                output = self.reset_stockfish()
                if output == True:
                    print("Stockfish died but we brought it back to life")
                else:
                    print("Failed to reset Stockfish, look into this")
                return (0, False)

            try:
                # Set the turn based on whose turn it is
                if self.WhoseTurnIsIt == "chess.WHITE":
                    board.turn = chess.WHITE
                else:
                    board.turn = chess.BLACK

                self.stockfish.set_fen_position(board.fen())
                self.stockfish.set_depth(self.depth)
                best_move = self.stockfish.get_best_move()

                if best_move:
                    return chess.Move.from_uci(best_move)
                return None

            except Exception as e:
                print(f"Dead Stockfish in GoEval: {e}")
                output = self.reset_stockfish()
                if output == True:
                    print("Stockfish died but we brought it back to life")
                else:
                    print("Failed to reset Stockfish, look into this")
                return (0, False)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(GoEvalBestMove)
            try:
                result = future.result(timeout=self.timeout)
                return result
            except concurrent.futures.TimeoutError:
                print(f"Stockfish evaluation timed out after {self.timeout} seconds")
                # Cancel the future
                future.cancel()
                # Shutdown executor with wait=False to avoid hanging
                executor.shutdown(wait=False, cancel_futures=True)

                # Reset Stockfish and try once more with a simpler depth
                self.olddepth=self.depth
                self.depth = self.depth-3*self.youcomehereoften  #Reduce depth for retry
                print(f"Attempting evaluation with reduced depth... {self.depth}")

                try:
                    if self.reset_stockfish():
                        #Try again with reduced depth
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor2:
                            future2 = executor2.submit(GoEvalBestMove)
                            try:
                                result = future2.result(timeout=self.timeout*3)
                                return result
                            except concurrent.futures.TimeoutError:
                                self.youcomehereoften=self.youcomehereoften+1
                                if self.youcomehereoften<=1:
                                    self.youcomehereoften=3
                                future2.cancel()
                                executor2.shutdown(wait=False, cancel_futures=True)
                                print("Second attempt also timed out")
                finally:
                    self.depth=self.olddepth

                #If all attempts fail, return a default value
                return (0.0, False)

    # ============== SELECT MOVE ==============

    def ExplorationRateThreshold(self):
        '''This will randomly determine if it explores or exploits while keeping it consistent whenever it's used (Using the "self." part, allowing a pointer).'''
        self.ExplorThreshold=np.random.rand()

    def select_move(self, board=None):
        """Selects a move based on current board position, actually does it in a different function."""
        if board is None:
            board = self.board

        state = self.board_to_tensor(board)  # Convert board to 12 input tensor

        if self.WhoseTurnIsIt=="chess.WHITE":
            board.turn=chess.WHITE
        else:
            board.turn=chess.BLACK

        #Get all legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves: #If it's empty
            return None #This shouldn't happen but if it does it wont blow things up

        self.ExplorationRateThreshold() #Gets a random number which is then used to decide if it wants to explore or exploit
        if self.ExplorThreshold < self.epsilon:
            return np.random.choice(legal_moves)
        else:
            #Exploit, dont explore

            #Get Q-values for all moves
            with torch.no_grad():
                q_values = self.forward(state).squeeze()

            #Checks for what q-values are legal moves
            legal_move_indices = [self.move_to_index(move) for move in legal_moves]
            legal_q_values = q_values[legal_move_indices]

            #Find the best legal move
            best_move_idx = torch.argmax(legal_q_values).item()
            best_move = legal_moves[best_move_idx]

            return best_move

    # ============== REPLAY ==============

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def ReplayBatch(self, Elaborate=False):
        """Train on a batch of experiences from replay buffer"""

        #Get a random batch from the memory deque
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]

        #Prepare batch data, all this is current state
        states = torch.cat([exp[0] for exp in batch])
        actions = [exp[1] for exp in batch]
        CurrentRewards = torch.tensor([exp[2] for exp in batch], device=device, dtype=torch.float32)
        next_states = torch.cat([exp[3] for exp in batch])
        dones = torch.tensor([exp[4] for exp in batch], device=device, dtype=torch.float32)

        #Current Q-values
        CurrentQValues = self.forward(states)
        batch_idx = torch.arange(len(actions), device=device)
        action_idx = torch.tensor(actions, device=device)
        ActionStateQValues = CurrentQValues[batch_idx, action_idx]

        #Next Q-values for calculating targets
        with torch.no_grad():
            next_q_values = self.target_forward(next_states)
            maxNextQ=torch.max(next_q_values, dim=1)[0]
            targetQ=maxNextQ.clone()

            #Calculate target Q-values using Bellman equation
            TargetQValuEs=(CurrentRewards+self.γ*maxNextQ*(1-dones)) #The (1-dones) is because if it's done then it's a terminal state and there is no future reward
            if Elaborate==True:
                print(f"States:\n{self.tensor_to_board(states)}\nActions:\n{actions}\nCurrent Rewards:\n{CurrentRewards}\nNext States:\n{self.tensor_to_board(next_states)}\nDones:\n{dones}\nCurrentQValues:\n{CurrentQValues}\nmaxNextQ:\n{maxNextQ}\ntargetQ:\n{targetQ}\n")

        #Calculate loss and update network
        loss=self.LossFunction(ActionStateQValues, TargetQValuEs)
        print(f"IMPORTANT: Loss for this batch is {loss}")
        self.lossBox=np.append(self.lossBox,loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.target_update_counter+=1

        if self.target_update_counter%self.target_update_freq==0:
            self.update_target_network()
            print("Target network updated")

    # ============== TRAINING ==============

    def Train(self, CurrentBot, Elaborate=False, ExtraElaborate=False):
        '''
        In this part of the training loop, the AI will be using Stockfish. This will be constantly asking,
        "Was the board better or worse after my move?" The board will be scored by stockfish. Weights will be adjusted using DQN from NathanChessEngine
        '''

        try:
            #Validate Stockfish before starting
            if not self.IsStockFishALive():
                print("Stockfish validation failed, attempting reset...")
                if not self.reset_stockfish():
                    print("Failed to reset Stockfish, skipping this episode")
                    return "Skip this Episode"

            #Get random board
            WhichRandomBoard=self.BetterRandomZeroOrOne()
            if WhichRandomBoard==0:
                #RNGBoard = CurrentBot.GenerateRandomBoard()
                RNGBoard= CurrentBot.AltGenerateRandomBoard()
            else:
                RNGBoard= CurrentBot.AltGenerateRandomBoard()
            if Elaborate==True:
                print(RNGBoard)
                print(f"It is {"White's" if self.WhoseTurnIsIt=="chess.WHITE" else "Black's"} turn.")

            #Get position evaluations
            try:
                EvaluationBefore, m8inB = self.get_stockfish_evaluation(RNGBoard)
            except Exception as e:
                print(f"Error in get_stockfish_evaluation: {e}")
                return "Skip this Episode"

            #Get our move
            yhat=CurrentBot.select_move(RNGBoard)
            if yhat is None:
                print("No legal moves, skipping this episode")
                return "Skip this Episode"

            #Do our move
            BoardAfter=RNGBoard.copy()
            BoardAfter.push(yhat)

            #Evaluation the board after our move
            try:
                EvaluationAfter, m8inA = self.get_stockfish_evaluation(BoardAfter)
            except Exception as e:
                print(f"Error in post get_stockfish_evaluation move: {e}")
                return "Skip this Episode"

            #Get stockfish Best Move
            try:
                yi=CurrentBot.get_stockfish_best_move(RNGBoard)
            except Exception as e:
                print("Error in get_stockfish_best_move")
                return "Skip this episode"

            if Elaborate==True and yi is not None:
                print(f"My Move: {yhat.uci()}, Mate in: {m8inB}\nStockfish Move: {yi.uci()}, Mate in: {m8inA}\n")

            #Calculate reward
            if RNGBoard.turn == chess.WHITE:  #This is needed becuase stockfish evaluates from Whites Prespective
                reward=EvaluationAfter-EvaluationBefore
                reward = np.abs(EvaluationAfter * 2) if np.round(EvaluationAfter, 5) == np.round(EvaluationBefore, 5) else reward
            elif RNGBoard.turn == chess.BLACK:
                reward=(EvaluationAfter-EvaluationBefore)*-1
                reward = (EvaluationAfter + EvaluationBefore) * -1 if np.round(EvaluationAfter, 5) == np.round(EvaluationBefore, 5) else reward
            if Elaborate==True:
                print(f"EvaluationBefore: {EvaluationBefore}\nEvaluationAfter: {EvaluationAfter}\nReward: {reward}\n")

            #Store experience in replay buffer
            DONE=BoardAfter.is_game_over()
            CurrentBot.remember(CurrentBot.board_to_tensor(RNGBoard), CurrentBot.move_to_index(yhat), reward, CurrentBot.board_to_tensor(BoardAfter), DONE)
            if len(CurrentBot.memory) > CurrentBot.batch_size:
                CurrentBot.ReplayBatch(Elaborate=ExtraElaborate)
            else:
                if Elaborate==True:
                    print(f"Not enough experiences in memory ({len(CurrentBot.memory)}/{CurrentBot.batch_size}). Skipping replay.")
                else:
                    pass

        except Exception as e:
            print(f"Unexpected error in training: {e}")
            import traceback
            traceback.print_exc()

    # ============== Target Network ==============

    def target_forward(self, x):
        """Forward pass through target network"""
        push1 = F.elu(self.target_bn1(self.target_conv1(x)))
        push2 = F.elu(self.target_bn2(self.target_conv2(push1)))
        push3 = F.elu(self.target_bn3(self.target_conv3(push2)))
        push4 = push3.view(-1, 256 * 8 * 8)
        push5 = F.elu(self.target_fc1(push4))
        QvaluEs = self.target_fc2(push5)
        return QvaluEs

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_conv1.load_state_dict(self.conv1.state_dict())
        self.target_bn1.load_state_dict(self.bn1.state_dict())
        self.target_conv2.load_state_dict(self.conv2.state_dict())
        self.target_bn2.load_state_dict(self.bn2.state_dict())
        self.target_conv3.load_state_dict(self.conv3.state_dict())
        self.target_bn3.load_state_dict(self.bn3.state_dict())
        self.target_fc1.load_state_dict(self.fc1.state_dict())
        self.target_fc2.load_state_dict(self.fc2.state_dict())

    # ============== PROSPERITY ==============

    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")