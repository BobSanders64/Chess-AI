import chess
import chess.svg
import webbrowser
import os
import torch
import numpy as np
from NathanEngine import NathanChessEngine

global board, NathansGreatChessEngine

def CommandLineGame(HumanColor):
    #1) Is it players turn or not?
    human_turn = (board.turn == chess.WHITE and HumanColor == "chess.WHITE") or \
                 (board.turn == chess.BLACK and HumanColor == "chess.BLACK")

    if human_turn:
        #2) Allow human to move if its their turn
        print(f"\n{board}\n")
        prettylegal = [board.san(legMoves) for legMoves in board.legal_moves]
        print(f"Type one of the following legal moves:\n{prettylegal}\n")
        playersinput = input(f"NOTE: WHITE IS ON BOTTOM ALWAYS\nEnter your move: ")
        didplayertypelegalmove = False
        while didplayertypelegalmove == False:
            if playersinput in prettylegal:
                board.push_san(playersinput)
                didplayertypelegalmove = True
                print(board)
            else:
                print(f"Please enter a legal move!\n{board}\n{prettylegal}\n")
                playersinput = input("Enter your move: ")

    else:
        #3) Allow ai to move
        print("Engine is thinking...")
        with torch.no_grad():
            move = NathansGreatChessEngine.select_move(board)
        print(f"Engine plays: {board.san(move)}")
        board.push(move)

    #4) Sync engine’s internal state
    NathansGreatChessEngine.board = board.copy()

def WebGame(HumanColor):
    # Determine whose turn it is
    human_turn = (board.turn == chess.WHITE and HumanColor == "chess.WHITE") or \
                 (board.turn == chess.BLACK and HumanColor == "chess.BLACK")

    if human_turn:
        # Human move (same SAN input as CLI)
        print(f"\n{board}\n")
        prettylegal = [board.san(m) for m in board.legal_moves]
        print(f"Type one of the following legal moves:\n{prettylegal}\n")
        while True:
            move_input = input("Enter your move: ")
            try:
                board.push_san(move_input)
                break
            except ValueError:
                print("Invalid move, please try again.")
    else:
        # Engine move
        print("Engine is thinking...")
        prettylegalAI = [board.uci(m) for m in board.legal_moves]
        i=0
        while True:
            with torch.no_grad():
                ai_move = NathansGreatChessEngine.select_move(board)
            if ai_move is not None and ai_move.uci() in prettylegalAI: #Begins checks to make sure the move selected was legal
                break
            if i>3:
                ai_move=np.random.choice(list(board.legal_moves))
                break
            else:
                i=i+1
                continue
        print(f"Engine plays: {board.san(ai_move)}")
        board.push(ai_move)

    # Update engine state
    NathansGreatChessEngine.board = board.copy()

    # Render to SVG and open in browser
    svg = chess.svg.board(
        board=board,
        lastmove=board.peek(),
        size=700
    )
    path = os.path.abspath("current_board.svg")
    with open(path, "w") as f:
        f.write(svg)
    webbrowser.open(f"file://{path}", new=0)

#1- Load in the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NathansGreatChessEngine = NathanChessEngine(training=False)
model_path = os.path.join(os.path.dirname(__file__), 'NathanChessEngine 14500.pth')
NathansGreatChessEngine.load_model(model_path)
# Move to device, set eval mode, disable exploration
NathansGreatChessEngine.to(device)
NathansGreatChessEngine.eval()
NathansGreatChessEngine.epsilon = 0.0

gamemode=True
initalgame=False
board = chess.Board()
WebOrCommandLine="Web"
while gamemode==True:
    #2- Allow the player to pick their color
    while initalgame==False:
        playerside=input("Do you want to be white or black? (white/black)\n").strip().lower()
        if playerside=="white" or playerside=="w":
            NathansGreatChessEngine.WhoseTurnIsIt = "chess.BLACK"
            HumanColor="chess.WHITE"
            initalgame=True
        elif playerside=="black" or playerside=="b":
            NathansGreatChessEngine.WhoseTurnIsIt = "chess.WHITE"
            HumanColor="chess.BLACK"
            initalgame=True
        else:
            print("Please enter 'white' or 'black'!")
            continue

    #3- Allow player to choose medium played though
    WOC=input("Do you want to play on a Web Browser (with a pretty image) or command line? (Web/Command Line)\n").strip().lower()
    if WOC=="command line" or WOC=="cli":
        #4- Run the game
        while not board.is_game_over():
            CommandLineGame(HumanColor=HumanColor)
        print(f"The result is {board.outcome(claim_draw = True)}")
        exit()
    if WOC=="web" or WOC=="browser":
        while not board.is_game_over():
            WebGame(HumanColor)
        print(f"The result is {board.outcome(claim_draw=True)}")
        exit()