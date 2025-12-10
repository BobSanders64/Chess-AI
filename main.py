import chess
import chess.svg
import webbrowser
import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
from NathanEngine import NathanChessEngine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def run(LossDesired=.005):
    realstart=time.time()
    '''NOTE TO FUTURE ME: THE WAY THE BOARD IS PRINTED IS A BIT BACKWARDS. THE FIRST ROW (FROM TENSORBOARD) IS THE BOTTOM OF THE VISUAL BOARD (RPBOARD).'''
    NathanBot = NathanChessEngine(training=True)
    defaultspot=r"C:\Users\iwana\OneDrive\Documents\Python Scripts\NathanChessAI\NathanChessEngine.pth"

    def PlotMyCost():
        """Self expanitory"""
        import matplotlib.pyplot as plt
        import numpy as np
        steps=np.arange(len(NathanBot.lossBox))
        plt.plot(steps,NathanBot.lossBox, label='Training loss')
        plt.ylabel('Cost')
        plt.title("Nathan Chess Engine Training Loss Over Time")
        plt.legend()
        plt.xlabel('Training Steps')
        plt.show()

    try:
        for i in range(NathanBot.MaxEpisodesToTrain):
            start = time.time()
            time.sleep(25E-2)  #Waits for a sec because we dont wanna jam Stockfish
            print(f"\n=== Episode {i}/{NathanBot.MaxEpisodesToTrain - 1} ===")
            try:
                NathanBot.Train(NathanBot, Elaborate=True, ExtraElaborate=False)
                #Decay exploration rate
                NathanBot.epsilon = max(NathanBot.epsilon_min,
                                        NathanBot.epsilon * (1 - NathanBot.epsilon_decay_rate))
                if NathanBot.epsilon < NathanBot.epsilon_min:
                    NathanBot.epsilon = NathanBot.epsilon_min
                if i>NathanBot.batch_size+4:
                    if NathanBot.lossBox[-1]<LossDesired: #ALTERNATIVE END CONDITION
                        break
                print(f"Episode time: {time.time() - start:.2f}s")
            except Exception as e:
                print(f"Error in episode {i + 1}: {e}")
                import traceback

                traceback.print_exc()
                continue  # Continue to next episode

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Clean up only once at the very end
        print("\nCleaning up Stockfish...")
        NathanBot.cleanup_stockfish()
        print("Training complete!")
        elapsed = time.time() - realstart
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)

        print(f"Total training time: {int(hrs)}h {int(mins)}m {int(secs)}s")

        while True:
            try:
                botpath=str(input(f"Where do you want to save the model? ('skip' to skip, 'default' to save in default location with default name)\n{defaultspot} "))
                if botpath.lower() == "skip":
                    break
                elif botpath.lower() == "default":
                    NathanBot.save_model(f"{defaultspot}")
                    break
                else:
                    NathanBot.save_model(botpath)
                    break
            except Exception as e:
                print(f"Please try again, if you don't want to save it type 'Skip' or 'skip'\nError: {e}\n")
        PlotMyCost()

if __name__ == "__main__":
    run(LossDesired=.003)
# while Gamemode==True:
#     print(board) #Prints ascii art diagram
#
#     legallist = auto(board.legal_moves)
#     WasItALegalMove = False
#     while WasItALegalMove == False:
#         move = input("Enter move: ")
#         try:
#             board.parse_san(move)
#             WasItALegalMove = True
#         except ValueError:
#             pass
#     WasItALegalMove = False
#
#     board.push_san(move) #Does the action
#
#     #Render the board as SVG, highlighting the last move:
#     svg = chess.svg.board(
#         board=board, #Which position should I draw? Oh, ill draw board
#         lastmove=board.peek(),        #highlight the most recent move
#         size=700                     #size in pixels
#     )
#
#     pm = board.piece_map() #(lowercase=black, uppercase=white)
#     #"{key: value for item in iterable if condition}" is the default dict compression
#     WhitePieces = {square: piece for square, piece in pm.items() if piece.symbol().isupper()}
#     BlackPieces = {square: piece for square, piece in pm.items() if square not in WhitePieces}
#
#     #print(f"WhitePeices are \n{WhitePieces}")
#     #print(f"BlackPeices are \n{BlackPieces}")
#     #print(svg)
#
#     # Write it to a temporary file and launch your default browser:
#     path = os.path.abspath("current_board.svg")
#     with open(path, "w") as f:
#         f.write(svg)
#
#     webbrowser.open("file://" + path, new=0)  # open in existing browser window/tab
#
#     turn=turn+1