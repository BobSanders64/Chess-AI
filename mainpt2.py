import chess
import chess.svg
import webbrowser
import numpy as np
import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
from NathanEngine import NathanChessEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def run(LossDesired=.005):
    realstart = time.time()
    '''NOTE TO FUTURE ME: THE WAY THE BOARD IS PRINTED IS A BIT BACKWARDS. THE FIRST ROW (FROM TENSORBOARD) IS THE BOTTOM OF THE VISUAL BOARD (RPBOARD).'''

    # Initialize bot
    NathanBot = NathanChessEngine(training=True)
    # Default path to saved model
    defaultspot = r"C:\Users\iwana\OneDrive\Documents\Python Scripts\NathanChessAI\NathanChessEngine 7500.pth"
    changespot = r"C:\Users\iwana\OneDrive\Documents\Python Scripts\NathanChessAI\\"


    # Load existing model if available to continue training
    if os.path.isfile(defaultspot):
        print(f"Loading existing model from {defaultspot}...")
        try:
            NathanBot.load_model(defaultspot)
        except Exception as e:
            print(f"Warning: failed to load model: {e}")
    else:
        print(f"No pre-trained model found at {defaultspot}, starting training from scratch.")

    def PlotMyCost():
        """Self explanatory"""
        import matplotlib.pyplot as plt
        import numpy as np
        steps = np.arange(len(NathanBot.lossBox))
        plt.plot(steps, NathanBot.lossBox, label='Training loss')
        plt.ylabel('Cost')
        plt.title("Nathan Chess Engine Training Loss Over Time")
        plt.legend()
        plt.xlabel('Training Steps')
        plt.show()

    NathanBot.MaxEpisodesToTrain=7011
    NathanBot.batch_size=64*4

    try:
        for i in range(NathanBot.MaxEpisodesToTrain):
            start = time.time()
            time.sleep(0.25)  # Wait a bit to avoid jamming Stockfish
            print(f"\n=== Episode {i}/{NathanBot.MaxEpisodesToTrain - 1} ===")
            try:
                NathanBot.Train(NathanBot, Elaborate=True, ExtraElaborate=False)
                # Decay exploration rate
                NathanBot.epsilon = max(
                    NathanBot.epsilon_min,
                    NathanBot.epsilon * (1 - NathanBot.epsilon_decay_rate)
                )
                if NathanBot.epsilon < NathanBot.epsilon_min:
                    NathanBot.epsilon = NathanBot.epsilon_min

                # Optional early stop based on loss
                if i > NathanBot.batch_size + 4 and NathanBot.lossBox[-1] < LossDesired:
                    break
                # if i>101:
                #     if np.mean(NathanBot.lossBox[-30:]*4) > np.mean(NathanBot.lossBox[-100:-70]):
                #         print("Loss is increasing weidly high, stop early.")
                #         break

                print(f"Episode time: {time.time() - start:.2f}s")

            except Exception as e:
                print(f"Error in episode {i + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue  # Continue to next episode

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    finally:
        # Clean up Stockfish once at the end
        print("\nCleaning up Stockfish...")
        NathanBot.cleanup_stockfish()
        print("Training complete!")
        elapsed = time.time() - realstart
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        print(f"Total training time: {int(hrs)}h {int(mins)}m {int(secs)}s")

        # Prompt to save model
        while True:
            try:
                botpath = input(
                    f"Where do you want to save the model? ('skip' to skip, 'default' to save in default location, 'new name' to keep default folder but change the name.)\n{defaultspot} "
                ).strip()
                if botpath.lower() == "skip":
                    break
                elif botpath.lower() == "default":
                    NathanBot.save_model(defaultspot)
                    break
                elif botpath.lower() == "new name":
                    filename=input("What do you want your new model to be called?")
                    botpath=os.path.join(changespot,filename+".pth")
                    NathanBot.save_model(botpath)
                    break
                else:
                    NathanBot.save_model(botpath)
                    break
            except Exception as e:
                print(f"Please try again or type 'skip'. Error: {e}")

        PlotMyCost()


if __name__ == "__main__":
    run(LossDesired=.003)
