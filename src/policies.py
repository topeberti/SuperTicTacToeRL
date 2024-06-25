
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Probabilistic:

    def __init__(self,exploration_rate):
        self.exploration_rate = exploration_rate

    def normalize(self,lst):
        min_val = min(lst)
        max_val = max(lst)
        return [(x-min_val) / (max_val-min_val) for x in lst]
    
    def select_action(self,qs,steps,state = [], render = False, show = False):

        #for each q value, compute e to the power of (q * exploration_rate)

        qs_original = qs.copy()

        if show:
            print("Qs: ",np.round(qs_original,6))

        qs = np.array([np.exp(q * self.exploration_rate * steps) for q in qs])

        if show:
            print("Qs after exp: ",qs,6)

        #check if there are infinites in qs
        if np.any(np.isinf(qs)):
            #find the infinites and set them to a high value
            for i in range(len(qs)):
                if np.isinf(qs[i]):
                    qs[i] = 1e6

        #compute the sum of all the qs
        sum_qs = sum(qs)

        # compute the probabilities
        probs = [q/sum_qs for q in qs]

        # check for NaNs in the probabilities
        if np.any(np.isnan(probs)):
            # if there are NaNs, replace them with a uniform probability
            probs = [1/len(qs) for _ in qs]

        if show:
            print("Probs: ",np.round(probs,6))

        #select an action based on the probabilities
        action = np.random.choice(range(len(qs)),p=probs)

        if render:
            self.render_with_probabilities(state,probs,action)

        if show:
            print("Action: ",action)
            print("Prob: ",probs[action])
            print("Q: ",qs_original[action])

        return action

    def render_with_probabilities(self, state, probabilities, action, path=r'D:\berti\Documents\Machine Learning\SuperTicTacToe\SuperTicTacToeRL\resources\heatmaps'):
        # Create a copy of the state to avoid modifying the original state
        state_copy = np.copy(state)

        # Create a copy of the state to avoid modifying the original state
        state_copy = state_copy.astype('object')

        # Replace player numbers with 'X' and 'O'
        state_copy[state_copy == 1] = 'X'
        state_copy[state_copy == 2] = 'O'
        state_copy[state_copy == 0] = ''

        # Reshape probabilities to match the shape of the game state
        probabilities = np.array(probabilities).reshape((9,9))
        state_copy = state_copy.reshape((9,9))

        # Create a figure and axes
        fig, ax = plt.subplots()

        # Create a heatmap from the game state
        # Use probabilities for color mapping
        sns.heatmap(probabilities, ax=ax, cmap='coolwarm', linecolor='black', linewidths=2, cbar=True, square=True, annot=state_copy, fmt='', vmin=0, vmax=1)

        # Draw lines to separate sub-boards
        for x in range(0, 9, 3):
            ax.axhline(x, color='black', linewidth=3)
            ax.axvline(x, color='black', linewidth=3)

        # Highlight the action taken
        action_row, action_col = divmod(action, 9)
        rect = plt.Rectangle((action_col, action_row), 1, 1, fill=False, color='yellow', linewidth=3)
        ax.add_patch(rect)

        # Show the probability of the action taken
        action_probability = probabilities[action_row, action_col]
        plt.text(0.5, -0.1, f'Probability of action taken: {action_probability:.2f}', size=12, ha="center", transform=ax.transAxes)

        # Remove x and y labels
        ax.set(xticklabels=[], yticklabels=[], title="Super Tic Tac Toe Board")

        # Save the figure, but not overwrite existing files
        i = 0
        while os.path.exists(f'{path}/game_state_{i}.png'):
            i += 1
        # Save the figure
        fig.savefig(f"{path}/game_state_{i}.png")

        # Close the figure to free up memory
        plt.close(fig)