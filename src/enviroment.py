#Create a custom gym environment for the tic tac toe game
import gym
from gym import spaces
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns

class SuperTicTacToe(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self):
        super(SuperTicTacToe, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, Box for continuous
        self.action_space = spaces.Box(low=0, high=2, shape=(9, 9), dtype=np.uint8)
        self.observation_space = spaces.Box(low=0, high=2, shape=(9, 9), dtype=np.uint8)
        self.state = np.zeros((9, 9), dtype=np.uint8)
        self.board_to_play = -1
        self.player = 1
        self.previous_action = -1
        self.playable_boards = np.ones(9, dtype=np.uint8)

    def step(self, action):
        # Execute one time step within the environment
        #Action must be a tuple of 2 integers, the first one indicating the position of the board (0 to 80) and the second the value to store (1 to 2)
        #The action is valid if the position is empty
        #The action is invalid if the position is not empty
        #The game is over if a player wins or the board is full
        #The reward is 1 if the player wins, -1 if the player loses, 0 if the game is a draw
        #The info is a dictionary with the state of the game
        position = action
        x, y = position // 9, position % 9

        #check if is valid
        valid,info =  self.check_valid(x,y)
        if not valid:
            return self.state, -10, True, info
        
        #check if the player won the 3x3 grid where the action was made
        semi_win = self.check_semi_win(action)

        if semi_win == 2:
            if self.player == 1:
                semireward = -4
            else:
                semireward = 4
            done = True
        elif semi_win == 1:
            if self.player == 1:
                semireward = 4
            else:
                semireward = -4
        else:
            semireward = 0
        
        self.state[x,y] = self.player
        result = self.check_win()

        if result == 3:
            reward = 5
            done = True
        elif result == 2:
            if self.player == 1: #NEVER ACCESSED
                reward = -5
            else:
                reward = 10
            done = True
        elif result == 1:
            if self.player == 1:
                reward = 10
            else: #NEVER ACCESSED
                reward = -5
            done = True
        else:
            reward = 0
            done = False

        info =  "Correct move" if reward == 0 \
        else "Player 1 wins" if reward == 10 \
        else "Draw" if reward == 5 \
        else "Invalid move"

        self.change_next_board(x,y)

        next_state,_ = self.get_state()

        self.previous_action = action

        self.player = 1 if self.player == 2 else 2

        return next_state, reward+semireward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.board_to_play = -1
        self.state = np.zeros((9, 9), dtype=np.uint8)
        #randomize next player
        self.player = 1 if random.random() > 0.5 else 2
        self.previous_action = -1
        return self.state # reward, done, info can't be included
    
    def get_state(self):

        state_return = self.state.copy()

        #if its player one turn, the state is the same
        if self.player == 1:
            return state_return, self.previous_action
        else:
            #if its player two turn, the state is the same, but the 1s and 2s are swapped
            return np.where(state_return == 1,2,np.where(state_return == 2,1,state_return)), self.previous_action
        

    def render(self, path = 'resources/matches'):

        # Create a copy of the state to avoid modifying the original state
        state_copy = np.copy(self.state)

        # Create a copy of the state to avoid modifying the original state
        state_copy = state_copy.astype('object')

        # Replace player numbers with 'X' and 'O'
        state_copy[state_copy == 1] = 'X'
        state_copy[state_copy == 2] = 'O'
        state_copy[state_copy == 0] = ''

        # Create a figure and axes
        fig, ax = plt.subplots()

        # Create a heatmap from the game state
        # Increase linewidths to 2 for larger boundaries and add lines at sub-board boundaries
        sns.heatmap(self.state, ax=ax, cmap='coolwarm', linecolor='black', linewidths=2, cbar=False, square=True, annot=state_copy, fmt='')

        # Draw lines to separate sub-boards
        for x in range(0, 9, 3):
            ax.axhline(x, color='black', linewidth=3)
            ax.axvline(x, color='black', linewidth=3)

        # Remove x and y labels
        ax.set(xticklabels=[], yticklabels=[], title="Super Tic Tac Toe Board")

        # Save the figure, but not overwrite existing files
        i = 0
        while os.path.exists(f'{path}/board_{i}.png'):
            i += 1
        plt.savefig(f'{path}/board_{i}.png')

        # Close the figure to free up memory
        plt.close(fig)

    def change_next_board(self,x,y):

        # Convert the relative (x, y) position to a single ID (0 to 8)s
        relative_x = x % 3
        relative_y = y % 3
        grid_id = relative_x * 3 + relative_y

        # Calculate the top-left corner of the sub-board
        start_x = (grid_id // 3) * 3
        start_y = (grid_id % 3) * 3
        
        # Extract and return the 3x3 sub-board
        grid = self.state[start_x:start_x+3, start_y:start_y+3]

        #if the grid is full, the next board to play is -1
        if self.check_win_3x3(grid) != 0:
            self.board_to_play = -1
        else:
            self.board_to_play = grid_id

    def check_valid(self, x,y):

        #check if the position is empty
        if self.state[x,y] != 0:
            return False, "Not empty"
        
        #the state is a 9x9 grid, that can be divided into 9 3x3 grids
        # Calculate the top-left corner of the sub-board
        start_x = (x // 3) * 3
        start_y = (y // 3) * 3
        
        # Extract and return the 3x3 sub-board
        grid = self.state[start_x:start_x+3, start_y:start_y+3]

        #from 0 to 8 which grid is this
        row = x // 3
        col = y // 3
        grid_id = row * 3 + col
        
        #if grid number is not the board to play, the move is invalid
        if self.board_to_play != -1 and self.board_to_play != grid_id:
            return False, "Invalid board"
        
        #check if the 3x3 grid its playable
        if self.check_win_3x3(grid) != 0:
            return False, "Closed board"
        return True, ""
    
    def possible_moves(self):
        #return a list with the possible moves for the current state
        moves = []
        for i in range(9):
            for j in range(9):
                if self.state[i,j] == 0:
                    if self.check_valid(i,j)[0]:
                        moves.append((i,j))
        return np.array(moves)

    def check_win_3x3(self, grid):
        #check if a player wins the tic tac toe game in a 3x3 grid
        #the grid is a 3x3 numpy array
        #return 0 if the game is not over
        #return 1 if player 1 wins
        #return 2 if player 2 wins
        #return 3 if the game is a draw

        #check rows
        for i in range(3):
            if grid[i,0] == grid[i,1] == grid[i,2] and grid[i,0] != 0:
                return grid[i,0]
        #check columns
        for i in range(3):
            if grid[0,i] == grid[1,i] == grid[2,i] and grid[0,i] != 0:
                return grid[0,i]
        #check diagonals
        if grid[0,0] == grid[1,1] == grid[2,2] and grid[0,0] != 0:
            return grid[0,0]
        if grid[0,2] == grid[1,1] == grid[2,0] and grid[0,2] != 0:
            return grid[0,2]
        #check if the game is a draw
        if np.all(grid != 0):
            return 3
        return 0
    
    def check_semi_win(self,action):
        #check if the player won the 3x3 grid where the action was made
        x, y = action // 9, action % 9
        grid_x = x//3
        grid_y = y//3
        grid = self.state[grid_x*3:grid_x*3+3,grid_y*3:grid_y*3+3]
        return self.check_win_3x3(grid)

    def check_win(self):
        #check if a player wins the tic tac toe game in the 9x9 grid
        #the grid is a 9x9 numpy array
        #return 0 if the game is not over
        #return 1 if player 1 wins
        #return 2 if player 2 wins
        #return 3 if the game is a draw

        #check the 3x3 grids, and store the result in a 3x3 numpy array
        result = np.zeros((3,3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                result[i,j] = self.check_win_3x3(self.state[i*3:i*3+3,j*3:j*3+3])
        
        #check the new 3x3 array using the same function
        #return the result
        return self.check_win_3x3(result)
    
    def print_board(self):
        #print the 9x9 grid, print the 0s as empty spaces, the 1s as X and the 2s as O. The grid bounds should be visible
        for i in range(9):
            if i % 3 == 0:
                print("-"*13)
            for j in range(9):
                if j % 3 == 0:
                    print("|", end="")
                if self.state[i,j] == 0:
                    print(" ", end="")
                elif self.state[i,j] == 1:
                    print("X", end="")
                else:
                    print("O", end="")
            print("|")
        
    
