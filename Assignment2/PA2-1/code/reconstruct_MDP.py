import time
import pickle
import numpy as np
from vis_gym import *

gui_flag = False  # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game  # Gym environment already initialized within vis_gym.py

# env.render()  # Uncomment to print game state info

def hash(obs):
    x, y = obs['player_position']
    h = obs['player_health']
    g = obs['guard_in_cell']
    if not g:
        g = 0
    else:
        g = int(g[-1])
    return x * (5*3*5) + y * (3*5) + h * 5 + g

'''
Complete the function below to do the following:

    1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
       configuration and taking actions until a terminal state is reached.
    2. Instead of saving all gameplay history, maintain and update Q-values for each state-action pair that your agent encounters in a dictionary.
    3. Use the Q-values to select actions in an epsilon-greedy manner. Refer to assignment instructions for a refresher on this.
    4. Update the Q-values using the Q-learning update rule. Refer to assignment instructions for a refresher on this.

    Some important notes:
        
        - The state space is defined by the player's position (x,y), the player's health (h), and the guard in the cell (g).
        
        - To simplify the representation of the state space, each state may be hashed into a unique integer value using the hash function provided above.
          For instance, the observation {'player_position': (1, 2), 'player_health': 2, 'guard_in_cell='G4'} 
          will be hashed to 1*5*3*5 + 2*3*5 + 2*5 + 4 = 119. There are 375 unique states.

        - Your Q-table should be a dictionary with the following format:

                - Each key is a number representing the state (hashed using the provided hash() function), and each value should be an np.array
                  of length equal to the number of actions (initialized to all zeros).

                - This will allow you to look up Q(s,a) as Q_table[state][action], as well as directly use efficient numpy operators
                  when considering all actions from a given state, such as np.argmax(Q_table[state]) within your Bellman equation updates.

                - The autograder also assumes this format, so please ensure you format your code accordingly.
  
          Please do not change this representation of the Q-table.
        
        - The four actions are: 0 (UP), 1 (DOWN), 2 (LEFT), 3 (RIGHT), 4 (FIGHT), 5 (HIDE)

        - Don't forget to reset the environment to the initial configuration after each episode by calling:
          obs, reward, done, info = env.reset()

        - The value of eta is unique for every (s,a) pair, and should be updated as 1/(1 + number of updates to Q_opt(s,a)).

        - The value of epsilon is initialized to 1. You are free to choose the decay rate.
          No default value is specified for the decay rate, experiment with different values to find what works.

        - To refresh the game screen if using the GUI, use the refresh(obs, reward, done, info) function, with the 'if gui_flag:' condition.
          Example usage below. This function should be called after every action.
          if gui_flag:
              refresh(obs, reward, done, info)  # Update the game screen [GUI only]

    Finally, return the dictionary containing the Q-values (called Q_table).

'''

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1.0, decay_rate=0.99999):
    """
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon is decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    # Initialize Q-table and N-table (state-action visit counts)
    Q_table = {}      # Q_table[state] = np.array([0, 0, 0, 0, 0, 0])
    N_table = {}      # N_table[state] = np.array([0, 0, 0, 0, 0, 0]) to track updates per (s,a)

    n_actions = env.action_space.n  # should be 6: [UP, DOWN, LEFT, RIGHT, FIGHT, HIDE]

    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        # Decay epsilon at the start of each episode except the first
        if episode > 0:
            epsilon *= decay_rate

        while not done:
            state = hash(obs)

            # If state not in Q_table yet, initialize Q_table[state] and N_table[state]
            if state not in Q_table:
                Q_table[state] = np.zeros(n_actions, dtype=float)
                N_table[state] = np.zeros(n_actions, dtype=int)

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                # Choose action with highest Q-value (break ties arbitrarily via argmax)
                action = int(np.argmax(Q_table[state]))

            # Take action in the environment
            next_obs, reward, done, info = env.step(action)

            next_state = hash(next_obs)

            # If next_state is unseen, initialize its Q and N entries
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(n_actions, dtype=float)
                N_table[next_state] = np.zeros(n_actions, dtype=int)

            # Compute learning rate alpha = 1 / (1 + number of times we've updated Q[state][action])
            alpha = 1.0 / (1 + N_table[state][action])

            # Q-learning update:
            # Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a' Q(next_s, a') - Q(s,a)]
            best_next = np.max(Q_table[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - Q_table[state][action]
            Q_table[state][action] += alpha * td_error

            # Increment the visit count for this (state, action)
            N_table[state][action] += 1

            # Move to next state
            obs = next_obs

    return Q_table

# Choose a decay rate (tuned for 1,000,000 episodes so epsilon decays slowly)
decay_rate = 0.99999

# Run Q-learning for 1,000,000 episodes
Q_table = Q_learning(num_episodes=1000, gamma=0.9, epsilon=1.0, decay_rate=decay_rate)

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.

Comment before final submission or autograder may fail.
'''

# # To load the Q-table:
# Q_table = pickle.load(open('Q_table.pickle', 'rb'))

# obs, reward, done, info = env.reset()
# total_reward = 0
# while not done:
#     state = hash(obs)
#     action = int(np.argmax(Q_table[state]))
#     obs, reward, done, info = env.step(action)
#     total_reward += reward
#     if gui_flag:
#         refresh(obs, reward, done, info)  # Update the game screen [GUI only]

# print("Total reward:", total_reward)

# env.close()  # Close the environment
