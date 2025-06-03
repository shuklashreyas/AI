import time
import pickle
import numpy as np
from vis_gym import *

gui_flag = False  # Set to True to enable the game state visualization
setup(GUI=gui_flag)
env = game  # Gym / Gymnasium environment from vis_gym

def get_obs(reset_output):
    """
    Helper: grab the observation dict from env.reset().
    env.reset() may return:
      - just obs, or
      - (obs, info), or
      - (obs, done, info), etc.
    We always treat the very first element as the observation.
    """
    if isinstance(reset_output, tuple) or isinstance(reset_output, list):
        return reset_output[0]
    else:
        return reset_output

def hash(obs):
    x, y = obs['player_position']
    h = obs['player_health']
    g = obs['guard_in_cell']
    if not g:
        g = 0
    else:
        g = int(g[-1])
    return x * (5*3*5) + y * (3*5) + h * 5 + g

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1.0, decay_rate=0.99999):
    """
    Run Q-learning for num_episodes. Returns a dict: {state_hash: np.array([Q(s,0),…,Q(s,5)])}.
    """
    Q_table = {}
    N_table = {}
    n_actions = env.action_space.n  # should be 6 (UP, DOWN, LEFT, RIGHT, FIGHT, HIDE)

    for episode in range(num_episodes):
        # Correctly unpack just the observation:
        raw = env.reset()
        obs = get_obs(raw)
        done = False

        if episode > 0:
            epsilon *= decay_rate

        while not done:
            state = hash(obs)

            if state not in Q_table:
                Q_table[state] = np.zeros(n_actions, dtype=float)
                N_table[state] = np.zeros(n_actions, dtype=int)

            # ε‐greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = int(np.argmax(Q_table[state]))

            # Step the env
            next_raw = env.step(action)
            # env.step(...) in Gym/Gymnasium always returns at least (obs, reward, done, info)
            # so we can unpack:
            if len(next_raw) == 4:
                next_obs, reward, done, info = next_raw
            else:
                # e.g. if it returns (obs, reward, terminated, truncated, info)
                next_obs = next_raw[0]
                reward = next_raw[1]
                # If there is a “terminated” and “truncated” pair, treat done = either of them
                if len(next_raw) == 5:
                    terminated, truncated = next_raw[2], next_raw[3]
                    done = terminated or truncated
                    info = next_raw[4]
                else:
                    # fallback—grab the usual positions
                    done = next_raw[2]
                    info = next_raw[-1]

            next_state = hash(next_obs)

            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(n_actions, dtype=float)
                N_table[next_state] = np.zeros(n_actions, dtype=int)

            # Learning rate α = 1/(1 + N(s,a))
            alpha = 1.0 / (1 + N_table[state][action])
            best_next = np.max(Q_table[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - Q_table[state][action]

            Q_table[state][action] += alpha * td_error
            N_table[state][action] += 1

            obs = next_obs

    return Q_table

# Choose a decay rate that decays epsilon slowly for many episodes
decay_rate = 0.99999

# Run Q-learning (this time, get_obs handles reset’s extra values)
Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1.0, decay_rate=decay_rate)

with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------- Test-play snippet ----------------
# Save this in a separate file (e.g. test_play.py) or paste below your training code.
#
# import pickle
# import numpy as np
# from vis_gym import setup, game, refresh
#
# gui_flag = True
# setup(GUI=gui_flag)
# env = game
#
# def get_obs(reset_output):
#     if isinstance(reset_output, tuple) or isinstance(reset_output, list):
#         return reset_output[0]
#     else:
#         return reset_output
#
# def hash(obs):
#     x, y = obs['player_position']
#     h = obs['player_health']
#     g = obs['guard_in_cell']
#     if not g:
#         g = 0
#     else:
#         g = int(g[-1])
#     return x * (5*3*5) + y * (3*5) + h * 5 + g
#
# # Load the trained Q-table
# with open('Q_table.pickle', 'rb') as f:
#     Q_table = pickle.load(f)
#
# raw = env.reset()
# obs = get_obs(raw)
# done = False
# total_reward = 0
#
# while not done:
#     s = hash(obs)
#     if s not in Q_table:
#         action = np.random.randint(env.action_space.n)
#     else:
#         action = int(np.argmax(Q_table[s]))
#
#     next_raw = env.step(action)
#     if len(next_raw) == 4:
#         next_obs, reward, done, info = next_raw
#     else:
#         next_obs = next_raw[0]
#         reward = next_raw[1]
#         if len(next_raw) == 5:
#             terminated, truncated = next_raw[2], next_raw[3]
#             done = terminated or truncated
#             info = next_raw[4]
#         else:
#             done = next_raw[2]
#             info = next_raw[-1]
#
#     total_reward += reward
#     if gui_flag:
#         refresh(next_obs, reward, done, info)
#     obs = next_obs
#
# print("Test episode finished. Total reward:", total_reward)
# env.close()
