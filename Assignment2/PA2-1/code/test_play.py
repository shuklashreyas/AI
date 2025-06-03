import pickle
import numpy as np
from vis_gym import setup, game, refresh

gui_flag = True
setup(GUI=gui_flag)
env = game

# (Re‐use the same get_obs/hash functions from Q_learning.py)
def get_obs(reset_output):
    return reset_output[0] if isinstance(reset_output, (tuple, list)) else reset_output

def hash(obs):
    x, y = obs['player_position']
    h = obs['player_health']
    g = obs['guard_in_cell']
    if not g:
        g = 0
    else:
        g = int(g[-1])
    return x * (5*3*5) + y * (3*5) + h * 5 + g

# Load the trained Q-table
with open('Q_table.pickle', 'rb') as f:
    Q_table = pickle.load(f)

raw = env.reset()
obs = get_obs(raw)
done = False
total_reward = 0

while not done:
    s = hash(obs)
    if s not in Q_table:
        action = np.random.randint(env.action_space.n)
    else:
        action = int(np.argmax(Q_table[s]))

    next_raw = env.step(action)
    if len(next_raw) == 4:
        next_obs, reward, done, info = next_raw
    else:
        next_obs = next_raw[0]
        reward = next_raw[1]
        if len(next_raw) == 5:
            terminated, truncated = next_raw[2], next_raw[3]
            done = terminated or truncated
            info = next_raw[4]
        else:
            done = next_raw[2]
            info = next_raw[-1]

    if gui_flag:
        refresh(next_obs, reward, done, info)

    obs = next_obs
    total_reward += reward

print("Test episode finished. Total reward:", total_reward)
env.close()
