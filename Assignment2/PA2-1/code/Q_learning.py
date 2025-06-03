import pickle
import numpy as np
from vis_gym import setup, game, refresh

gui_flag = False
setup(GUI=gui_flag)
env = game

def get_obs(reset_output):
    return reset_output[0] if isinstance(reset_output, (tuple, list)) else reset_output

def parse_step(step_output):
    if len(step_output) == 4:
        return step_output
    next_obs, reward, terminated, truncated, info = step_output
    done = terminated or truncated
    return next_obs, reward, done, info

def hash(obs):
    x, y = obs['player_position']
    h = obs['player_health']
    g = obs['guard_in_cell']
    if not g:
        g = 0
    else:
        g = int(g[-1])
    return x * (5 * 3 * 5) + y * (3 * 5) + h * 5 + g

def estimate_victory_probability(num_episodes=100000):
    n_guards = len(env.guards)  # should be 4
    wins = np.zeros(n_guards, dtype=float)
    visits = np.zeros(n_guards, dtype=float)
    FIGHT_ACTION = 5

    for _ in range(num_episodes):
        raw = env.reset()
        obs = get_obs(raw)
        done = False

        while not done:
            guard_str = obs['guard_in_cell']
            health_before = obs['player_health']
            action = np.random.randint(env.action_space.n)
            step_out = env.step(action)
            next_obs, reward, done, info = parse_step(step_out)

            if guard_str is not None and action == FIGHT_ACTION:
                guard_id = int(guard_str[-1])
                idx = guard_id - 1
                health_after = next_obs['player_health']
                if health_after == health_before:
                    wins[idx] += 1
                visits[idx] += 1

            obs = next_obs

    P = np.zeros(n_guards, dtype=float)
    nonzero = visits > 0
    P[nonzero] = wins[nonzero] / visits[nonzero]
    return P

def Q_learning(num_episodes, gamma=0.9, epsilon=1.0, decay_rate=0.99999):
    Q_table = {}
    N_table = {}
    n_actions = env.action_space.n  # should be 6

    for episode in range(num_episodes):
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

            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = int(np.argmax(Q_table[state]))

            step_out = env.step(action)
            next_obs, reward, done, info = parse_step(step_out)

            next_state = hash(next_obs)
            if next_state not in Q_table:
                Q_table[next_state] = np.zeros(n_actions, dtype=float)
                N_table[next_state] = np.zeros(n_actions, dtype=int)

            alpha = 1.0 / (1 + N_table[state][action])
            best_next = np.max(Q_table[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - Q_table[state][action]
            Q_table[state][action] += alpha * td_error
            N_table[state][action] += 1

            obs = next_obs

    return Q_table

if __name__ == "__main__":
    # Only run training when invoked directly — the grader will import functions instead.
    NUM_EPISODES = 1_000_000
    GAMMA = 0.9
    EPSILON_START = 1.0
    DECAY_RATE = 0.99999

    print("Training Q-learning (this runs ~1 000 000 episodes)...")
    Q_table = Q_learning(NUM_EPISODES, gamma=GAMMA, epsilon=EPSILON_START, decay_rate=DECAY_RATE)

    with open("Q_table.pickle", "wb") as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Q_table.pickle saved.")
