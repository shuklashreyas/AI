import numpy as np
from vis_gym import setup, game

# Turn off GUI for speed
gui_flag = False
setup(GUI=gui_flag)
env = game  # Gym/Gymnasium environment


def get_obs(reset_output):
    """
    Extract just the observation dict from env.reset().
    If env.reset() returns a tuple/list, grab the first element.
    """
    return reset_output[0] if isinstance(reset_output, (tuple, list)) else reset_output


def parse_step(step_output):
    """
    Normalize env.step(...) output into (next_obs, reward, done, info).
    Gym/Gymnasium can return either 4 items (obs, reward, done, info)
    or 5 items (obs, reward, terminated, truncated, info). We unify to 4.
    """
    if len(step_output) == 4:
        return step_output  # (obs, reward, done, info)
    next_obs, reward, terminated, truncated, info = step_output
    done = terminated or truncated
    return next_obs, reward, done, info


def hash(obs):
    """
    Turn an observation into a unique integer in [0..374].
    - obs['player_position'] = (x,y), x,y ∈ {0..4}
    - obs['player_health'] ∈ {0,1,2}
    - obs['guard_in_cell'] is None or 'G#'
    Map g = 0 if None, else g = int(#). Then:
      hash = x*(5*3*5) + y*(3*5) + h*5 + g
    """
    x, y = obs['player_position']
    h = obs['player_health']
    g = obs['guard_in_cell']
    if not g:
        g = 0
    else:
        g = int(g[-1])
    return x * (5 * 3 * 5) + y * (3 * 5) + h * 5 + g


def estimate_victory_probability(num_episodes=1000000):
    """
    Estimate P(defeat G_i | fight from full health) for i=1..4 under a random policy.
    Only count fights when player_health == 2 before initiating the fight.
    Returns a length-4 array P, where P[i] is the win-rate vs G{i+1}.
    """
    n_guards = len(env.guards)  # should be 4
    wins = np.zeros(n_guards, dtype=float)
    visits = np.zeros(n_guards, dtype=float)

    # Correct index for the “fight” action in vis_gym
    FIGHT_ACTION = 4

    for _ in range(num_episodes):
        raw = env.reset()
        obs = get_obs(raw)
        done = False

        while not done:
            guard_str = obs['guard_in_cell']      # None or 'G#'
            health_before = obs['player_health']  # 0, 1, or 2

            action = np.random.randint(env.action_space.n)
            step_out = env.step(action)
            next_obs, reward, done, info = parse_step(step_out)

            # Only record if we were at full health (h == 2) and chose “fight”
            if guard_str is not None and action == FIGHT_ACTION and health_before == 2:
                guard_id = int(guard_str[-1])  # e.g. 'G3' → 3
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


if __name__ == "__main__":
    NUM_EPISODES = 1_000_000
    P_est = estimate_victory_probability(num_episodes=NUM_EPISODES)
    print(f"Estimated win‐rates over {NUM_EPISODES} episodes: {P_est}")
