from reconstruct_MDP import estimate_victory_probability

P = estimate_victory_probability(num_episodes=50_000)
print("Empirical win‐rates vs G1–G4:", P)
