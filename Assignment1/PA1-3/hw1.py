import math
import random
import time
import numpy as np
from gridgame import *

# --- Game Initialization ---
# For final submission, set GUI to False and remove render_delay_sec or set to 0
# game = ShapePlacementGrid(GUI=False, gs=6, num_colored_boxes=5)
# For testing:
game = ShapePlacementGrid(
    GUI=True, render_delay_sec=0.05, gs=6, num_colored_boxes=5)


# Initial export
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute(
    'export')
np.savetxt('initial_grid.txt', grid, fmt="%d")

grid_size = grid.shape[0]
total_grid_cells = grid_size * \
    grid_size if grid_size > 0 else 1  # Avoid division by zero
shape_areas = [int(s.sum()) for s in game.shapes]


# --- Timing Start ---
start_time = time.time()

# --- Helper Functions ---


def count_uncolored_cells(grid_state):
    """Counts the number of uncolored (-1) cells in the grid."""
    return np.sum(grid_state == -1)


def get_percent_uncolored(grid_state):
    """Calculates the percentage of uncolored cells."""
    uncolored = count_uncolored_cells(grid_state)
    return uncolored / total_grid_cells


def evaluate_cost(grid_state, current_placed_shapes_list):
    """
    Evaluates the cost of a given grid state.
    Lower cost is better.
    Cost components:
    1. Conflicts (adjacent same colors).
    2. Uncolored cells (incomplete grid).
    3. Number of shapes used.
    """
    n = grid_state.shape[0]
    conflicts = 0
    uncolored_cell_penalty = 0

    for r in range(n):
        for c in range(n):
            color_val = grid_state[r, c]
            if color_val == -1:
                uncolored_cell_penalty += 1  # Each uncolored cell adds to penalty
                continue
            # Check right neighbor
            if c + 1 < n and grid_state[r, c + 1] == color_val:
                conflicts += 1
            # Check bottom neighbor
            if r + 1 < n and grid_state[r + 1, c] == color_val:
                conflicts += 1

    # Heavily penalize incompleteness and conflicts.
    # The multiplier for conflicts and uncolored_cell_penalty should be high enough
    # to ensure they are prioritized over minimizing shapes.
    # (n * n * 10) was used, let's ensure it's dominant. Max shapes could be n*n.
    # Max conflicts could be ~2*n*n.
    cost = (conflicts * n * n * 100) + (uncolored_cell_penalty *
                                        n * n * 50) + len(current_placed_shapes_list)
    return cost


def goto_target(current_brush_pos, target_brush_pos):
    """
    Moves the brush from current_brush_pos to target_brush_pos by executing
    game commands. Returns the new brush position.
    Does NOT export the full game state.
    """
    x0, y0 = current_brush_pos
    x1, y1 = target_brush_pos

    # Horizontal moves
    while x1 > x0:
        game.execute('right')
        x0 += 1
    while x1 < x0:
        game.execute('left')
        x0 -= 1
    # Vertical moves
    while y1 > y0:
        game.execute('down')
        y0 += 1
    while y1 < y0:
        game.execute('up')
        y0 -= 1
    return [x1, y1]


def get_allowed_colors_for_placement(grid_state_to_check, shape_matrix, pos_xy_top_left, num_total_colors):
    """
    Identifies and returns a list of color indices that can be legally used
    to place the given shape_matrix at pos_xy_top_left on the grid_state_to_check
    without creating immediate color conflicts with existing cells in grid_state_to_check.
    Assumes the placement itself is valid in terms of grid boundaries and empty cells (checked by game.canPlace).
    """
    allowed_color_indices = []
    n_grid = grid_state_to_check.shape[0]
    shape_h, shape_w = shape_matrix.shape

    for color_idx_to_try in range(num_total_colors):
        is_this_color_ok = True
        for r_offset in range(shape_h):
            for c_offset in range(shape_w):
                if shape_matrix[r_offset, c_offset]:  # If this part of the shape is solid
                    gr, gc = pos_xy_top_left[1] + \
                        r_offset, pos_xy_top_left[0] + c_offset

                    # Check for conflicts with neighbors in grid_state_to_check
                    # Directions: up, down, left, right
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = gr + dr, gc + dc
                        if 0 <= nr < n_grid and 0 <= nc < n_grid and grid_state_to_check[nr, nc] == color_idx_to_try:
                            is_this_color_ok = False
                            break
                    if not is_this_color_ok:
                        break
            if not is_this_color_ok:
                break

        if is_this_color_ok:
            allowed_color_indices.append(color_idx_to_try)

    return allowed_color_indices


# --- Greedy Initial Solution Attempt ---
print("Starting greedy seed generation...")
# Get initial game state
gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done = game.execute(
    'export')

ordered_shape_indices = sorted(
    range(len(game.shapes)), key=lambda i: shape_areas[i], reverse=True)

for shape_idx_to_place in ordered_shape_indices:
    current_shape_obj = game.shapes[shape_idx_to_place]
    s_h, s_w = current_shape_obj.shape

    # Iterate over all possible top-left positions for the current shape
    for r in range(grid_size - s_h + 1):
        for c in range(grid_size - s_w + 1):
            # Refresh current game state before each potential placement attempt
            gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done = game.execute(
                'export')

            # 1. Switch to the target shape
            while gs_currentShapeIndex != shape_idx_to_place:
                game.execute('switchshape')
                gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done = game.execute(
                    'export')

            # 2. Move brush to target position
            gs_shapePos = goto_target(gs_shapePos, [c, r])
            # Export state after moving to ensure game's internal brush position is correct
            gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done = game.execute(
                'export')

            # 3. Check if placement is possible (empty cells, within bounds)
            if not game.canPlace(gs_grid, current_shape_obj, [c, r]):
                continue

            # 4. Find a valid color
            # get_allowed_colors_for_placement uses the exported gs_grid
            valid_colors = get_allowed_colors_for_placement(
                gs_grid, current_shape_obj, [c, r], len(game.colors))
            if not valid_colors:
                continue

            # Or choose the first one, or based on a heuristic
            chosen_color_idx = random.choice(valid_colors)

            # 5. Switch to the chosen color
            while gs_currentColorIndex != chosen_color_idx:
                game.execute('switchcolor')
                gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done = game.execute(
                    'export')

            # 6. Place the shape
            game.execute('place')
            # gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done are updated by the next export at loop start or end of greedy

# Greedy phase finished. Record this solution.
gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done = game.execute(
    'export')

# These are our local copies representing the current state of the solution
current_grid_config = gs_grid.copy()
current_shapes_list = list(gs_placedShapes)  # List of tuples from game
current_solution_cost = evaluate_cost(current_grid_config, current_shapes_list)

best_overall_cost = current_solution_cost
best_shapes_list = list(current_shapes_list)
best_grid_config = current_grid_config.copy()

print(
    f"Greedy seed cost: {current_solution_cost}, Shapes used: {len(current_shapes_list)}")

# --- Simulated Annealing ---
print("Starting simulated annealing...")
T = 100.0  # Initial temperature - might need tuning
T_min = 0.01  # Minimum temperature to stop
alpha = 0.995  # Cooling rate (e.g., 0.99 or 0.995)
max_time_seconds = 580  # Time limit for annealing phase

# Use game state variables from the end of greedy or last accepted SA move
# current_grid_config, current_shapes_list, current_solution_cost are our primary state trackers

sa_iteration = 0
while T > T_min and (time.time() - start_time) < max_time_seconds:
    sa_iteration += 1

    # If grid is empty (e.g., greedy failed or very small grid)
    if not current_shapes_list:
        # Try to place a small random shape. This is a basic recovery/initialization.
        # This block is less likely to be hit if greedy places at least one shape.
        pass  # For now, assume greedy places something or this needs a robust "place random"

    # Perturbation: Undo one random shape, then try to place one new (potentially different) shape.
    if not current_shapes_list:  # Cannot undo if no shapes are placed
        T *= alpha  # Cool down and skip if nothing to perturb
        continue

    # --- Create a Neighbor Solution ---
    # 1. Store the current accepted state (in case we need to revert fully)
    # We already have current_grid_config and current_shapes_list.

    # 2. Decide which shape to undo (from current_shapes_list, then reflect in game)
    # For simplicity, always undo the *last* shape placed in current_shapes_list
    # More advanced: undo a random shape from current_shapes_list. This requires replaying up to that point.
    # For now, stick to undoing the most recent operation the game knows about.
    # This means our 'current_shapes_list' must be perfectly in sync with game's internal placedShapes.

    # To ensure sync: Rebuild the game state to 'current_shapes_list' before each SA step's perturbation.
    # This is very safe but can be slow.
    # Alternative: Trust game.execute('undo') and manage our lists.

    # Let's try to manage lists and game state carefully.
    # The game's "last shape" is the one to be undone by game.execute('undo')
    # Our current_shapes_list should match the game's history.

    # --- Perturb by undoing and then trying a new placement ---
    game.execute('undo')  # Undoes the game's last placed shape

    # Get game state AFTER the undo
    gsa_shapePos, gsa_cSI, gsa_cCI, gsa_grid_after_undo, gsa_pS_after_undo, _ = game.execute(
        'export')

    # Generate a new move (shape, position, color)
    percent_unfilled_after_undo = get_percent_uncolored(gsa_grid_after_undo)

    # Dynamic brush selection strategy
    if percent_unfilled_after_undo > 0.7:  # Lots of space
        # Add small base to avoid zero probability
        weights = [(area / sum(shape_areas)) + 0.01 for area in shape_areas]
    elif percent_unfilled_after_undo < 0.15 and percent_unfilled_after_undo > 0:  # Filling small gaps
        weights = [(1.0 / (area + 1e-6)) + 0.01 for area in shape_areas]
    else:  # Balanced approach
        weights = [1.0 for _ in shape_areas]  # Uniform

    # Normalize weights
    sum_weights = sum(weights)
    normalized_weights = [w / sum_weights for w in weights]

    try:
        new_shape_idx_to_try = random.choices(
            range(len(game.shapes)), weights=normalized_weights, k=1)[0]
    except ValueError:  # In case all weights somehow become zero
        new_shape_idx_to_try = random.randrange(len(game.shapes))

    # Switch to the new shape
    temp_cSI = gsa_cSI
    while temp_cSI != new_shape_idx_to_try:
        game.execute('switchshape')
        _, temp_cSI, _, _, _, _ = game.execute(
            'export')  # Only need updated cSI

    new_shape_obj = game.shapes[new_shape_idx_to_try]
    s_h, s_w = new_shape_obj.shape

    # Determine valid random position (ensure shape fits within grid dimensions)
    max_rand_x = grid_size - s_w
    max_rand_y = grid_size - s_h
    # Shape is too big for the grid (should not happen with default shapes/grid)
    if max_rand_x < 0 or max_rand_y < 0:
        T *= alpha
        continue

    rand_pos_x = random.randrange(max_rand_x + 1)
    rand_pos_y = random.randrange(max_rand_y + 1)

    temp_shapePos = gsa_shapePos
    temp_shapePos = goto_target(temp_shapePos, [rand_pos_x, rand_pos_y])

    # Get game state after shape switch and brush move, before choosing color
    gsb_sP, gsb_cSI, gsb_cCI, gsb_grid_before_color, _, _ = game.execute(
        'export')

    # Check if placement is possible at new position on the grid_after_undo
    if not game.canPlace(gsb_grid_before_color, new_shape_obj, [rand_pos_x, rand_pos_y]):
        # If cannot place, this perturbation is invalid. The game is currently in the "undone" state.
        # The cost of this "undone" state will be evaluated.
        # To simplify, we treat "cannot place" as a "no-op" for the 'place' part of perturbation.
        # The 'new_candidate_grid' will effectively be 'gsa_grid_after_undo'.
        new_candidate_grid = gsa_grid_after_undo.copy()
        new_candidate_shapes_list = list(gsa_pS_after_undo)
    else:
        # Find a valid color
        allowed_colors = get_allowed_colors_for_placement(
            gsb_grid_before_color, new_shape_obj, [rand_pos_x, rand_pos_y], len(game.colors))

        if not allowed_colors:
            # No valid color, similar to cannot place. Evaluate the "undone" state.
            new_candidate_grid = gsa_grid_after_undo.copy()
            new_candidate_shapes_list = list(gsa_pS_after_undo)
        else:
            new_color_idx = random.choice(allowed_colors)
            temp_cCI = gsb_cCI
            while temp_cCI != new_color_idx:
                game.execute('switchcolor')
                _, _, temp_cCI, _, _, _ = game.execute('export')  # Update cCI

            game.execute('place')  # Place the new shape
            # Get the state AFTER this placement attempt
            _, _, _, new_candidate_grid_after_place, new_candidate_shapes_list_after_place, _ = game.execute(
                'export')
            new_candidate_grid = new_candidate_grid_after_place.copy()
            new_candidate_shapes_list = list(
                new_candidate_shapes_list_after_place)

    # --- Evaluate this new candidate state ---
    cost_new_candidate = evaluate_cost(
        new_candidate_grid, new_candidate_shapes_list)

    delta_cost = cost_new_candidate - current_solution_cost

    if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
        # Accept the new state
        current_grid_config = new_candidate_grid.copy()
        current_shapes_list = list(
            new_candidate_shapes_list)  # Critical update
        current_solution_cost = cost_new_candidate
        # The game environment is already in this accepted state.

        if current_solution_cost < best_overall_cost:
            best_overall_cost = current_solution_cost
            best_shapes_list = list(current_shapes_list)
            best_grid_config = current_grid_config.copy()
            if sa_iteration % 10 == 0:  # Print progress periodically
                print(
                    f"SA New Best: Cost={best_overall_cost:.0f}, Shapes={len(best_shapes_list)}, T={T:.3f}, Unfilled%={get_percent_uncolored(best_grid_config)*100:.1f}")
    else:
        # Reject the new state. Revert the game environment to the previous accepted state.
        # The game environment is currently in 'new_candidate_state'.
        # We need to restore it to the state represented by 'current_grid_config' / 'current_shapes_list' (before this iteration's perturbation).

        # To do this, clear the game board and replay 'current_shapes_list' (which was the state before this rejected move)
        temp_sP, temp_cSI, temp_cCI, temp_g, temp_pS, _ = game.execute(
            'export')
        while temp_pS:  # Undo all shapes currently on the game board
            game.execute('undo')
            temp_sP, temp_cSI, temp_cCI, temp_g, temp_pS, _ = game.execute(
                'export')

        # Replay the shapes from the last accepted state ('current_shapes_list')
        for shp_idx, pos_tuple, col_idx in current_shapes_list:
            pos_list = list(pos_tuple)
            # Switch shape
            while temp_cSI != shp_idx:
                game.execute('switchshape')
                temp_sP, temp_cSI, temp_cCI, _, _, _ = game.execute('export')
            # Switch color
            while temp_cCI != col_idx:
                game.execute('switchcolor')
                temp_sP, _, temp_cCI, _, _, _ = game.execute('export')
            # Move
            temp_sP = goto_target(temp_sP, pos_list)
            # Export after move, before place
            temp_sP, temp_cSI, temp_cCI, temp_g_before_place, _, _ = game.execute(
                'export')
            # Place (assuming it was valid since it's from an accepted solution)
            # Should be true
            if game.canPlace(temp_g_before_place, game.shapes[temp_cSI], pos_list):
                game.execute('place')
            else:
                # This case should ideally not happen if current_shapes_list is always valid.
                # If it does, it implies a bug in state synchronization or logic.
                print(
                    "Error: Failed to replay an accepted shape during SA rejection. State might be inconsistent.")
                break
            temp_sP, temp_cSI, temp_cCI, _, temp_pS, _ = game.execute(
                'export')  # Update state after place

    T *= alpha  # Cool down

print(
    f"Simulated annealing finished. Final best cost found: {best_overall_cost}")

# --- Replay Best Solution to Set Final Game State ---
print("Replaying the overall best solution found...")
# 1. Clear the current game board
sP, cSI, cCI, g, pS, d = game.execute('export')
while pS:  # Undo all shapes
    game.execute('undo')
    sP, cSI, cCI, g, pS, d = game.execute('export')

# 2. Apply the shapes from best_shapes_list
current_replay_sP = sP
current_replay_cSI = cSI
current_replay_cCI = cCI

for shp_idx, pos_tuple, col_idx in best_shapes_list:
    pos_list = list(pos_tuple)

    while current_replay_cSI != shp_idx:
        game.execute('switchshape')
        _, current_replay_cSI, _, _, _, _ = game.execute('export')

    while current_replay_cCI != col_idx:
        game.execute('switchcolor')
        _, _, current_replay_cCI, _, _, _ = game.execute('export')

    current_replay_sP = goto_target(current_replay_sP, pos_list)

    # Export state after moving, before placing, to ensure game uses correct internal brush pos
    # and to get the grid for canPlace
    final_sP_before_place, final_cSI_before_place, final_cCI_before_place, final_g_before_place, _, _ = game.execute(
        'export')

    # It's from best_shapes_list, so it should be placeable.
    # However, game.canPlace provides a safeguard.
    if game.canPlace(final_g_before_place, game.shapes[final_cSI_before_place], pos_list):
        game.execute('place')
        current_replay_sP, current_replay_cSI, current_replay_cCI, _, _, _ = game.execute(
            'export')  # Update state vars after place
    else:
        # This would indicate an issue if a shape from a "best solution" cannot be replayed.
        print(
            f"Error during final replay: Cannot place shape {shp_idx} at {pos_list}. Skipping.")


# Get the definitive final state from the game
final_sP, final_cSI, final_cCI, final_grid_output, final_placedShapes_output, final_done_output = game.execute(
    'export')

# --- Timing End & Final Output ---
end_time = time.time()
total_duration = end_time - start_time

print(f"\n--- Final Results ---")
print(f"Total execution time: {total_duration:.2f} seconds")
final_cost_check = evaluate_cost(final_grid_output, final_placedShapes_output)
print(f"Cost of final grid: {final_cost_check}")
print(f"Number of shapes in final solution: {len(final_placedShapes_output)}")
# Use game.checkGrid for the official validity check as per assignment instructions
is_solution_valid_by_game = game.checkGrid(final_grid_output)
print(f"Is solution valid (by game.checkGrid): {is_solution_valid_by_game}")
if not is_solution_valid_by_game:
    print(
        f"Debug: Conflicts or Unfilled. Unfilled cells: {count_uncolored_cells(final_grid_output)}")


# --- Write Output Files (as per original instructions) ---
np.savetxt('grid.txt', final_grid_output, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(final_placedShapes_output))
with open("time.txt", "w") as outfile:
    outfile.write(str(total_duration))

print("\nOutput files (grid.txt, shapes.txt, time.txt) written.")

if game.GUI:
    print("GUI window active. Close window to exit program.")
    pygame_running = True
    while pygame_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame_running = False
        try:
            pygame.display.flip()  # Keep display updated
            game.clock.tick(30)  # Control Pygame loop speed
        except pygame.error:  # Handle cases where display might be closed prematurely
            pygame_running = False
    pygame.quit()
