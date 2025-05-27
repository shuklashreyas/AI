import math
import random
import time
import numpy as np
from gridgame import *

# initialize grid game (False disables GUI)
game_gui_enabled = False
game = ShapePlacementGrid(GUI=game_gui_enabled, render_delay_sec=0.05, gs=6, num_colored_boxes=5)

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute("export")
np.savetxt("initial_grid.txt", grid, fmt="%d")

grid_size = grid.shape[0]
total_grid_cells = grid_size * grid_size if grid_size else 1
shape_areas = [int(shape.sum()) for shape in game.shapes]

start_time = time.time()

def count_uncolored_cells(grid_state):
    return np.sum(grid_state == -1)

def get_percent_uncolored(grid_state):
    return count_uncolored_cells(grid_state) / total_grid_cells

# composite cost: conflicts, unfilled cells, shape count
def evaluate_cost(grid_state, shapes_list):
    n = grid_state.shape[0]
    conflicts = 0
    unfilled = 0
    for r in range(n):
        for c in range(n):
            val = grid_state[r, c]
            if val == -1:
                unfilled += 1
            else:
                if c + 1 < n and grid_state[r, c+1] == val:
                    conflicts += 1
                if r + 1 < n and grid_state[r+1, c] == val:
                    conflicts += 1
    return conflicts*n*n*100 + unfilled*n*n*50 + len(shapes_list)

# move brush to target position
def goto_target(cur_pos, tgt_pos):
    x0, y0 = cur_pos
    x1, y1 = tgt_pos
    while x1 > x0:
        game.execute("right"); x0 += 1
    while x1 < x0:
        game.execute("left"); x0 -= 1
    while y1 > y0:
        game.execute("down"); y0 += 1
    while y1 < y0:
        game.execute("up"); y0 -= 1
    return [x1, y1]

# check safe colors for placement
def get_allowed_colors_for_placement(grid_state, shape_mat, pos, num_colors):
    n = grid_state.shape[0]
    h, w = shape_mat.shape
    allowed = []
    for color in range(num_colors):
        ok = True
        for dy in range(h):
            for dx in range(w):
                if not shape_mat[dy, dx]: continue
                r, c = pos[1]+dy, pos[0]+dx
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<n and 0<=nc<n and grid_state[nr,nc]==color:
                        ok = False; break
                if not ok: break
            if not ok: break
        if ok: allowed.append(color)
    return allowed

print("Starting greedy seed generation...")
(
    gs_shapePos,
    gs_currentShapeIndex,
    gs_currentColorIndex,
    gs_grid,
    gs_placedShapes,
    gs_done,
) = game.execute("export")

order = sorted(range(len(game.shapes)), key=lambda i: shape_areas[i], reverse=True)
for idx in order:
    shape = game.shapes[idx]
    h, w = shape.shape
    for r in range(grid_size - h + 1):
        for c in range(grid_size - w + 1):
            gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done = \
                game.execute("export")
            while gs_currentShapeIndex != idx:
                game.execute("switchshape")
                gs_shapePos, gs_currentShapeIndex, *_ = game.execute("export")
            gs_shapePos = goto_target(gs_shapePos, [c, r])
            gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done = \
                game.execute("export")
            if not game.canPlace(gs_grid, shape, [c, r]): continue
            valid = get_allowed_colors_for_placement(gs_grid, shape, [c, r], len(game.colors))
            if not valid: continue
            chosen = random.choice(valid)  # AI-assisted heuristic
            while gs_currentColorIndex != chosen:
                game.execute("switchcolor")
                gs_shapePos, gs_currentShapeIndex, gs_currentColorIndex, gs_grid, gs_placedShapes, gs_done = \
                    game.execute("export")
            game.execute("place")

(
    gs_shapePos,
    gs_currentShapeIndex,
    gs_currentColorIndex,
    gs_grid,
    gs_placedShapes,
    gs_done,
) = game.execute("export")
current_grid = gs_grid.copy()
current_shapes = list(gs_placedShapes)
current_cost = evaluate_cost(current_grid, current_shapes)
best_cost = current_cost
best_shapes = list(current_shapes)
best_grid = current_grid.copy()
print(f"Greedy cost: {current_cost}, shapes: {len(current_shapes)}")

print("Starting simulated annealing...")
T, T_min, alpha = 100.0, 0.01, 0.995
deadline = start_time + 580
iteration = 0
while T > T_min and time.time() < deadline:
    iteration += 1
    if not current_shapes:
        T *= alpha; continue
    game.execute("undo")
    gsa_shapePos, gsa_cSI, gsa_cCI, gsa_grid, gsa_shapes, _ = game.execute("export")
    p = get_percent_uncolored(gsa_grid)
    if p > 0.7:
        weights = [(a/sum(shape_areas))+0.01 for a in shape_areas]
    elif p < 0.15:
        weights = [(1/(a+1e-6))+0.01 for a in shape_areas]
    else:
        weights = [1]*len(shape_areas)
    total = sum(weights)
    weights = [w/total for w in weights]
    new_idx = random.choices(range(len(shape_areas)), weights, k=1)[0]
    while gsa_cSI != new_idx:
        game.execute("switchshape")
        _, gsa_cSI, _, _, _, _ = game.execute("export")
    shape = game.shapes[new_idx]
    h, w = shape.shape
    x = random.randrange(grid_size - w + 1)
    y = random.randrange(grid_size - h + 1)
    gsa_shapePos = goto_target(gsa_shapePos, [x, y])
    gsb_sP, gsb_cSI, gsb_cCI, gsb_grid, _, _ = game.execute("export")
    if game.canPlace(gsb_grid, shape, [x, y]):
        allowed = get_allowed_colors_for_placement(gsb_grid, shape, [x, y], len(game.colors))
        if allowed:
            c = random.choice(allowed)
            while gsb_cCI != c:
                game.execute("switchcolor")
                _, _, gsb_cCI, _, _, _ = game.execute("export")
            game.execute("place")
            _, _, _, cand_grid, cand_shapes, _ = game.execute("export")
        else:
            cand_grid, cand_shapes = gsa_grid, gsa_shapes
    else:
        cand_grid, cand_shapes = gsa_grid, gsa_shapes
    new_cost = evaluate_cost(cand_grid, cand_shapes)
    delta = new_cost - current_cost
    if delta < 0 or random.random() < math.exp(-delta/T):
        current_grid, current_shapes, current_cost = cand_grid.copy(), list(cand_shapes), new_cost
        if current_cost < best_cost:
            best_cost, best_shapes, best_grid = current_cost, list(current_shapes), current_grid.copy()
    else:
        pass
    T *= alpha
print(f"SA done. Best cost: {best_cost}")

# write outputs
np.savetxt("grid.txt", best_grid, fmt="%d")
with open("shapes.txt", "w") as f: f.write(str(best_shapes))
with open("time.txt", "w") as f: f.write(str(time.time()-start_time))
