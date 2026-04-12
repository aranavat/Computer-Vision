# %%
import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
from matplotlib.animation import FuncAnimation
import json
from IPython.display import HTML, display



# %%
# Grid size
# GRID_SIZE = 25

# # Create empty grid
# grid = np.zeros((GRID_SIZE, GRID_SIZE))

# # Add random obstacles
# obstacle_density = 0.2

# for i in range(GRID_SIZE):
#     for j in range(GRID_SIZE):
#         if random.random() < obstacle_density:
#             grid[i, j] = 1

# # %%
# grid.shape

# %%
# with open('astar_grid_test.json', 'r') as f:
#     all_grids = json.load(f)

# # %%
# all_grids

# # %%
# start = (chosen_grid['start'][0], chosen_grid['start'][1])
# goal = (chosen_grid['goal'][0], chosen_grid['goal'][1])

# # Ensure start and goal are not obstacles
# grid[start] = 0
# grid[goal] = 0

# # %%
# print(start)
# print(goal)

# %%
def heuristic(a, b):
    """
    Compute the Euclidean distance between two grid points.

    Parameters
    ----------
    a : tuple[int, int]
        Coordinates of the first point (row, col)
    b : tuple[int, int]
        Coordinates of the second point (row, col)

    Returns
    -------
    float
        Euclidean distance between point a and point b
    """
    return np.linalg.norm(np.array(a) - np.array(b))

# %%
def get_neighbors(node, grid):
    r, c = node
    rows, cols = grid.shape

    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    neighbors = []

    for dr, dc in directions:
        nr, nc = r + dr, c + dc

        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == 0:
                neighbors.append((nr, nc))

    return neighbors

# %%
def astar_with_steps(start, goal, grid):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    steps = []
    closed_set = set()

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        closed_set.add(current)

        steps.append((current, list(open_set), list(closed_set)))

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, steps

        for neighbor in get_neighbors(current, grid):

            tentative_g = g_score[current] + 1

            if neighbor in g_score and tentative_g >= g_score[neighbor]:
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g

            f_score = tentative_g + heuristic(neighbor, goal)
            heapq.heappush(open_set, (f_score, neighbor))

    return None, steps

# %%
# from matplotlib.animation import FuncAnimation
# from IPython.display import HTML, display
# import matplotlib.pyplot as plt
# import numpy as np

# def testing(all_grids):
#     for i, grid_info in enumerate(all_grids):
#         print(f"\nGrid {i}: Start: {grid_info['start']}, Goal: {grid_info['goal']}")

#         grid = np.array(grid_info['grid'])
#         GRID_SIZE = grid.shape[0]
#         start = tuple(grid_info['start'])
#         goal = tuple(grid_info['goal'])

#         # Ensure start and goal are not obstacles
#         grid[start] = 0
#         grid[goal] = 0

#         path, steps = astar_with_steps(start, goal, grid)

#         if path is None:
#             print("No path found")
#         else:
#             print("Path length:", len(path) - 1)

#         fig, ax = plt.subplots(figsize=(6, 6))

#         # Define update INSIDE loop so it captures current variables
#         def update(frame):
#             ax.clear()

#             current, open_nodes, closed_nodes = steps[frame]

#             # obstacles
#             obstacles = np.where(grid == 1)
#             ax.scatter(obstacles[1], obstacles[0], marker="s")

#             # visited
#             if closed_nodes:
#                 cy = [n[0] for n in closed_nodes]
#                 cx = [n[1] for n in closed_nodes]
#                 ax.scatter(cx, cy, alpha=0.3)

#             # frontier
#             if open_nodes:
#                 frontier_nodes = [n[1] for n in open_nodes]
#                 oy = [n[0] for n in frontier_nodes]
#                 ox = [n[1] for n in frontier_nodes]
#                 ax.scatter(ox, oy, marker="x")

#             # current node
#             ax.scatter(current[1], current[0], s=100)

#             # start & goal
#             ax.scatter(start[1], start[0], s=200)
#             ax.scatter(goal[1], goal[0], s=200)

#             # final path
#             if frame == len(steps) - 1 and path:
#                 px = [p[1] for p in path]
#                 py = [p[0] for p in path]
#                 ax.plot(px, py, linewidth=3)

#             ax.set_xlim(0, GRID_SIZE)
#             ax.set_ylim(0, GRID_SIZE)
#             ax.invert_yaxis()
#             ax.grid(True)
#             ax.set_title(f"Grid {i} - Step {frame+1}/{len(steps)}")

#         ani = FuncAnimation(fig, update, frames=len(steps), interval=100)

#         display(HTML(ani.to_jshtml()))

#         plt.close(fig)  # prevent duplicate static plots

# %%
# testing(all_grids)

# # %%
# all_grids[2]

# %%

def update(frame):
    """
    Update function for animation.

    Parameters
    ----------
    frame : int
        Current frame index in the animation

    Returns
    -------
    None
        Updates the matplotlib axis in-place
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.clear()

    current, open_nodes, closed_nodes = steps[frame]

    # obstacles
    obstacles = np.where(grid == 1)
    ax.scatter(obstacles[1], obstacles[0], marker="s", label="Obstacles")

    # visited
    if closed_nodes:
        cy = [n[0] for n in closed_nodes]
        cx = [n[1] for n in closed_nodes]
        ax.scatter(cx, cy, alpha=0.3, label="Visited")

    # frontier
    if open_nodes:
        frontier_nodes = [n[1] for n in open_nodes]
        oy = [n[0] for n in frontier_nodes]
        ox = [n[1] for n in frontier_nodes]
        ax.scatter(ox, oy, marker="x", label="Frontier")

    # current node
    ax.scatter(current[1], current[0], s=100, label="Current")

    # start & goal
    ax.scatter(start[1], start[0], s=200, label="Start")
    ax.scatter(goal[1], goal[0], s=200, label="Goal")

    # final path
    if frame == len(steps) - 1 and path:
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, linewidth=3, label="Path")

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.set_title(f"A* Step {frame+1}/{len(steps)}")

# ani = FuncAnimation(fig, update, frames=len(steps), interval=100)

# from IPython.display import HTML
# HTML(ani.to_jshtml())


import json

def run_from_json(config):
    """
    Run A* pathfinding and visualization from a JSON configuration.

    This function allows users to provide a grid setup (including start,
    goal, and optional metadata) in JSON format. It will parse the input,
    execute the A* algorithm, and display an animated visualization of
    the search process.

    Parameters
    ----------
    config : str or dict
        JSON string or Python dictionary containing grid configuration.
        Expected format:
        {
            "grid": [[0, 1, 0, ...], ...],
            "start": [row, col],
            "goal": [row, col],
            "name": str (optional),
            "description": str (optional)
        }

        Where:
        - 0 represents free space
        - 1 represents obstacles

    Returns
    -------
    None
        - Prints path existence and path length
        - Displays an animated visualization of A* search

    Raises
    ------
    ValueError
        If required keys ('grid', 'start', 'goal') are missing
        or improperly formatted

    Notes
    -----
    - Uses 4-directional movement (up, down, left, right)
    - Assumes uniform movement cost between adjacent cells
    - Requires a valid heuristic function to be defined externally
    """

    # Parse JSON if needed
    if isinstance(config, str):
        config = json.loads(config)

    # Validate required fields
    required_keys = ["grid", "start", "goal"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: '{key}'")

    grid = np.array(config["grid"])
    start = tuple(config["start"])
    goal = tuple(config["goal"])

    name = config.get("name", "Custom Grid")
    description = config.get("description", "")

    print(f"\n{name}")
    if description:
        print(description)
    print(f"Start: {start}, Goal: {goal}")

    # Ensure valid start/goal
    grid[start] = 0
    grid[goal] = 0

    # Run A*
    path, steps = astar_with_steps(start, goal, grid)

    if path is None:
        print("No path found")
    else:
        print("Path length:", len(path) - 1)
        print(path)

    # Visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    rows, cols = grid.shape

    def update(frame):
        ax.clear()

        current, open_nodes, closed_nodes = steps[frame]

        # obstacles
        obstacles = np.where(grid == 1)
        ax.scatter(obstacles[1], obstacles[0], marker="s")

        # visited
        if closed_nodes:
            cy = [n[0] for n in closed_nodes]
            cx = [n[1] for n in closed_nodes]
            ax.scatter(cx, cy, alpha=0.3)

        # frontier
        if open_nodes:
            frontier_nodes = [n[1] for n in open_nodes]
            oy = [n[0] for n in frontier_nodes]
            ox = [n[1] for n in frontier_nodes]
            ax.scatter(ox, oy, marker="x")

        # current node
        ax.scatter(current[1], current[0], s=100)

        # start & goal
        ax.scatter(start[1], start[0], s=200)
        ax.scatter(goal[1], goal[0], s=200)

        # final path
        if frame == len(steps) - 1 and path:
            px = [p[1] for p in path]
            py = [p[0] for p in path]
            ax.plot(px, py, linewidth=3)

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()
        ax.grid(True)
        ax.set_title(f"{name} - Step {frame+1}/{len(steps)}")

    ani = FuncAnimation(fig, update, frames=len(steps), interval=100)

    display(HTML(ani.to_jshtml()))
    plt.close(fig)

# run_from_json(all_grids[0])