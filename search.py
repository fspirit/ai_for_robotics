import numpy as np


def run_a_star_search(grid, heuristic, action_cost):

    starting_point = np.array([0, 0])
    final_point = np.array(grid.shape) - 1

    moves = np.array([[-1, 0],  # go up
                      [0, -1],  # go left
                      [1, 0],  # go down
                      [0, 1]])  # go right

    # move_symbols = ['^', '<', 'v', '>']

    def find_path(current_point, current_cost, visited_points, path):

        path.append(current_point)
        visited_points.add(tuple(current_point.tolist()))

        if np.all(current_point == final_point):
            return True

        def can_move_to(p):
            point_in_the_grid = np.all(p >= np.zeros_like(grid.shape)) and np.all(p < np.array(grid.shape))
            return point_in_the_grid and tuple(p.tolist()) not in visited_points and grid[p[0], p[1]] == 0

        next_point_candidates = [current_point + move for move in moves if can_move_to(current_point + move)]

        print(next_point_candidates)

        if len(next_point_candidates) == 0:
            return False

        next_point_candidates = [(p, current_cost + action_cost + heuristic[p[0], p[1]]) for p in next_point_candidates]
        next_point_candidates = sorted(next_point_candidates, key=lambda x: x[1])

        best_next_point = next_point_candidates[0][0]

        return find_path(best_next_point, current_cost + action_cost, visited_points, path)

    path = []
    result = find_path(starting_point, 0, set(), path)

    return result, path

if __name__ == "__main__":

    grid = np.array([[0, 1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0]])
    heuristic = np.array([[9, 8, 7, 6, 5, 4],
                          [8, 7, 6, 5, 4, 3],
                          [7, 6, 5, 4, 3, 2],
                          [6, 5, 4, 3, 2, 1],
                          [5, 4, 3, 2, 1, 0]])

    print('Grid:')
    print(grid)

    result, path = run_a_star_search(grid, heuristic, action_cost=1)

    if result:
        for step in path:
            grid[step[0], step[1]] = -1
        print('Path:')
        print(grid)
    else:
        print('No path found.')



