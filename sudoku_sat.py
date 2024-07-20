import numpy as np
from ortools.sat.python import cp_model
from typing import Tuple

def solve_sudoku(initial_grid: np.ndarray, cell_size: int = 3, max_sols: int = 2) -> Tuple[np.ndarray, int]:
    """Solves the sudoku problem with the CP-SAT solver.
        Arguments: 
        - cell_size: the shape of the sudoku puzzle (cell_size = 3 for standard sudoku)
        - initial_grid: an unsolved sudoku; the grid should have a cell_size**2 x cell_size**2 shape
        - max_sols: the maximum number of feasible solutions to search over
        Returns:
        - a tuple containing a solution to the puzzle and the number of solutions <= max_sols
            - if there is no solution, returns ([], 0)
    """
    # Create the model.
    model = cp_model.CpModel()

    cell_size = 3
    line_size = cell_size * cell_size
    line = list(range(0, line_size))
    cell = list(range(0, cell_size))

    grid = {}
    for i in line:
        for j in line:
            grid[(i, j)] = model.new_int_var(1, line_size, "grid %i %i" % (i, j))

    # AllDifferent on rows.
    for i in line:
        model.add_all_different(grid[(i, j)] for j in line)

    # AllDifferent on columns.
    for j in line:
        model.add_all_different(grid[(i, j)] for i in line)

    # AllDifferent on cells.
    for i in cell:
        for j in cell:
            one_cell = []
            for di in cell:
                for dj in cell:
                    one_cell.append(grid[(i * cell_size + di, j * cell_size + dj)])

            model.add_all_different(one_cell)

    # Initial values.
    for i in line:
        for j in line:
            if initial_grid[i][j]:
                model.add(grid[(i, j)] == initial_grid[i][j])

    # Solves and prints out the solution.
    solver = cp_model.CpSolver()
    counter = SolutionCounter(max_sols)
    solver.parameters.enumerate_all_solutions = True
    status = solver.solve(model, counter)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return np.asarray([[int(solver.value(grid[(i, j)])) for j in line] for i in line]), counter.get_count()
    else:
        return np.asarray([], dtype=int), 0

class SolutionCounter(cp_model.CpSolverSolutionCallback):
    def __init__(self, max_sols):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.count = 0
        self.max_sols = max_sols

    def on_solution_callback(self):
        self.count += 1
        if self.count >= self.max_sols:
            self.stop_search()

    def get_count(self) -> int:
        return self.count


print(solve_sudoku(initial_grid = np.asarray([
        [0, 0, 5, 7, 4, 3, 8, 6, 1],
        [4, 3, 1, 8, 6, 5, 9, 0, 0],
        [8, 7, 6, 1, 9, 2, 5, 4, 3],
        [3, 8, 7, 4, 5, 9, 2, 1, 6],
        [6, 1, 2, 3, 8, 7, 4, 9, 5],
        [5, 4, 9, 2, 1, 6, 7, 3, 8],
        [7, 6, 3, 5, 2, 4, 1, 8, 9],
        [0, 0, 8, 6, 7, 1, 3, 5, 4],
        [1, 5, 4, 9, 3, 8, 6, 0, 0]
    ])))
