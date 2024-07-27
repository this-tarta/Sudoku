import torch as pt
from ortools.sat.python import cp_model
from typing import Tuple
from utils import puzzle_from_string

def solve_sudoku(initial_grid: pt.Tensor, cell_size: int = 3, max_sols: int = 2) -> Tuple[pt.Tensor, int]:
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
            init = initial_grid[i][j].item()
            if init:
                model.add(grid[(i, j)] == init)

    # Solves and prints out the solution.
    solver = cp_model.CpSolver()
    counter = SolutionCounter(max_sols)
    solver.parameters.enumerate_all_solutions = True
    status = solver.solve(model, counter)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        l = [str(i) for i in solver.values(grid.values()).to_list()] # sacrifices efficiency for consistency
        return puzzle_from_string(''.join(l)), counter.get_count()
    else:
        return pt.tensor([], dtype=int), 0

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
