import numpy as np

from typing import List, Tuple

def puzzle_from_string(string: str, n: int = 3) -> np.ndarray:
    ''' Takes a string of digits which can contain '.', 'X', 'x' and
        returns a formatted sudoku puzzle of size n**2 x n**2
    '''
    string = string.replace('.', '0')
    string = string.replace('X', '0')
    string = string.replace('x', '0')
    dim = n * n
    return np.asarray(list(string), dtype=int).reshape((dim, dim))

def string_from_puzzle(puzzle: np.ndarray) -> str:
    ''' Given a numpy 2d array, returns a string representation of the board.
        This is the inverse operation of puzzle_from_string()
        i.e., x = string_from_puzzle(puzzle_from_string(x))
    '''
    string = ''
    for i in np.ravel(puzzle):
        string += str(i)
    return string

def get_geometric_symmetries(puzzle: np.ndarray, solution: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    ''' Produces a list with the following symmetries of the puzzle and its solution:
        1.  Identity: i.e., the input (puzzle, solution)
        2.  90 degree CW rotation
        3.  180 deg rotation
        4.  270 deg CW rotation
        5.  Horizontal reflection
        6.  Vertical reflection
        7.  Diagonal reflection--top left to bottom right
        8.  Diagonal reflection--top right to bottom left
        List returned is of tuples T s.t. T[0] = puzzle' and T[1] = soln(puzzle')
        Returns in unspecified order
    '''
    raise NotImplementedError()


def get_nongeometric_symmetries(puzzle: np.ndarray, solution: np.ndarray, n: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    ''' Produces a list with the following symmetries of the puzzle and its solution:
        1.  Permutation of major stacks (n! of them)
        2.  Permutation of major bands (n! of them)
        (Overall (n!)**2 nongeometric symmetries)
        List returned is of tuples T s.t. T[0] = puzzle' and T[1] = soln(puzzle')
        Returns in unspecified order
    '''
    raise NotImplementedError()
