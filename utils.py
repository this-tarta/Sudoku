import torch as pt

from typing import List, Tuple
from copy import deepcopy

def puzzle_from_string(string: str, n: int = 3) -> pt.Tensor:
    ''' Takes a string of digits which can contain '.', 'X', 'x' and
        returns a formatted sudoku puzzle of size n**2 x n**2
    '''
    string = string.replace('.', '0')
    string = string.replace('X', '0')
    string = string.replace('x', '0')
    dim = n * n
    return pt.tensor([int(i) for i in string]).reshape((dim, dim, 1))

def string_from_puzzle(puzzle: pt.Tensor) -> str:
    ''' Given a tensor 2d array, returns a string representation of the board.
        This is the inverse operation of puzzle_from_string()
        i.e., x = string_from_puzzle(puzzle_from_string(x))
    '''
    string = ''.join([str(i.item()) for i in pt.flatten(puzzle)])
    return string

def get_geometric_symmetries(puzzle: pt.Tensor, solution: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
    ''' Produces a list with the following symmetries of the puzzle and its solution:
        1.  Identity: i.e., the input (puzzle, solution)
        2.  90 degree CW rotation
        3.  180 deg rotation
        4.  270 deg CW rotation
        5.  Horizontal reflection
        6.  Vertical reflection
        7.  Diagonal reflection--top left to bottom right
        8.  Diagonal reflection--top right to bottom left
        Tuple returned of tensors where T[0][:, :, i] = puzzle and T[1][:, :, i] = soln(puzzle)
        Returns in unspecified order
    '''
    def diaglr(grid: pt.Tensor) -> pt.Tensor:
        return pt.flipud(pt.rot90(grid, -1))
    
    def diagrl(grid: pt.Tensor) -> pt.Tensor:
        return pt.flipud(pt.rot90(grid, 1))

    syms_p = [puzzle]
    syms_s = [solution]
    transformations = [
        [pt.rot90, -1],
        [pt.rot90, 2],
        [pt.rot90, 1],
        [pt.fliplr],
        [pt.flipud],
        [diaglr],
        [diagrl]
    ]
    for tform in transformations:
        puzzle_p, solution_p = tform[0](puzzle, *tform[1:]), tform[0](solution, *tform[1:])
        syms_p.append(puzzle_p)
        syms_s.append(solution_p)
    
    return pt.cat(syms_p, dim = 2), pt.cat(syms_s, dim = 2)


def get_nongeometric_symmetries(puzzle: pt.Tensor, solution: pt.Tensor, n: int = 3) -> Tuple[pt.Tensor, pt.Tensor]:
    ''' Produces a list with the following symmetries of the puzzle and its solution:
        1.  Permutation of major stacks (n! of them)
        2.  Permutation of major bands (n! of them)
        (Overall (n!)**2 nongeometric symmetries)
        Tuple returned of tensors where T[0][:, :, i] = puzzle and T[1][:, :, i] = soln(puzzle)
        Returns in unspecified order
    '''
    indices_b = permute(list(range(n)))
    indices_s = deepcopy(indices_b)
    syms_p = []
    syms_s = []
    for i in indices_b:
        band_p = pt.cat([puzzle[:, n * x : n * (x + 1)] for x in i], dim=1)
        band_s = pt.cat([solution[:, n * x : n * (x + 1)] for x in i], dim=1)
        for j in indices_s:
            stack_p = pt.cat([band_p[n * x : n * (x + 1), :] for x in j], dim=0)
            stack_s = pt.cat([band_s[n * x : n * (x + 1), :] for x in j], dim=0)
            syms_p.append(stack_p)
            syms_s.append(stack_s)
    return pt.cat(syms_p, dim = 2), pt.cat(syms_s, dim = 2)

def get_all_symmetries(puzzle: pt.Tensor, solution: pt.Tensor, n: int = 3) -> Tuple[pt.Tensor, pt.Tensor]:
    ''' Returns the list of all symmetries as produced by get_nongeometric_symmetries and get_geometric_symmetries '''
    syms_p, syms_s = get_nongeometric_symmetries(puzzle, solution, n)
    return get_geometric_symmetries(syms_p, syms_s)

def permute(list: List) -> List[List]:
    ''' Returns a list of all permutations of the input list '''
    if (len(list) <= 1):
        return [list]
    permutations = []
    for i in range(len(list)):
        list[0], list[i] = list[i], list[0]
        permutations += [[list[0]] + p for p in permute(list[1:])]
    return permutations

def nn_input(puzzles: pt.Tensor, n: int = 3) -> pt.Tensor:
    ''' Takes n**2 x n**2 x d tensor and returns d x n**4 tensor for NN input
    '''
    d = puzzles.size()[2]
    N = n**4

    return pt.transpose(pt.reshape(puzzles, shape=(N, d)), dim0=0, dim1=1)
