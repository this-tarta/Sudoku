import torch as pt

from typing import List, Tuple
from copy import deepcopy
# from sudoku_sat import solve_sudoku

def puzzle_from_string(string: str, n: int = 3) -> pt.Tensor:
    ''' Takes a string of digits which can contain '.', 'X', 'x' and
        returns a formatted sudoku puzzle of size n**2 x n**2
    '''
    string = string.replace('.', '0')
    string = string.replace('X', '0')
    string = string.replace('x', '0')
    dim = n * n
    return pt.tensor([int(i) for i in string]).reshape((dim, dim))

def string_from_puzzle(puzzle: pt.Tensor) -> str:
    ''' Given a tensor 2d array, returns a string representation of the board.
        This is the inverse operation of puzzle_from_string()
        i.e., x = string_from_puzzle(puzzle_from_string(x))
    '''
    string = ''
    for i in pt.flatten(puzzle):
        string += str(i.item())
    return string

def get_geometric_symmetries(puzzle: pt.Tensor, solution: pt.Tensor) -> List[Tuple[pt.Tensor, pt.Tensor]]:
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
    def diaglr(grid: pt.Tensor) -> pt.Tensor:
        return pt.flipud(pt.rot90(grid, -1))
    
    def diagrl(grid: pt.Tensor) -> pt.Tensor:
        return pt.flipud(pt.rot90(grid, 1))

    syms = [(puzzle, solution)]
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
        syms.append((puzzle_p, solution_p))
    
    return syms


def get_nongeometric_symmetries(puzzle: pt.Tensor, solution: pt.Tensor, n: int = 3) -> List[Tuple[pt.Tensor, pt.Tensor]]:
    ''' Produces a list with the following symmetries of the puzzle and its solution:
        1.  Permutation of major stacks (n! of them)
        2.  Permutation of major bands (n! of them)
        (Overall (n!)**2 nongeometric symmetries)
        List returned is of tuples T s.t. T[0] = puzzle' and T[1] = soln(puzzle'), including the original puzzle and soln
        Returns in unspecified order
    '''
    indices_b = permute(list(range(n)))
    indices_s = deepcopy(indices_b)
    syms = []
    for i in indices_b:
        band_p = pt.cat([puzzle[:, n * x : n * (x + 1)] for x in i], dim=1)
        band_s = pt.cat([solution[:, n * x : n * (x + 1)] for x in i], dim=1)
        for j in indices_s:
            stack_p = pt.cat([band_p[n * x : n * (x + 1), :] for x in j], dim=0)
            stack_s = pt.cat([band_s[n * x : n * (x + 1), :] for x in j], dim=0)
            syms.append((stack_p, stack_s))
    return syms

def get_all_symmetries(puzzle: pt.Tensor, solution: pt.Tensor, n: int = 3) -> List[Tuple[pt.Tensor, pt.Tensor]]:
    ''' Returns the list of all symmetries as produced by get_nongeometric_symmetries and get_geometric_symmetries '''
    nongeo_syms = get_nongeometric_symmetries(puzzle, solution, n)
    syms = []
    for p, s in nongeo_syms:
        syms += get_geometric_symmetries(p, s)
    
    return syms

def permute(list: List) -> List[List]:
    ''' Returns a list of all permutations of the input list '''
    if (len(list) <= 1):
        return [list]
    permutations = []
    for i in range(len(list)):
        list[0], list[i] = list[i], list[0]
        permutations += [[list[0]] + p for p in permute(list[1:])]
    return permutations

# puzzle = puzzle_from_string('620740100070100052508000370067300900090000060800970031002000006000800000450002003')
# solution = puzzle_from_string('623745198974138652518269374267381945391524867845976231782493516136857429459612783')

# p2 = puzzle_from_string('900000002010060390083900100804095007130670049060041000302010050000500000541080030')
# s2 = puzzle_from_string('956134782417268395283957164824395617135672849769841523372416958698523471541789236')

# cated_p = pt.stack([puzzle, p2], dim=2)
# cated_s = pt.stack([solution, s2], dim=2)

# grids = get_all_symmetries(puzzle, solution)
# print(len(grids))
# print(grids[0][0].shape)
# flat = pt.reshape(grids[0][0], shape=(81, 2))
# for f in pt.transpose(flat, dim0=0, dim1=1):
#     print(f)
#     print('-------------------')
# unstacked = []
# for i in range(2):
#     unstacked.append(grids[0][0][:,:,i])
#     print(unstacked[i])
#     print('----------------')

# for i in range(len(grids)):
#     for j in range(i + 1, len(grids)):
#         assert not pt.equal(grids[i][0], grids[j][0])
    
#     assert pt.equal(pt.from_numpy(solve_sudoku(grids[i][0].detach().numpy())[0]),
#                     grids[i][1]) # requires import from sudoku_sat
