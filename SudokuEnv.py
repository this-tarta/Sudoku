import pandas as pd
import torch as pt
import numpy as np

from math import ceil
from concurrent.futures import ProcessPoolExecutor

from utils import nn_input
from utils import puzzle_from_string
from utils import get_all_symmetries

class SudokuEnv():
    def __init__(self, db_path: str, n: int = 3,
                 chunksize: int = 1024, num_workers:int = 1):
        self.db_path = db_path
        self.n = n
        self.chunksize = chunksize
        self.num_workers = num_workers
        self.it = iter([])
        self.exec = ProcessPoolExecutor(self.num_workers)
    
    def __del__(self):
        self.exec.shutdown(wait=False, cancel_futures=True)

    def reset(self) -> pt.Tensor:
        chunk = next(self.it, None)
        if chunk is None:
            self.it = (pd.read_sql_query('SELECT * FROM "Sudoku" WHERE test=false;',
                       con=self.db_path,
                       coerce_float=False, chunksize=self.chunksize))
            chunk = next(self.it)
        puzzles = [puzzle_from_string(p, self.n) for p in chunk['puzzle'].tolist()]
        solutions = [puzzle_from_string(s, self.n) for s in chunk['solution'].tolist()]
        split_size =  int(ceil(len(puzzles) / self.num_workers))
        futures = []
        p_list = []
        s_list = []
        for i in (split_size * np.arange(self.num_workers)):
            j = i + split_size
            futures.append(self.exec.submit(f_pool, puzzles[i:j], solutions[i:j], self.n))
        for f in futures:
            p, s = f.result(timeout=5)
            p_list.append(p)
            s_list.append(s)
        p_list = pt.cat(p_list, dim = 2)
        s_list = pt.cat(s_list, dim = 2)
        self.state = nn_input(p_list, self.n)
        self.solns = nn_input(s_list, self.n).numpy()
        return self.state
    
    def step(self, actions: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        ''' actions should have shape (m, 2) where n is type of sudoku, and
            m = 8 * (n!)**2 * self.chunksize. actions[i, 0] holds the index to update of the ith puzzle
            while actions[i, 1] holds the value to update the ith puzzle
            returns (next_state, reward, done) as a tuple of tensors
        '''
        state_copy = self.state.numpy(force = True)
        m = len(state_copy)
        done = np.zeros((m, 1), dtype=bool)
        reward = np.zeros((m, 1), dtype=int)
        for i in range(m):
            j = actions[i, 0].item()
            state_copy[i, j] = actions[i, 1].item()
            if state_copy[i, j] != self.solns[i, j]:
                reward[i, 0] = -1024
                done[i, 0] = True
            elif np.equal(state_copy[i], self.solns[i]).all():
                reward[i, 0] = 1024
                done[i, 0] = True
            else:
                reward[i, 0] = 16
                done[i, 0] = False
        
        self.state = pt.from_numpy(state_copy)
        return self.state, pt.from_numpy(reward), pt.from_numpy(done)


def f_pool(puzzles: list, solutions: list, n: int):
    puzzles = pt.cat(puzzles, dim = 2)
    solutions = pt.cat(solutions, dim = 2)
    return get_all_symmetries(puzzles, solutions, n)