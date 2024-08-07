import torch as pt
import numpy as np

class SudokuEnv():
    def reset(self, puzzles: pt.Tensor, solns: pt.Tensor) -> pt.Tensor:
        ''' Resets the environment to new puzzles and solutions
            Arguments:
            - puzzles: Tensor in the form of nn_input
            - solns: solns[i] = soln(puzzles[i]) also in form of nn_input
            - both are assumed to be shuffled
            Returns: puzzles as inputted
        '''
        self.state = puzzles
        self.solns = solns.numpy()

        return self.state
    
    def step(self, actions: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        ''' actions should have shape (m, 2) where n is type of sudoku, and
            m = number of sudokus inputted in self.reset().
            actions[i, 0] holds the index to update of the ith puzzle
            while actions[i, 1] holds the value to update the ith puzzle
            returns (next_state, reward, done) as a tuple of tensors
        '''
        state_copy = self.state.numpy(force = True)
        m = len(state_copy)
        done = np.zeros((m, 1), dtype=bool)
        reward = np.zeros((m, 1), dtype=int)
        for i in range(m):
            j = actions[i, 0].item()
            if state_copy[i, j] != 0:
                reward[i, 0] = 0
                done[i, 0] = np.equal(state_copy[i], self.solns[i]).all()
                continue
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