import torch as pt

class SudokuEnv():
    def __init__(self, win_reward: float = 1024, mistake_reward: float = -1024, good_reward: float = 16, end_fail: bool = True):
        self.win_reward = win_reward
        self.mistake_reward = mistake_reward
        self.good_reward = good_reward
        self.end_fail = end_fail

    def reset(self, puzzles: pt.Tensor, solns: pt.Tensor) -> pt.Tensor:
        ''' Resets the environment to new puzzles and solutions
            Arguments:
            - puzzles: Tensor in the form of nn_input
            - solns: solns[i] = soln(puzzles[i]) also in form of nn_input
            - both are assumed to be shuffled
            Returns: puzzles as inputted, on CPU
        '''
        self.state = puzzles.float().to('cpu')
        self.solns = solns.to('cpu')

        return self.state
    
    def step(self, actions: pt.Tensor, device = 'cpu') -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        ''' actions should have shape (m, 2) where n is type of sudoku, and
            m = number of sudokus inputted in self.reset().
            actions[i, 0] holds the index to update of the ith puzzle
            while actions[i, 1] holds the value to update the ith puzzle
            returns (next_state, reward, done) as a tuple of tensors
        '''
        state_copy = self.state.detach().int().to(device)
        solns = self.solns.detach().to(device)
        m = actions.size(0)

        # Extract columns and values from actions
        j = actions[:, 0].to(device)
        new_values = actions[:, 1].int().to(device)

        # Initialize reward and done tensors
        reward = pt.zeros((m,), dtype=pt.float32).to(device)
        done = pt.zeros((m,), dtype=pt.bool).to(device)

        mask_zero = (state_copy[pt.arange(m), j] == 0).to(device)

        state_copy[mask_zero, j[mask_zero]] = new_values[mask_zero]
        
        
        done[mask_zero] = (state_copy != solns)[mask_zero, j[mask_zero]]
        done[~mask_zero] = True
        reward[mask_zero & done | ~mask_zero] = self.mistake_reward
        reward[mask_zero & ~done] = self.good_reward
        done[done] = self.end_fail

        mask_equal = pt.all(state_copy == solns, dim=1, keepdim=True).flatten().to(device)
        done[mask_equal] = True
        reward[mask_equal & mask_zero] = self.win_reward

        self.state = state_copy.float()

        return self.state, reward.reshape((m, 1)), done.reshape((m, 1))