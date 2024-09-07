import torch as pt

class SudokuEnv():
    def __init__(self, win_reward: float = 1024, mistake_reward: float = -1024, good_reward: float = 16, same_reward: float = -1):
        self.win_reward = win_reward
        self.mistake_reward = mistake_reward
        self.good_reward = good_reward
        self.same_reward = same_reward

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

        # TODO: change back
        state_copy[pt.arange(m), j] = new_values

        reward = -pt.log(pt.linalg.vector_norm((state_copy - solns).float(), ord=2, dim=1, keepdim=True) + 0.1)     # +0.1 upper bounds the function to 1
        done = (state_copy == solns).all(dim=1, keepdim=True) | ((state_copy != solns) & (state_copy != 0)).any(dim=1, keepdim=True)

        self.state = state_copy.float()
        return self.state, reward, done