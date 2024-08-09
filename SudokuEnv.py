import torch as pt

class SudokuEnv():
    def reset(self, puzzles: pt.Tensor, solns: pt.Tensor) -> pt.Tensor:
        ''' Resets the environment to new puzzles and solutions
            Arguments:
            - puzzles: Tensor in the form of nn_input
            - solns: solns[i] = soln(puzzles[i]) also in form of nn_input
            - both are assumed to be shuffled
            Returns: puzzles as inputted
        '''
        self.state = puzzles.float()
        self.solns = solns

        return self.state.float()
    
    def step(self, actions: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        ''' actions should have shape (m, 2) where n is type of sudoku, and
            m = number of sudokus inputted in self.reset().
            actions[i, 0] holds the index to update of the ith puzzle
            while actions[i, 1] holds the value to update the ith puzzle
            returns (next_state, reward, done) as a tuple of tensors
        '''
        state_copy = self.state.detach().int()
        solns = self.solns.detach()
        m = actions.size(0)

        # Extract columns and values from actions
        j = actions[:, 0]
        new_values = actions[:, 1].int()

        # Initialize reward and done tensors
        reward = pt.zeros((m,), dtype=pt.float32)
        done = pt.full((m,), fill_value=False, dtype=pt.bool)

        mask_zero = (state_copy[pt.arange(m), j] == 0)

        state_copy[mask_zero, j[mask_zero]] = new_values[mask_zero]
        
        
        done[mask_zero] = (state_copy != solns)[mask_zero, j[mask_zero]]
        reward[mask_zero & done] = -1024
        reward[mask_zero & ~done] = 16

        mask_equal = pt.all(state_copy == solns, dim=1, keepdim=True).flatten()
        done[mask_equal] = True
        reward[mask_equal & mask_zero] = 1024


        self.state = state_copy.float()

        return self.state, reward.reshape((m, 1)), done.reshape((m, 1))