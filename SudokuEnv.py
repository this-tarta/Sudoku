import random
import torch

class SudokuEnv():
    def __init__(self, win_reward: float = 1024, mistake_reward: float = -1024, good_reward: float = 16, same_reward: float = 0):
        self.win_reward = win_reward
        self.mistake_reward = mistake_reward
        self.good_reward = good_reward
        self.same_reward = same_reward

    def reset(self, puzzles: torch.Tensor, solns: torch.Tensor) -> torch.Tensor:
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
    
    def step(self, actions: torch.Tensor, device = 'cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        state_copy[torch.arange(m), j] = new_values

        same = (state_copy == self.state.int().to(device)).all(dim=1, keepdim=True)
        win = (state_copy == solns).all(dim=1, keepdim=True)
        mistake = ((state_copy != solns) & (state_copy != 0)).any(dim=1, keepdim=True) | (new_values == 0).unsqueeze(1)

        done = win | mistake
        reward = win * self.win_reward + same * self.same_reward + mistake * self.mistake_reward + ~(win | mistake | same) * self.good_reward

        self.state = state_copy.float()
        return self.state, reward, done

class ReplayMemory():
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.state_memory = list(range(capacity))
        self.action_memory = list(range(capacity))
        self.reward_memory = list(range(capacity))
        self.done_memory = list(range(capacity))
        self.next_memory = list(range(capacity))
        self.curr = 0
        self.size = 0

    def insert(self, transition: list[torch.Tensor]):
        state, action, reward, done, next_state = transition
        for i in range(len(state)):
            self.state_memory[self.curr] = state[i].unsqueeze(0)
            self.action_memory[self.curr] = action[i].unsqueeze(0)
            self.reward_memory[self.curr] = reward[i].unsqueeze(0)
            self.done_memory[self.curr] = done[i].unsqueeze(0)
            self.next_memory[self.curr] = next_state[i].unsqueeze(0)
            self.curr = (self.curr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[torch.Tensor]:
        assert self.can_sample(batch_size)

        batch_indices = random.sample(range(self.size), batch_size)
        def f(list, indices):
            return [list[i] for i in indices]

        return [
            torch.cat(f(self.state_memory, batch_indices), dim=0),
            torch.cat(f(self.action_memory, batch_indices), dim=0),
            torch.cat(f(self.reward_memory, batch_indices), dim=0),
            torch.cat(f(self.done_memory, batch_indices), dim=0),
            torch.cat(f(self.next_memory, batch_indices), dim=0)
        ]

    def can_sample(self, batch_size: int):
        return self.size >= batch_size * 10

    def __len__(self):
        return self.size