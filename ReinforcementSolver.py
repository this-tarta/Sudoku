import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from copy import deepcopy

from utils import parse_cmd, plot_stats
from SudokuEnv import SudokuEnv
from DB_Management import SudokuDataset
from SudokuNet import SudokuNet, SudokuNetClassifier
from ClassifierSolver import train

def main():
    args = {
        'epochs': {
            'type': int,
            'default': 128,
            'help': 'the number of epochs to run'
        },

        'CUDA': {
            'type': bool,
            'help': 'enables using CUDA'
        },

        'learning_rate': {
            'type': float,
            'help': 'learning rate of the gradient descent',
            'default': 1e-3
        },

        'dbpath': {
            'type': str,
            'help': 'path of the database',
            'default': 'postgresql:///Sudoku'
        },

        'tablename': {
            'type': str,
            'help': 'name of the table in the db',
            'default': 'Sudoku'
        },

        'symmetries': {
            'type': bool,
            'help': 'enables use of the Sudoku symmetries'
        },

        'batch_size': {
            'type': int,
            'default': 128,
            'help': 'the size of the batch to load'
        },

        'exploration_rate': {
            'type': float,
            'default': 0.2,
            'help': 'the probability [0, 1] that a random action is taken'
        },

        'exploration_decay': {
            'type': float,
            'default': 0.96,
            'help': 'the factor at which the exploration rate decays'
        }
    }
    args = parse_cmd(args, progname='ReinforcementSolver.py', description='Solves Sudoku through deep reinforcement learning methods')
    print(args)
    env = SudokuEnv(win_reward=1, mistake_reward=-1, good_reward=1, same_reward=-0.05)
    sudokus = SudokuDataset(db_path=args['dbpath'], table_name=args['tablename'], n=3, include_symmetries=args['symmetries'])
    device = 'cuda' if torch.cuda.is_available() and args['CUDA'] else 'cpu'
    trn, val, test = random_split(sudokus, [1e-4, 1e-5, 1 - 1e-4 - 1e-5])
    batch_size = args['batch_size']
    trainloader = DataLoader(trn, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val, batch_size=batch_size, shuffle=False)
    demonstrator = SudokuNetClassifier(hidden_size=512, num_convs=16, n=3).to(device)
    # stats = train(demonstrator, 450, trainloader, alpha=1e-3, device=device, filename='classifier')
    # plot_stats(stats)
    dem_memory = fill_demonstration_replay(env, demonstrator, trainloader, device=device)
    # demonstrator = demonstrator.to('cpu')
    # net = SudokuNet(hidden_size=512, num_convs=8, n=3).to(device)
    # stats = deep_q_learning(env, net, args['epochs'], trainloader, alpha=args['learning_rate'], device=device, epsilon=args['exploration_rate'], epsilon_decay_rate=args['exploration_decay'],
    #                         demonstration_mem=dem_memory, demonstration_split=1)
    # plot_stats(stats)
    # stats = actor_critic(env, Actor().to(device), Critic().to(device), args['epochs'], trainloader, alpha_actor=1e-6, alpha_critic=1e-6, device=device, entropy_regularization=0.1)
    # plot_stats(stats)

# class Actor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.sd_net = SudokuNet(hidden_size=512, num_convs=12, n=3)
    
#     def forward(self, x):
#         x = self.sd_net.forward(x).transpose(1, 2).flatten(1)
#         return F.softmax(x, dim=-1)

# class Critic(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.sd_net = SudokuNet(hidden_size=512, num_convs=12, n=3)
#         self.lins = nn.Sequential(nn.Linear(810, 2048), nn.ReLU(), nn.Linear(2048, 1))
    
#     def forward(self, x):
#         x = self.sd_net.forward(x).transpose(1, 2).flatten(1)
#         return self.lins.forward(x)


def av_to_actions(av: torch.Tensor, epsilon: float = 0.) -> torch.Tensor:
    ''' av is a (n, 10, 81) tensor
        returns a tensor T: (n, 2) where T[i,0] is the index of the cell for game i and T[i, 1] is the entry for the cell 
    '''
    n = av.size(0)
    action = av.transpose(dim0=1, dim1=2).flatten(1).argmax(dim=1, keepdim=True).to('cpu')
    index = action // 10
    entry = action % 10
    mask = (torch.rand(size=(n,)) < epsilon)
    index[mask] = (torch.randint(low=0, high=81, size=index.size()))[mask]
    entry[mask] = (torch.randint(low=0, high=10, size=entry.size()))[mask]
    return torch.cat([index, entry], dim=1)

def index_to_actions(idx: torch.Tensor) -> torch.Tensor:
    index = idx // 10
    entry = idx % 10
    return torch.cat([index, entry], dim=1)

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

def fill_demonstration_replay(env: SudokuEnv, demonstrator: SudokuNetClassifier, trainloader: DataLoader, mem_capacity: int = 10000000,
                              device: torch.DeviceObjType = 'cpu') -> ReplayMemory:
    dem_memory = ReplayMemory(mem_capacity)

    for p_batch, s_batch in tqdm(trainloader):
        state: torch.Tensor = env.reset(p_batch, s_batch).to(device)
        done_batch = torch.zeros((state.size(0), 1), dtype=torch.bool)
        steps = torch.zeros((state.size(0), 1), dtype=torch.int)
        while not done_batch.all():
            steps += ~done_batch
            actions = index_to_actions(torch.argmax(
                        (demonstrator(state).transpose(1, 2) - F.one_hot(state.long(), num_classes=10)).flatten(1),
                        dim=1, keepdim=True
            ))
            j = torch.arange(state.size(0))
            next_state, reward, done = env.step(actions, device)
            reward = ~done_batch.to(device, copy=True) * reward
            done_batch = done_batch.squeeze()
            dem_memory.insert([state[~done_batch].to('cpu', copy=True),
                                actions[~done_batch].to('cpu', copy=True),
                                reward[~done_batch].to('cpu', copy=True),
                                done[~done_batch].to('cpu', copy=True),
                                next_state[~done_batch].to('cpu', copy=True)])
            done_batch = done_batch.unsqueeze(1)
            done_batch |= done.to('cpu')
            state = next_state.to(device)
        print('Mean: ', steps.float().mean().item())
        print('Min: ', steps.min().item())
        print('Max: ', steps.max().item())
    return dem_memory

def deep_q_learning(env: SudokuEnv, q_network: nn.Module, epochs: int, trainloader: DataLoader, demonstration_mem: ReplayMemory | None = None,
                alpha: float = 1e-4, epsilon: float = 0.2, epsilon_decay_rate: float = 0.96, gamma: float = 0.99, copy_rate: int = 10, demonstration_split: float = 0.5,
                device: torch.DeviceObjType = 'cpu', filename: str | None = None) -> dict[str, list[float]]:
    
    optim = AdamW(q_network.parameters(), lr=alpha)
    stats = {'Training Loss': [], 'Training Return': []}
    
    replay_mem = ReplayMemory(1000000)
    
    target_network = deepcopy(q_network).to(device)
    target_network.eval()
    for i in tqdm(range(epochs)):
        for idx_batch, (p_batch, s_batch) in enumerate(trainloader):
            state: torch.Tensor = env.reset(p_batch, s_batch).to(device)
            done_batch = torch.zeros((state.size(0), 1), dtype=torch.bool)
            returns = torch.zeros((state.size(0), 1), dtype=torch.int)
            while not done_batch.all():
                av = q_network(state)
                actions = av_to_actions(av, epsilon=epsilon)
                next_state, reward, done = env.step(actions, device)
                reward = ~done_batch.to(device, copy=True) * reward
                returns += reward
                done_batch = done_batch.squeeze()
                replay_mem.insert([state[~done_batch].to('cpu', copy=True),
                                   actions[~done_batch].to('cpu', copy=True),
                                   reward[~done_batch].to('cpu', copy=True),
                                   done[~done_batch].to('cpu', copy=True),
                                   next_state[~done_batch].to('cpu', copy=True)])
                done_batch = done_batch.unsqueeze(1)
                done_batch |= done.to('cpu')
                state = next_state.to(device)

                replay_batch_size = len(p_batch) if demonstration_mem is None else int((1 - demonstration_split) * len(p_batch))
                if replay_mem.can_sample(replay_batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = replay_mem.sample(replay_batch_size) if replay_batch_size > 0 else [
                        torch.FloatTensor(), torch.LongTensor(), torch.FloatTensor(), torch.BoolTensor(), torch.FloatTensor()]
                    if demonstration_mem is not None:
                        dem_batch_size = len(p_batch) - replay_batch_size
                        state_d, action_d, reward_d, done_d, next_state_d = demonstration_mem.sample(dem_batch_size)
                        state_b = torch.cat([state_b, state_d])
                        action_b = torch.cat([action_b, action_d])
                        reward_b = torch.cat([reward_b, reward_d])
                        done_b = torch.cat([done_b, done_d])
                        next_state_b = torch.cat([next_state_b, next_state_d])
                    
                    state_b, action_b, reward_b, done_b, next_state_b = [t.to(device) for t in [state_b, action_b, reward_b, done_b, next_state_b]]

                    q_network.eval()
                    j = torch.arange(state_b.size(0))
                    qnet_qsa = q_network(state_b)[j, action_b[j, 1], action_b[j, 0]]  # get predicted q values for chosen actions
                    q_network.train()
                    target_qsa = torch.max(target_network(next_state_b).flatten(1), dim=1, keepdim=True).values

                    loss_t = F.mse_loss(qnet_qsa, (target_qsa * gamma + reward_b).flatten())  # Q-loss

                    q_network.zero_grad()
                    loss_t.backward()
                    optim.step()

                    stats['Training Loss'].append(loss_t.item())
            stats['Training Return'].append(returns.mean().item())
            if idx_batch % copy_rate == 0:
                sd = q_network.state_dict()
                target_network.load_state_dict(sd)
                target_network.eval()
                
        
        if i % 10 == 0 and filename is not None:
            torch.save(q_network, filename + '.pkl')
            torch.save(optim, f=filename + '_adam.pkl')
            torch.save(stats, f=filename + '_stats.pkl')
        
        epsilon *= epsilon_decay_rate

    if filename is not None:
        torch.save(q_network, filename + '.pkl')
        torch.save(optim, f=filename + '_adam.pkl')
        torch.save(stats, f=filename + '_stats.pkl')
    
    return stats

# def actor_critic(env: SudokuEnv, actor: nn.Module, critic: nn.Module, epochs: int, trainloader: DataLoader, valloader: DataLoader = None,
#                 alpha_actor: float = 1e-4, alpha_critic: float = 1e-4, gamma: float = 0.99, entropy_regularization: float = 0.01,
#                 device: torch.DeviceObjType = 'cpu', filename: str | None = None) -> dict[str, list[float]]:
#     optim_actor = AdamW(actor.parameters(), lr=alpha_actor)
#     optim_critic = AdamW(critic.parameters(), lr=alpha_critic)
#     stats = {'Actor Loss': [], 'Critic Loss': [], 'Training Return': []}
#     if valloader is not None:
#         stats['Validation Return'] = []
    
#     for i in tqdm(range(epochs)):
#         for idx_batch, (p_batch, s_batch) in enumerate(trainloader):
#             state: torch.Tensor = env.reset(p_batch, s_batch).to(device)
#             done_b = torch.zeros((state.size(0), 1), dtype=torch.bool)
#             ep_return = torch.zeros((state.size(0), 1))
#             I = 1.

#             while not done_b.all():
#                 indices = actor(state).multinomial(1).detach()
#                 action = index_to_actions(indices)
#                 next_state, reward, done = env.step(action, device)

#                 value = critic(state)
#                 target = reward + ~done * gamma * critic(next_state).detach()
#                 critic_loss = F.mse_loss(value[~done_b[:, 0]], target[~done_b[:, 0]])
#                 critic.zero_grad()
#                 critic_loss.backward()
#                 optim_critic.step()

#                 advantage = (target - value).detach()[~done_b[:, 0]]
#                 probs = actor(state)[~done_b[:, 0]]
#                 log_probs = torch.log(probs + 1e-6)
#                 action_log_prob = log_probs.gather(1, indices[~done_b[:, 0]])
#                 entropy = - torch.sum(probs * log_probs, dim=-1, keepdim=True)
#                 actor_loss = - I * action_log_prob * advantage - entropy_regularization * entropy
#                 actor_loss = actor_loss.mean()
#                 actor.zero_grad()
#                 actor_loss.backward()
#                 optim_actor.step()

#                 ep_return += reward.to('cpu') * ~done_b
#                 done_b |= done.to('cpu')
#                 state = next_state
#                 I = I * gamma
            
#             stats['Actor Loss'].append(actor_loss.item())
#             stats['Critic Loss'].append(critic_loss.item())
#             stats['Training Return'].append(ep_return.mean().item())

#     return stats

if __name__ == '__main__':

    main()
