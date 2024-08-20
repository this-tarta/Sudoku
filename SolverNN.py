import torch
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Callable, Tuple
from copy import deepcopy



from SudokuEnv import SudokuEnv
from DB_Management import SudokuLoader

torch.autograd.set_detect_anomaly(True)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q_net = nn.Sequential(
        nn.Unflatten(dim=1, unflattened_size=(1, 9, 9)),
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=1),
        nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(490, 2048),
        nn.ReLU(),
        # nn.Unflatten(dim=1, unflattened_size=(2, 32, 32)),
        # nn.Conv2d(in_channels=2, out_channels=9, kernel_size=4, stride=2),
        # nn.MaxPool2d(kernel_size=2, stride=1),
        # nn.Flatten(),
        # nn.Linear(1764, 2048),
        # nn.Sigmoid(),
        nn.Linear(2048, 729)
    ).to(device)

    # with open('model.pkl', 'rb') as f:
    #     policy.load_state_dict(pickle.load(f))
    
    pre_training(q_net, device)


def pre_training(q_net: nn.Module, device = 'cpu'):
    env = SudokuEnv(win_reward=2, mistake_reward=-1, good_reward=1, end_fail=False)
    ldr = SudokuLoader(db_path='postgresql://chris:@/Sudoku.db', chunksize=1)

    target = deepcopy(q_net).to(device)
    
    p, s = ldr.next(device)
    def mini_reset() -> Tuple[torch.Tensor, torch.Tensor]:
        # indices = torch.randperm(n=p.size()[0])
        # return p[indices], s[indices]
        return p[0:1], s[0:1]
    
    stats = reinforce(env, q_net, target, episodes=256, f_reset=mini_reset, device=device, alpha=1e-5, gamma=0.99)
    plot_stats(stats)

# def training(policy: nn.Module, device = 'cpu'):
#     env = SudokuEnv()
#     ldr = SudokuLoader(db_path='postgresql://chris:@/Sudoku.db', chunksize=512)

#     def reset() -> Tuple[torch.Tensor, torch.Tensor]:
#         return ldr.next(device)
    
#     stats = reinforce(env, policy, episodes=1e6, f_reset=reset, device=device, alpha=1e-3)
#     plot_stats(stats)

# from Udemy code 
def plot_stats(stats):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        vals = [np.mean(vals[i-10:i+10]) for i in range(10, len(vals)-10)]
        if len(stats) > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.show()

def get_avs(q_net: nn.Module, state: torch.Tensor, epsilon: float = 0., device = 'cpu') -> torch.Tensor:
    q_net.eval()
    size = state.size(0)
    mask = (torch.rand(size=(size, 1)) < epsilon).broadcast_to((size, 729)).to(device)
    avs = ((-1024 - 1024) * torch.rand(size=(size, 729)) + 1024).to(device) * mask + q_net(state) * ~mask
    return avs
    
def av_to_actions(av: torch.Tensor, n: int = 3) -> torch.Tensor:
    index = torch.argmax(av, dim=1, keepdim=True) // (n**2)
    entry = torch.argmax(av, dim=1, keepdim=True) % (n**2) + 1
    return torch.cat([index, entry], dim = 1)

# def prob_to_actions(probs: torch.Tensor, n: int = 3) -> torch.Tensor:
#     action = probs.multinomial(1).detach().long()
#     index = action // (n**2)
#     entry = action % (n**2) + 1
#     return torch.cat([index, entry], dim = 1)

def reinforce(env: SudokuEnv, q_network: nn.Module, target_network: nn.Module, episodes: int,
              f_reset: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
              alpha: float = 1e-4, gamma: float = 0.99, epsilon: float = 0.2,
              device = 'cpu', filepath = 'model.pkl'):
    optim = AdamW(q_network.parameters(), lr=alpha)
    stats = {'Loss': [], 'Returns': []}
    target_network.eval()
    for i in tqdm(range(episodes)):
        state: torch.Tensor = env.reset(*f_reset()).to(device)
        done_b = torch.zeros((state.size()[0], 1), dtype=torch.bool)
        # transitions = []
        ep_return = torch.zeros((state.size()[0], 1))
        j = 0

        while not done_b.all() and j < 81:
            av = get_avs(q_network,  state.to(device, copy=True), epsilon, device)
            next_state, reward, done = env.step(av_to_actions(av), device)
            reward = ~done_b * reward.to('cpu')
            # transitions.append([state, av.to('cpu', copy=False), reward, next_state])

            action = torch.argmax(av, dim=1, keepdim=True)
            q_network.eval()
            qnet_qsa = q_network(state).gather(1, action)
            q_network.train()
            target_qsa = torch.max(target_network(next_state), dim=1, keepdim=True).values

            loss_t = F.mse_loss(qnet_qsa, (target_qsa * gamma + reward.to(device, copy=True)))

            q_network.zero_grad()
            loss_t.backward()
            optim.step()

            ep_return += reward
            done_b |= done.to('cpu')
            state = next_state
            j += 1

        # for t in transitions:
        #     state_t, av_t, reward_t, next_t = [i.to(device) for i in t]
        #     action_t = torch.argmax(av_t, dim=1, keepdim=True)
        #     q_network.eval()
        #     qnet_qsa = q_network(state_t).gather(1, action_t)
        #     q_network.train()
        #     target_qsa = torch.max(target_network(next_t), dim=1, keepdim=True).values

        #     loss_t = F.mse_loss(qnet_qsa, (target_qsa * gamma + reward_t))

        #     q_network.zero_grad()
        #     loss_t.backward()
        #     optim.step()
        
        if i % 10 == 0:
            sd = q_network.state_dict()
            target_network.load_state_dict(sd)
            target_network.eval()
            with open(filepath, 'wb') as f: # could multiprocess I/O
                pickle.dump(sd, f)

        stats['Loss'].append(loss_t.item())
        stats['Returns'].append(ep_return.mean().item())
    
    with open(filepath, 'wb') as f:
        pickle.dump(q_network.state_dict(), f)
    return stats

if __name__ == '__main__':
    main()
