import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Callable, Tuple


from DB_Management import SudokuLoader

torch.autograd.set_detect_anomaly(True)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q_net = nn.Sequential(
        nn.Unflatten(dim=1, unflattened_size=(1, 9, 9)),
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=1),
        nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
        nn.Flatten(),
        nn.Linear(490, 4096),
        nn.ReLU(),
        nn.Unflatten(dim=1, unflattened_size=(4, 32, 32)),
        nn.Conv2d(in_channels=4, out_channels=9, kernel_size=4, stride=2),
        nn.MaxPool2d(kernel_size=2, stride=1),
        nn.Flatten(),
        nn.Linear(1764, 2048),
        nn.ReLU(),
        nn.Linear(2048, 729),
        nn.Unflatten(dim=1, unflattened_size=(81, 9)),
        nn.Softmax(dim=-1)
    ).to(device)

    # with open('model.pkl', 'rb') as f:
    #     policy.load_state_dict(pickle.load(f))
    
    pre_training(q_net, device)


def pre_training(q_net: nn.Module, device = 'cpu'):
    ldr = SudokuLoader(db_path='postgresql://chris:@/Sudoku.db', chunksize=256)

    p, s = ldr.next(device)
    def mini_reset() -> Tuple[torch.Tensor, torch.Tensor]:
        # indices = torch.randperm(n=p.size(0))
        # return p[indices], s[indices]
        return p, s
    
    stats = train(q_net, 8192, mini_reset, alpha=1e-5, device=device, filepath='new_model.pkl')
    plot_stats(stats)
    
    q_net.eval()

    s_pred = torch.argmax(q_net(p[3:4].float()), dim=2) + 1
    print(s_pred)
    print(s[3])

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

def train(model: nn.Module, epochs: int,
              f_reset: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
              alpha: float = 1e-4, device = 'cpu', filepath = 'model.pkl'):
    optim = AdamW(params=model.parameters(), lr=alpha)
    stats = {'Loss': [], 'Training Error': []}
    for i in tqdm(range(epochs)):
        p, s = [t.to(device) for t in f_reset()]
        s_out = model(p.float())
        loss = F.cross_entropy(s_out, F.one_hot(s - 1, num_classes=9).float())

        model.zero_grad()
        loss.backward()
        optim.step()

        stats['Loss'].append(loss.item())
        s_out = 1 + torch.argmax(s_out, dim=2, keepdim=False)
        stats['Training Error'].append(
            torch.count_nonzero(s - s_out).item() / torch.numel(s))
        
        if i % 100 == 0:
            torch.save(model, filepath)
            torch.save(optim, f='adam_' + filepath)
            torch.save(stats, f='stats_' + filepath)

    
    return stats



if __name__ == '__main__':
    main()
