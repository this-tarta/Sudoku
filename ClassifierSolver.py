import sys
import torch
from tqdm import tqdm
from torch import nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Callable, Tuple
from sqlalchemy import create_engine
from sqlalchemy import text

from utils import puzzle_from_string, nn_input, plot_stats


from DB_Management import SudokuLoader

class SeqConv(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: int, num_convs: int):
        super().__init__()
        l = [nn.Conv2d(in_channels=in_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(hidden_size), nn.ReLU()]
        for i in range(num_convs - 2):
            l += [nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
                  nn.BatchNorm2d(hidden_size),
                  nn.ReLU()
            ]
        l.append(nn.Conv2d(in_channels=hidden_size, out_channels=out_size, kernel_size=1, stride=1))
        self.seq = nn.Sequential(*l)

    def forward(self, x):
        return self.seq.forward(x)

class SudokuNet(nn.Module):
    def __init__(self, hidden_size: int, num_convs: int, n: int = 3):
        super().__init__()
        n2 = n * n
        self.net = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(1, n2, n2)),
            SeqConv(in_size=1, out_size=n2 + 1, hidden_size=hidden_size, num_convs=num_convs),
            nn.Flatten(start_dim=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net.forward(x)

def main():
    if len(sys.argv) < 2:
        epochs = 128
    else:
        epochs = int(sys.argv[1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SudokuNet(hidden_size=203, num_convs=11, n=3).to(device)
    training(net, epochs, device, 'classifier_model.pkl')


def hyperparameter_search(num_tests: int = 256):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    engine = create_engine('postgresql:///Sudoku')
    num_rows = 1024
    with engine.connect() as conn:
        res = conn.execute(text(f'SELECT puzzle, solution FROM "Sudoku" WHERE test=false LIMIT {num_rows};'))
        p, s = [], []
        for row in res:
            p.append(puzzle_from_string(row.puzzle))
            s.append(puzzle_from_string(row.solution))
        p = nn_input(torch.cat(p, dim=2)).to(device)
        s = nn_input(torch.cat(s, dim=2)).to(device)
    
    def mini_reset() -> Tuple[torch.Tensor, torch.Tensor]:
        return p, s

    best_loss = float('inf')
    best_err = 100
    best_hidden_size = -1
    best_num_convs = -1
    best_stats = {}
    for _ in tqdm(range(num_tests)):
        # best found are hs: 512 and num_c: 8 or 12
        # search obtained hidden size 203 and num_c 11
        hidden_size = torch.randint(low=32, high=1024, size=(1,)).item()
        num_convs = torch.randint(low=2, high=12, size=(1,)).item()
        try:
            net = SudokuNet(hidden_size=hidden_size, num_convs=num_convs, n=3).to(device)

            stats = train(net, 128, mini_reset, alpha=1e-3, device=device)
            loss = min(stats['Loss'])
            err = min(stats['Training Error'])
            if loss < best_loss or (loss == best_loss and hidden_size * num_convs < best_hidden_size * best_num_convs):
                best_loss = loss
                best_err = err
                best_hidden_size, best_num_convs = hidden_size, num_convs
                best_stats = stats
        except torch.OutOfMemoryError:
            pass
    plot_stats(best_stats)
    return { 'Loss': best_loss, 'Error': best_err, 'Hidden Size': best_hidden_size, 'Num Convs': best_num_convs }

def training(model: nn.Module, epochs: int, device = 'cpu', filepath = None):
    ldr = SudokuLoader(db_path='postgresql:///Sudoku', chunksize=4)

    def reset() -> Tuple[torch.Tensor, torch.Tensor]:
        return ldr.next(device)

    stats = train(model, epochs, reset, 1e-3, device, filepath)
    plot_stats(stats)

def train(model: nn.Module, epochs: int,
              f_reset: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
              alpha: float = 1e-4, device = 'cpu', filepath = None):
    optim = AdamW(params=model.parameters(), lr=alpha)
    stats = {'Loss': [], 'Training Error': []}
    model.train()
    for i in tqdm(range(epochs)):
        p, s = [t.to(device) for t in f_reset()]
        s_out = model(p.float())
        loss = F.cross_entropy(s_out, s.long())

        model.zero_grad()
        loss.backward()
        optim.step()

        stats['Loss'].append(loss.item())
        s_out = torch.argmax(s_out, dim=1, keepdim=False)
        stats['Training Error'].append(
            torch.count_nonzero(s - s_out).item() / torch.numel(s))
        
        if i % 10 == 0 and filepath is not None:
            torch.save(model, filepath)
            torch.save(optim, f='adam_' + filepath)
            torch.save(stats, f='stats_' + filepath)

    
    return stats

def plot_progress(filepath='stats_model.pkl'):
    while True:
        plot_stats(torch.load(filepath))

if __name__ == '__main__':
    main()
