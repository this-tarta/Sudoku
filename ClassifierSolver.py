import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
from typing import Callable, Tuple
from sqlalchemy import create_engine
from sqlalchemy import text

from utils import puzzle_from_string, nn_input, plot_stats
from DB_Management import SudokuDataset

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

def parse_cmd():
    epochs = 128
    db = 'postgresql:///Sudoku'
    name = 'Sudoku'
    syms = False
    cuda = False

    parser = argparse.ArgumentParser(prog='ClassifierSolver.py',
                                     description='Solves Sudoku through classification machine learning methods')
    parser.add_argument('-e', '--epochs', help=f'the number of epochs you wish to run (default {epochs})', type=int)
    parser.add_argument('-D', '--dbpath', help=f'the path to the database (default "{db}")')
    parser.add_argument('-t', '--tablename', help=f'the name of the target table (default "{name}")')
    parser.add_argument('-s', '--symmetries', help='enables use of symmetries', action='store_true')
    parser.add_argument('-c', '--CUDA', help='enables use of CUDA if available', action='store_true')
    args = parser.parse_args()
    if args.epochs:
        epochs = args.epochs    
    if args.dbpath:
        db = args.dbpath
    if args.tablename:
        name = args.tablename
    if args.symmetries:
        syms = True
    if args.CUDA:
        cuda = True
    
    return [epochs, db, name, syms, cuda]

def main():
    epochs, db, name, syms, cuda = parse_cmd()
    print(f'Num epochs: {epochs}')
    print(f'DB: {db}')   
    print(f'Table name: {name}')
    print(f'Include symmetries: {syms}')
    print(f'Use CUDA: {cuda}')
    sudokus = SudokuDataset(db_path=db, table_name=name, n=3, include_symmetries=syms)
    device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'
    trn, val, test = random_split(sudokus, [0.05, 0.01, 0.94])
    trainloader = DataLoader(trn, batch_size=256, shuffle=True)
    valloader = DataLoader(val, batch_size=256, shuffle=False)
    net = SudokuNet(hidden_size=512, num_convs=16, n=3).to(device)
    stats = train(net, epochs, trainloader, valloader, 1e-3, device)
    plot_stats(stats)
    for s in stats.keys():
        print(f'{s}: {min(stats[s])}')
    # print(hyperparameter_search(trainloader, valloader, epochs=epochs, num_tests=10, alpha=1e-3, device=device))

def hyperparameter_search(trainloader: DataLoader, valloader: DataLoader, epochs: int = 32,
                          num_tests: int = 256, alpha: float = 1e-4, device: torch.DeviceObjType = 'cpu') -> dict[str, float]:
    best_loss = float('inf')
    best_err = 100
    best_hidden_size = -1
    best_num_convs = -1
    best_stats = {}
    for _ in tqdm(range(num_tests)):
        # best found are hs: 512 and num_c: 8 or 12
        # search obtained hidden size 512 and num_c 11
        hidden_size = torch.randint(low=32, high=1024, size=(1,)).item()
        num_convs = torch.randint(low=2, high=12, size=(1,)).item()
        try:
            net = SudokuNet(hidden_size=hidden_size, num_convs=num_convs, n=3).to(device)

            stats = train(net, epochs, trainloader, valloader, alpha, device)
            loss = min(stats['Validation Loss'])
            err = min(stats['Validation Error'])
            if loss < best_loss or (loss == best_loss and hidden_size * num_convs < best_hidden_size * best_num_convs):
                best_loss = loss
                best_err = err
                best_hidden_size, best_num_convs = hidden_size, num_convs
                best_stats = stats
        except torch.OutOfMemoryError:
            pass
    plot_stats(best_stats)
    return { 'Validation Loss': best_loss, 'Validation Error': best_err, 'Hidden Size': best_hidden_size, 'Num Convs': best_num_convs }

def train(model: nn.Module, epochs: int, trainloader: DataLoader, valloader: DataLoader = None,
              alpha: float = 1e-4, device: torch.DeviceObjType = 'cpu', filename: str | None = None) -> dict[str, list]:
    model = model.to(device)
    optim = AdamW(params=model.parameters(), lr=alpha)
    stats = {'Training Loss': [], 'Training Error': []}
    if valloader is not None:
        stats['Validation Loss'] = []
        stats['Validation Error'] = []

    for i in tqdm(range(epochs)):
        model.train()
        for p, s in trainloader:
            p = p.to(device)
            s = s.to(device)
            s_out = model(p.float())
            loss = F.cross_entropy(s_out, s.long())

            model.zero_grad()
            loss.backward()
            optim.step()

        stats['Training Loss'].append(loss.item())
        s_out = torch.argmax(s_out, dim=1, keepdim=False)
        stats['Training Error'].append(
            torch.count_nonzero(s - s_out).item() / torch.numel(s))
        
        if valloader is not None:
            model.eval()
            num_batch = len(valloader)
            total_loss = 0
            total_error = 0
            for p, s in valloader:
                p = p.to(device)
                s = s.to(device)
                s_out = model(p.float())
                total_loss += F.cross_entropy(s_out, s.long()).item()
                s_out = torch.argmax(s_out, dim=1, keepdim=False)
                total_error += torch.count_nonzero(s - s_out).item() / torch.numel(s)
            
            stats['Validation Loss'].append(total_loss / num_batch)
            stats['Validation Error'].append(total_error / num_batch)
        
        if i % 10 == 0 and filename is not None:
            torch.save(model, filename + '.pkl')
            torch.save(optim, f=filename + '_adam.pkl')
            torch.save(stats, f=filename + '_stats.pkl')

    return stats

# def plot_progress(filepath='stats_model.pkl'):
#     while True:
#         plot_stats(torch.load(filepath))

if __name__ == '__main__':
    main()
