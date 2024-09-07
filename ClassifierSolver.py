import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from utils import plot_stats, parse_cmd
from DB_Management import SudokuDataset
from SudokuNet import SudokuNetClassifier



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
        }
    }
    args = parse_cmd(args, progname='ClassifierSolver.py', description='Solves Sudoku through classification machine learning methods')
    print(args)
    sudokus = SudokuDataset(db_path=args['dbpath'], table_name=args['tablename'], n=3, include_symmetries=args['symmetries'])
    device = 'cuda' if torch.cuda.is_available() and args['CUDA'] else 'cpu'
    trn, val, test = random_split(sudokus, [0.005, 0.001, 0.994])
    batch_size = args['batch_size']
    trainloader = DataLoader(trn, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val, batch_size=batch_size, shuffle=False)
    net = SudokuNetClassifier(hidden_size=512, num_convs=16, n=3).to(device)
    stats = train(net, args['epochs'], trainloader, valloader, args['learning_rate'], device)
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
            net = SudokuNetClassifier(hidden_size=hidden_size, num_convs=num_convs, n=3).to(device)

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
              alpha: float = 1e-4, device: torch.DeviceObjType = 'cpu', filename: str | None = None) -> dict[str, list[float]]:
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

    if filename is not None:
        torch.save(model, filename + '.pkl')
        torch.save(optim, f=filename + '_adam.pkl')
        torch.save(stats, f=filename + '_stats.pkl')

    return stats

# def plot_progress(filepath='stats_model.pkl'):
#     while True:
#         plot_stats(torch.load(filepath))

if __name__ == '__main__':
    main()
