import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from utils import plot_stats, parse_cmd
from DB_Management import SudokuDataset
from SudokuNet import SudokuNetClassifier
from SudokuEnv import SudokuEnv, ReplayMemory
from ReinforcementSolver import classifier_actions



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
            'default': 'postgresql:///Sudoku.db'
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
    trn, val, test = random_split(sudokus, [0.0001, 0.00005, 0.99985])
    batch_size = args['batch_size']
    trainloader = DataLoader(trn, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val, batch_size=batch_size, shuffle=False)
    net = SudokuNetClassifier(hidden_size=512, num_convs=6, kernel_size=7, n=3).to(device)
    # stats = train(net, args['epochs'], trainloader, valloader, args['learning_rate'], device) # filename='./Models/classifier')
    stats = train_with_env(SudokuEnv(1, -1, 1, 0), net, args['epochs'], trainloader, alpha=args['learning_rate'], device=device)
    plot_stats(stats)

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
            torch.save(model, filename + '_model.pkl')
            torch.save(optim, f=filename + '_adam.pkl')
            torch.save(stats, f=filename + '_stats.pkl')

    if filename is not None:
        torch.save(model, filename + '_model.pkl')
        torch.save(optim, f=filename + '_adam.pkl')
        torch.save(stats, f=filename + '_stats.pkl')

    return stats

def train_with_env(env: SudokuEnv, model: nn.Module, epochs: int, trainloader: DataLoader, memory_capacity: int = int(1e7),
              alpha: float = 1e-4, device: torch.DeviceObjType = 'cpu', filename: str | None = None) -> dict[str, list[float]]:
    model = model.to(device)
    optim = AdamW(params=model.parameters(), lr=alpha)
    stats = {'Training Loss': [], 'Training Error': [], 'Training Returns': []}
    memory = ReplayMemory(capacity=memory_capacity)
    model.train()
    for i in tqdm(range(epochs)):
        for p, s in trainloader:
            state: torch.Tensor = env.reset(p, s).to(device)
            s = s.to(device)
            s_out = model(state)
            loss = F.cross_entropy(s_out, s.long())
            model.zero_grad()
            loss.backward()
            optim.step()

            done_b = torch.zeros(size=(p.size(0),), dtype=bool)
            reward_b = torch.zeros(size=(p.size(0),1))
            while not done_b.all():
                next_state, reward, done = env.step(classifier_actions(model, state), device)
                reward_b += reward.to('cpu')
                done_b |= done.squeeze().to('cpu')
                size = (~done_b).sum().item()
                memory.insert([next_state[~done_b].to('cpu'),
                               torch.ByteTensor(size=(size, 1)),
                               torch.ByteTensor(size=(size, 1)),
                               torch.ByteTensor(size=(size, 1)),
                               s[~done_b].to('cpu')])
                state = next_state.to(device)

            if memory.can_sample(trainloader.batch_size):
                p_mem, _, _, _, s_mem = memory.sample(trainloader.batch_size)
                p_mem = p_mem.float().to(device)
                s_out_mem = model(p_mem)
                loss = F.cross_entropy(s_out_mem, s_mem.long().to(device))
                model.zero_grad()
                loss.backward()
                optim.step()

        stats['Training Loss'].append(loss.item())
        s_out = torch.argmax(s_out, dim=1, keepdim=False)
        stats['Training Error'].append(
            torch.count_nonzero(s - s_out).item() / torch.numel(s))
        stats['Training Returns'].append(reward_b.mean().item())

        if i % 10 == 0 and filename is not None:
            torch.save(model, filename + '_model.pkl')
            torch.save(optim, f=filename + '_adam.pkl')
            torch.save(stats, f=filename + '_stats.pkl')

    if filename is not None:
        torch.save(model, filename + '_model.pkl')
        torch.save(optim, f=filename + '_adam.pkl')
        torch.save(stats, f=filename + '_stats.pkl')

    return stats

# def plot_progress(filepath='stats_model.pkl'):
#     while True:
#         plot_stats(torch.load(filepath))

if __name__ == '__main__':
    main()
