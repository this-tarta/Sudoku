import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Queue

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
            'type': int,
            'help': 'number of CUDA devices to use; for input of 0 < n <= number of CUDA devices, uses devices [0, n), n <= 0 uses CPU',
            'default': 0
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
            'default': 64,
            'help': 'the size of the batch to load'
        }
    }
    args = parse_cmd(args, progname='ClassifierSolver.py', description='Solves Sudoku through classification machine learning methods')
    print(args)
    sudokus = SudokuDataset(db_path=args['dbpath'], table_name=args['tablename'], n=3, include_symmetries=args['symmetries'])
    trn, test = [0.001, 0.999]
    batch_size = args['batch_size']
    if args['CUDA'] > 0:
        n = min(args['CUDA'], torch.cuda.device_count())
        split = random_split(sudokus, [trn / n] * n + [test])
        loaders = [DataLoader(d, batch_size, shuffle=True) for d in split]
        processes = []
        mp.set_start_method('spawn')
        q = Queue()
        for rank in range(n):
            p = mp.Process(target=worker_train, args=(rank, n, args['epochs'], loaders[rank], args['learning_rate'], q))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        plot_stats(q.get())
    else:
        net = SudokuNetClassifier(512, 6)
        split = random_split(sudokus, [trn, test])
        loaders = [DataLoader(d, batch_size, shuffle=True) for d in split]
        stats = train(net, args['epochs'], loaders[0], alpha=args['learning_rate'])
        plot_stats(stats)

def models_equal(net1: nn.Module, net2: nn.Module) -> bool:
    params1 = net1.named_parameters()
    params2 = net2.named_parameters()
    for (name1, param1), (name2, param2) in zip(params1, params2):
        if name1 != name2 or not torch.equal(param1, param2):
            return False
    return True

def distributed_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(rank=rank, world_size=world_size)

def distributed_cleanup():
    dist.destroy_process_group()

def worker_train(rank: int, world_size: int, epochs: int, loader: DataLoader, alpha:float, q: Queue):
    distributed_setup(rank, world_size)
    net = SudokuNetClassifier(512, 6, 7).to(rank)
    ddp_model = DDP(net)

    q.put(train(ddp_model, epochs, loader, alpha=alpha, device=rank, filename=f'cuda:{rank}'))

    distributed_cleanup()

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

def train(model: DDP, epochs: int, trainloader: DataLoader, valloader: DataLoader = None,
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
            torch.save(model.module, filename + '_model.pkl')
            torch.save(optim, f=filename + '_adam.pkl')
            torch.save(stats, f=filename + '_stats.pkl')

    if filename is not None:
        torch.save(model.module, filename + '_model.pkl')
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
