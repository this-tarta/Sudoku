from torch import nn as nn

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
        )

    def forward(self, x):
        return self.net.forward(x)
    
class SudokuNetClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_convs: int, n: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            SudokuNet(hidden_size, num_convs, n),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net.forward(x)