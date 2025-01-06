import torch
import pandas as pd
from utils import puzzle_from_string, nn_input

df = pd.read_csv('./Puzzles/sudoku-1m_for_GPT.csv')
puzzles = torch.cat([puzzle_from_string(i) for i in df['puzzle'].tolist()], dim=2)      # TODO: show how many clues are retained
actual = torch.cat([puzzle_from_string(i) for i in df['actual_solution'].tolist()], dim=2)
gpt = torch.cat([puzzle_from_string(i) for i in df['copilot_solution'].tolist()], dim=2)

mask = (actual == gpt)
cell_accuracy = torch.mean(mask.float()).item()
puzzle_accuracy = torch.mean(mask.all(dim=(0, 1)).float()).item()

print(f'Cell accuracy: {cell_accuracy}')
print(f'Puzzle accuracy: {puzzle_accuracy}')