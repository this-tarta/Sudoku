import os
import random
import torch as pt
import pandas as pd

from tqdm import tqdm
from sqlalchemy import String, Boolean
from sqlalchemy import create_engine, func
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.pool import QueuePool
from concurrent.futures import ProcessPoolExecutor

from sudoku_sat import solve_sudoku
from utils import puzzle_from_string
from utils import string_from_puzzle
from utils import nn_input
from utils import get_all_symmetries

class Base(DeclarativeBase):
    pass

class Sudoku(Base):
    __tablename__ = "Sudoku"
    id: Mapped[int] = mapped_column(primary_key=True)
    puzzle: Mapped[str] = mapped_column(String(81), unique=True)
    solution: Mapped[str] = mapped_column(String(81))
    test: Mapped[bool] = mapped_column(Boolean())

    def __repr__(self) -> str:
        return f"Sudoku(id={self.id!r}, puzzle={self.puzzle!r}, solution={self.solution!r}), test={self.test!r}"
    
def multi(chunk: pd.DataFrame, db_path: str):
    engine = create_engine(db_path, isolation_level = 'READ UNCOMMITTED',
                       poolclass=QueuePool, pool_size = 32, pool_timeout = 10)
    for puzzle in tqdm(chunk.loc[:,'puzzle']):
        puzzle_np = puzzle_from_string(puzzle)
        result = solve_sudoku(puzzle_np)
        if result[1] == 1:
            with Session(engine) as sess:
                sess.add(Sudoku(puzzle=puzzle, solution=string_from_puzzle(result[0]),
                                test=(random.random() < 0.2)))
                sess.commit()
                sess.close()

def csv_to_db(src_path: str, db_path: str, chunksize: int = 1e5, num_workers: int = os.cpu_count()):
    ''' Takes a filepath of form .csv with sudokus, verifies the validity of the puzzle, then adds
        it to the db given by engine
    '''
    
    with ProcessPoolExecutor(max_workers=num_workers) as exec:
        for chunk in pd.read_csv(src_path, chunksize=chunksize):
            exec.submit(multi, chunk, db_path)

# csv_to_db('./Puzzles/sudoku-3m.csv', 'postgresql://chris:@/Sudoku.db')
# chunksize = 7500 => 1600 chunks

class SudokuDataset(pt.utils.data.Dataset):
    __num_syms__ = 288

    def __init__(self, db_path: str, table_name: str, n: int = 3, include_symmetries: bool = False):
        super().__init__()
        self.db = db_path
        self.n = n
        self.include_symmetries = include_symmetries
        self.table_name = table_name
        self.len = pd.read_sql_query(f'SELECT COUNT(*) as count FROM "{table_name}";', con=db_path)['count'].iloc[0]
            
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx: int):
        df = pd.read_sql_query(f'SELECT puzzle as p, solution as s FROM "{self.table_name}"
                                    WHERE id={idx + 1};', con=self.db)  # since idx is 0-indexed, but db is 1-indexed
        p, s = puzzle_from_string(df['p'].iloc[0]), puzzle_from_string(df['s'].iloc[0])
        if self.include_symmetries:
            randsym = pt.randint(0, self.__num_syms__, size=(1,)).item()
            p, s = get_all_symmetries(p, s)
            p = nn_input(p)[randsym]
            s = nn_input(s)[randsym]
        else:
            p = nn_input(p)[0]
            s = nn_input(s)[0]
        
        return p, s