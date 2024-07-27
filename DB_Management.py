import os
import random
import pandas as pd

from tqdm import tqdm
from sqlalchemy import String, Boolean
from sqlalchemy import create_engine
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.pool import QueuePool
from concurrent.futures import ProcessPoolExecutor

from sudoku_sat import solve_sudoku
from utils import puzzle_from_string
from utils import string_from_puzzle

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

# chunksize = 3
# df = pd.read_sql_query('SELECT * FROM "Sudoku" WHERE test=false;',
#                        con='postgresql://chris:@/Sudoku.db',
#                        coerce_float=False, chunksize=chunksize)
# chunk = next(df)
# puzzles = chunk['puzzle'].tolist()
# solutions = chunk['solution'].tolist()
# puzzles = torch.cat([puzzle_from_string(p) for p in puzzles], dim = 2)#.to('cuda')
# solutions = torch.cat([puzzle_from_string(p) for p in solutions], dim = 2)#.to('cuda')
# # syms = get_all_symmetries(puzzles, solutions)
# print(nn_input(puzzles))

# p = puzzle_from_string('900000002010060390083900100804095007130670049060041000302010050000500000541080030')
# s = puzzle_from_string('956134782417268395283957164824395617135672849769841523372416958698523471541789236')
# solve_sudoku(torch.zeros((9, 9, 1)))
# string = '900000002010060390083900100804095007130670049060041000302010050000500000541080030'
# assert string == string_from_puzzle(puzzle_from_string(string))

# assert torch.equal(solve_sudoku(p)[0], s)
# assert solve_sudoku(torch.zeros((9, 9, 1)))[1] >= 2