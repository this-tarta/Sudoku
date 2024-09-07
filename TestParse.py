from utils import parse_cmd

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

    'throwaway': {
        'type': float
    }
}

args = parse_cmd(args, progname='TestParse.py', description='Tests the parse_cmd2 function')
print(args)