import torch
import unittest as tst

from utils import puzzle_from_string
from utils import string_from_puzzle
from utils import get_geometric_symmetries
from utils import get_nongeometric_symmetries
from utils import get_all_symmetries
from utils import nn_input
from sudoku_sat import solve_sudoku
from SudokuEnv import SudokuEnv
from DB_Management import SudokuLoader

class TestUtils(tst.TestCase):
    def test_puzzle_from_string(self):
        p1 = puzzle_from_string('840907600000500070000240003309471000060000100100005098000050006007804010650792830')
        p1_ = [
            [[8], [4], [0], [9], [0], [7], [6], [0], [0]],
            [[0], [0], [0], [5], [0], [0], [0], [7], [0]],
            [[0], [0], [0], [2], [4], [0], [0], [0], [3]],
            [[3], [0], [9], [4], [7], [1], [0], [0], [0]],
            [[0], [6], [0], [0], [0], [0], [1], [0], [0]],
            [[1], [0], [0], [0], [0], [5], [0], [9], [8]],
            [[0], [0], [0], [0], [5], [0], [0], [0], [6]],
            [[0], [0], [7], [8], [0], [4], [0], [1], [0]],
            [[6], [5], [0], [7], [9], [2], [8], [3], [0]]
        ]
        self.assertTrue(torch.equal(p1, torch.tensor(p1_)))

        p2 = puzzle_from_string('004081200901000703050007090206000000409275000035000004000806000007190350000000610')
        p2_ = [
            [[0], [0], [4], [0], [8], [1], [2], [0], [0]],
            [[9], [0], [1], [0], [0], [0], [7], [0], [3]],
            [[0], [5], [0], [0], [0], [7], [0], [9], [0]],
            [[2], [0], [6], [0], [0], [0], [0], [0], [0]],
            [[4], [0], [9], [2], [7], [5], [0], [0], [0]],
            [[0], [3], [5], [0], [0], [0], [0], [0], [4]],
            [[0], [0], [0], [8], [0], [6], [0], [0], [0]],
            [[0], [0], [7], [1], [9], [0], [3], [5], [0]],
            [[0], [0], [0], [0], [0], [0], [6], [1], [0]]
        ]
        self.assertTrue(torch.equal(p2, torch.tensor(p2_)))

        p3 = puzzle_from_string('..1..4..7...3..8...8....5..3..15...9.7.....2..5..3........2...6862..3.7.....8....')
        p3_ = [
            [[0], [0], [1], [0], [0], [4], [0], [0], [7]],
            [[0], [0], [0], [3], [0], [0], [8], [0], [0]],
            [[0], [8], [0], [0], [0], [0], [5], [0], [0]],
            [[3], [0], [0], [1], [5], [0], [0], [0], [9]],
            [[0], [7], [0], [0], [0], [0], [0], [2], [0]],
            [[0], [5], [0], [0], [3], [0], [0], [0], [0]],
            [[0], [0], [0], [0], [2], [0], [0], [0], [6]],
            [[8], [6], [2], [0], [0], [3], [0], [7], [0]],
            [[0], [0], [0], [0], [8], [0], [0], [0], [0]]
        ]
        self.assertTrue(torch.equal(p3, torch.tensor(p3_)))

        p4 = puzzle_from_string('...3....8..3.9.5..47...1.2........93.1.6.7....8.5......4.7...6.56...4...........7')
        p4_ = [
            [[0], [0], [0], [3], [0], [0], [0], [0], [8]],
            [[0], [0], [3], [0], [9], [0], [5], [0], [0]],
            [[4], [7], [0], [0], [0], [1], [0], [2], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [9], [3]],
            [[0], [1], [0], [6], [0], [7], [0], [0], [0]],
            [[0], [8], [0], [5], [0], [0], [0], [0], [0]],
            [[0], [4], [0], [7], [0], [0], [0], [6], [0]],
            [[5], [6], [0], [0], [0], [4], [0], [0], [0]],
            [[0], [0], [0], [0], [0], [0], [0], [0], [7]]
        ]
        self.assertTrue(torch.equal(p4, torch.tensor(p4_)))
    
    def test_string_from_puzzle(self):
        self.assertEqual(string_from_puzzle(puzzle_from_string('400005907007000350030090006500083640200001030006400020002076190000020460063140570')),
                         '400005907007000350030090006500083640200001030006400020002076190000020460063140570')
        
        self.assertEqual(string_from_puzzle(puzzle_from_string('630004701108390050000700390490001500562438000803009040000103270321900800740280139')),
                         '630004701108390050000700390490001500562438000803009040000103270321900800740280139')
        
        self.assertEqual(string_from_puzzle(puzzle_from_string('....45..78.7...3...3.8..94..5.9...7.19..6.2.3..24....1...78.........4.6.9........')),
                         '000045007807000300030800940050900070190060203002400001000780000000004060900000000')
        
        self.assertEqual(string_from_puzzle(puzzle_from_string('7....6.....9.4...62.1.........9..71.......25.4..5....9....681..9...2.6.......1..3')),
                         '700006000009040006201000000000900710000000250400500009000068100900020600000001003')
        
    def test_get_geometric_symmetries(self):
        p1 = puzzle_from_string('3.7........4...2.....9.3.....2.8..1.1..297..6...1..7....68.4....5..7..81....2..7.')
        s1 = puzzle_from_string('387562194964718253521943867672485319138297546495136728716854932253679481849321675')
        self.check_syms(*get_geometric_symmetries(p1, s1), 8)

        p2 = puzzle_from_string('120000408009005173730480629800004962600850040003006805594062700200900004300540096')
        s2 = puzzle_from_string('126793458489625173735481629857134962612859347943276815594362781268917534371548296')
        self.check_syms(*get_geometric_symmetries(p2, s2), 8)

        p3 = [
            '6.7....5.5..6....4..4.51..3.....7..5..54.3..7.93.15.4.......29.8.............4...',
            '008265070010800452025714986009170634034098000270006895000453209500021008060007500',
            '200095460060704000498100030700389040003200800050040720004000986986073152001008004'
        ]
        s3 = [
            '637248159518639724924751863482967315165423987793815642341576298856392471279184536',
            '948265371716839452325714986859172634634598127271346895187453269593621748462987513',
            '237895461165734298498126537712389645643257819859641723374512986986473152521968374'
        ]
        p3 = torch.cat([puzzle_from_string(p) for p in p3], dim = 2)
        s3 = torch.cat([puzzle_from_string(s) for s in s3], dim = 2)
        self.check_syms(*get_geometric_symmetries(p3, s3), 24)

    def test_get_nongeometric_symmetries(self):
        p1 = puzzle_from_string('560089100030000000180306092940137806013028000072400309006054001728603004000090683')
        s1 = puzzle_from_string('564289137239571468187346592945137826613928745872465319396854271728613954451792683')
        self.check_syms(*get_nongeometric_symmetries(p1, s1), 36)

        p2 = puzzle_from_string('7..........9.1.4...4..85.93...5....89.....3.43.8.2.....7.23.6.5...1...2.....6....')
        s2 = puzzle_from_string('783942516569317482142685793417593268926871354358426179871239645634158927295764831')
        self.check_syms(*get_nongeometric_symmetries(p2, s2), 36)

        p3 = [
            '120000408009005173730480629800004962600850040003006805594062700200900004300540096',
            '400005098960200030108309000006920783700008010820010900009000071370090806000400350',
            '200600001780009003000234785496070010070306040020001506007000200502098037831052004'
        ]
        s3 = [
            '126793458489625173735481629857134962612859347943276815594362781268917534371548296',
            '432765198967281435158349627516924783794638512823517964649853271375192846281476359',
            '253687491784519623619234785496875312175326849328941576947163258562498137831752964'
        ]
        p3 = torch.cat([puzzle_from_string(p) for p in p3], dim = 2)
        s3 = torch.cat([puzzle_from_string(s) for s in s3], dim = 2)
        self.check_syms(*get_nongeometric_symmetries(p3, s3), 108)
    
    def test_get_all_symmetries(self):
        p1 = puzzle_from_string('3.54.....6...1.....7..9.4.37..1..96.5...237....9.....1..8.....7.2.6...1.......3..')
        s1 = puzzle_from_string('395478126684312579271596483732184965516923748849765231458231697923657814167849352')
        self.check_syms(*get_all_symmetries(p1, s1), 288)

        p2 = puzzle_from_string('080600020001000400063020500702008000000050080000002697609003045450810002000400176')
        s2 = puzzle_from_string('584631729271589463963724518742968351196357284835142697619273845457816932328495176')
        self.check_syms(*get_all_symmetries(p2, s2), 288)

        p3 = [
            '.6..5879......15.....7..1.32.39...5.8.............42.8.........581..6.3...9.15...',
            '092700001004000270785000943273000006850001309409300080508020004021643008040508000',
            '598230614000800009007945020076080000004793560105004907250478090009100870781360245'
        ]
        s3 = [
            '162358794397641582458729163243987651875162349916534278624873915581296437739415826',
            '392754861164839275785216943273985416856471329419362587538127694921643758647598132',
            '598237614342816759617945328976581432824793561135624987253478196469152873781369245'
        ]
        p3 = torch.cat([puzzle_from_string(p) for p in p3], dim = 2)
        s3 = torch.cat([puzzle_from_string(s) for s in s3], dim = 2)
        self.check_syms(*get_all_symmetries(p3, s3), 864)
    
    def check_syms(self, syms_p, syms_s, expected_num):
        self.assertEqual(len(syms_p[0, 0]), expected_num)
        self.assertEqual(len(syms_s[0, 0]), expected_num)

        for i in range(expected_num):
            for j in range(i + 1, expected_num):
                self.assertFalse(torch.equal(syms_p[:, :, i], syms_p[:, :, j]))
                self.assertFalse(torch.equal(syms_s[:, :, i], syms_s[:, :, j]))
            s1_, c1 = solve_sudoku(syms_p[:, :, i:i+1])
            self.assertEqual(c1, 1)
            self.assertTrue(torch.equal(syms_s[:, :, i:i+1], s1_))

    def test_nn_input(self):
        p1 = [
            '.6..5879......15.....7..1.32.39...5.8.............42.8.........581..6.3...9.15...',
            '092700001004000270785000943273000006850001309409300080508020004021643008040508000',
            '598230614000800009007945020076080000004793560105004907250478090009100870781360245'
        ]
        p1 = torch.cat([puzzle_from_string(p) for p in p1], dim = 2)
        p1 = nn_input(p1)
        p1_exp = torch.tensor([
            [0, 6, 0, 0, 5, 8, 7, 9, 0,
             0, 0, 0, 0, 0, 1, 5, 0, 0,
             0, 0, 0, 7, 0, 0, 1, 0, 3,
             2, 0, 3, 9, 0, 0, 0, 5, 0,
             8, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 4, 2, 0, 8,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             5, 8, 1, 0, 0, 6, 0, 3, 0,
             0, 0, 9, 0, 1, 5, 0, 0, 0],

            [0, 9, 2, 7, 0, 0, 0, 0, 1,
             0, 0, 4, 0, 0, 0, 2, 7, 0, 
             7, 8, 5, 0, 0, 0, 9, 4, 3,
             2, 7, 3, 0, 0, 0, 0, 0, 6,
             8, 5, 0, 0, 0, 1, 3, 0, 9,
             4, 0, 9, 3, 0, 0, 0, 8, 0,
             5, 0, 8, 0, 2, 0, 0, 0, 4,
             0, 2, 1, 6, 4, 3, 0, 0, 8,
             0, 4, 0, 5, 0, 8, 0, 0, 0],

            [5, 9, 8, 2, 3, 0, 6, 1, 4,
             0, 0, 0, 8, 0, 0, 0, 0, 9,
             0, 0, 7, 9, 4, 5, 0, 2, 0,
             0, 7, 6, 0, 8, 0, 0, 0, 0,
             0, 0, 4, 7, 9, 3, 5, 6, 0,
             1, 0, 5, 0, 0, 4, 9, 0, 7,
             2, 5, 0, 4, 7, 8, 0, 9, 0,
             0, 0, 9, 1, 0, 0, 8, 7, 0, 
             7, 8, 1, 3, 6, 0, 2, 4, 5]
        ])
        self.assertTrue(torch.equal(p1, p1_exp))

        p2 = puzzle_from_string('601000907040810000002006100004370200537092408000654039200003014800020675000581090')
        p2 = nn_input(p2)
        p2_exp = torch.tensor([
            [6, 0, 1, 0, 0, 0, 9, 0, 7,
             0, 4, 0, 8, 1, 0, 0, 0, 0,
             0, 0, 2, 0, 0, 6, 1, 0, 0,
             0, 0, 4, 3, 7, 0, 2, 0, 0,
             5, 3, 7, 0, 9, 2, 4, 0, 8,
             0, 0, 0, 6, 5, 4, 0, 3, 9,
             2, 0, 0, 0, 0, 3, 0, 1, 4,
             8, 0, 0, 0, 2, 0, 6, 7, 5,
             0, 0, 0, 5, 8, 1, 0, 9, 0]
        ])
        self.assertTrue(torch.equal(p2, p2_exp))

        p3 = puzzle_from_string('632945817489173625517682943978431562354296178126758439245369781793814256861527394')
        p3 = nn_input(get_all_symmetries(p3, p3)[0])
        self.assertEqual(p3.size(), torch.Size((288, 81)))

class TestSudokuSAT(tst.TestCase):
    def test_sat_one_solution(self):
        p1 = puzzle_from_string('000000010600000503200093407015900200360080050007301940538260100000000870172000065')
        s1 = puzzle_from_string('783524619694718523251693487415976238369482751827351946538267194946135872172849365')
        s1_, c1 = solve_sudoku(p1)
        self.assertTrue(torch.equal(s1, s1_))
        self.assertEqual(c1, 1)

        p2 = puzzle_from_string('3.......2...684.1.9....5....5....4....3...8.....51...6..839.....37....8.6...2..7.')
        s2 = puzzle_from_string('386179542725684913941235768852763491163942857479518236518397624237456189694821375')
        s2_, c2 = solve_sudoku(p2)
        self.assertTrue(torch.equal(s2, s2_))
        self.assertEqual(c2, 1)
    
    def test_sat_no_solution(self):
        p1 = puzzle_from_string('023950060005601238016032000408000357100587094657009082000190073589700415301408025')
        _, c1 = solve_sudoku(p1)
        self.assertEqual(c1, 0)

        p2 = puzzle_from_string('....67.....5.4.8..8..3...7...7..........2.39...4..65..3.1.5...2...69....9.1.71...')
        _, c2 = solve_sudoku(p2)
        self.assertEqual(c2, 0)

        p3 = puzzle_from_string('120004905409005613020000042702040590040002000030610074204030000810050000000200000')
        _, c3 = solve_sudoku(p3)
        self.assertEqual(c3, 0)
    
    def test_sat_many_solutions(self):
        p1 = puzzle_from_string('000000000000000000000000000000000000000000000000000000000000000000000000000000000')
        _, c1 = solve_sudoku(p1, max_sols=23)
        self.assertEqual(c1, 23)

        p2 = puzzle_from_string('9.6.7.4.3...4..2...7..23.1.5.....1...4.2.8.6...3.....5.3.7...5...7..5...4.5.1.7.8')
        s2, c2 = solve_sudoku(p2, max_sols=5)
        sols = [
            puzzle_from_string('926571483351486279874923516582367194149258367763149825238794651617835942495612738'),
            puzzle_from_string('926571483351486279874923516582367194149258367763194825238749651617835942495612738')
        ]
        self.assertEqual(c2, 2)
        self.assertTrue(torch.equal(s2, sols[0]) or torch.equal(s2, sols[1]))

class TestSudokuEnv(tst.TestCase):
    def test_env_good_step(self):
        puzzle = nn_input(puzzle_from_string('301086504046521070500000001400800002080347900009050038004090200008734090007208103'))
        soln = nn_input(puzzle_from_string('371986524846521379592473861463819752285347916719652438634195287128734695957268143'))
        env = SudokuEnv()
        env.reset(puzzle, soln)

        # First step
        next_state, reward, done = env.step(torch.tensor([[1, 7]]))
        self.assertTrue(torch.equal(next_state,
                                    nn_input(puzzle_from_string('371086504046521070500000001400800002080347900009050038004090200008734090007208103'))))
        self.assertTrue(torch.equal(reward, torch.tensor([[16]])))
        self.assertFalse(done.item())

        # Second step
        next_state, reward, done = env.step(torch.tensor([[3, 9]]))
        self.assertTrue(torch.equal(next_state,
                                    nn_input(puzzle_from_string('371986504046521070500000001400800002080347900009050038004090200008734090007208103'))))
        self.assertTrue(torch.equal(reward, torch.tensor([[16]])))
        self.assertFalse(done.item())

    def test_env_bad_step(self):
        puzzle = nn_input(puzzle_from_string('...38...4.........13....5....6.....8.9.2.8.7......4..1..3..6.856..9.5.....7.3..1.'))
        soln = nn_input(puzzle_from_string('259387164764521893138469527416793258395218476872654931923146785681975342547832619'))
        env = SudokuEnv()
        env.reset(puzzle, soln)

        # Invalid move
        next_state, reward, done = env.step(torch.tensor([[8, 2]]))
        self.assertTrue(torch.equal(next_state,
                                    nn_input(puzzle_from_string('...38...4.........13....5....6.....8.9.2.8.7......4..1..3..6.856..9.5.....7.3..1.'))))
        self.assertTrue(torch.equal(reward, torch.tensor([[0]])))
        self.assertFalse(done.item())

        # Wrong move
        next_state, reward, done = env.step(torch.tensor([[75, 4]]))
        self.assertTrue(torch.equal(next_state,
                                    nn_input(puzzle_from_string('...38...4.........13....5....6.....8.9.2.8.7......4..1..3..6.856..9.5.....743..1.'))))
        self.assertTrue(torch.equal(reward, torch.tensor([[-1024]])))
        self.assertTrue(done.item())

    def test_env_end_step(self):
        # Puzzle 1
        puzzle = nn_input(puzzle_from_string('467529813031478526258316497685947231372165948149283765514832679793654182826791354'))
        soln = nn_input(puzzle_from_string('467529813931478526258316497685947231372165948149283765514832679793654182826791354'))
        env = SudokuEnv()
        env.reset(puzzle, soln)

        next_state, reward, done = env.step(torch.tensor([[9, 9]]))
        self.assertTrue(torch.equal(next_state,
                                    nn_input(puzzle_from_string('467529813931478526258316497685947231372165948149283765514832679793654182826791354'))))
        self.assertTrue(torch.equal(reward, torch.tensor([[1024]])))
        self.assertTrue(done.item())

        # Puzzle 2
        puzzle = nn_input(puzzle_from_string('867413592394852617251697843573126984946785321128349756432968175685271439719534208'))
        soln = nn_input(puzzle_from_string('867413592394852617251697843573126984946785321128349756432968175685271439719534268'))
        env.reset(puzzle, soln)

        next_state, reward, done = env.step(torch.tensor([[79, 6]]))
        self.assertTrue(torch.equal(next_state,
                                    nn_input(puzzle_from_string('867413592394852617251697843573126984946785321128349756432968175685271439719534268'))))
        self.assertTrue(torch.equal(reward, torch.tensor([[1024]])))
        self.assertTrue(done.item())

        # Another move after finished
        next_state, reward, done = env.step(torch.tensor([[79, 6]]))
        self.assertTrue(torch.equal(next_state,
                                    nn_input(puzzle_from_string('867413592394852617251697843573126984946785321128349756432968175685271439719534268'))))
        self.assertTrue(torch.equal(reward, torch.tensor([[0]])))
        self.assertTrue(done.item())        

    def test_env_parallel_steps(self):
        puzzles = [
            '000007500020090040800206000200008491051400008000009352015604789092005160040901200',
            '090380105318500974065179238006900712001203000024751306100690007652007400080402601',
            '183745692746921835295368741467853219831296457529174368912637584658412973304589126'
        ]
        solns = [
            '169847523527193846834256917273568491951432678486719352315624789792385164648971235',
            '297384165318526974465179238536948712871263549924751386143695827652817493789432651',
            '183745692746921835295368741467853219831296457529174368912637584658412973374589126'
        ]
        puzzles = nn_input(torch.cat([puzzle_from_string(p) for p in puzzles], dim = 2))
        solns = nn_input(torch.cat([puzzle_from_string(s) for s in solns], dim = 2))
        env = SudokuEnv()
        env.reset(puzzles, solns)
        actions = torch.tensor([
            [0, 1],
            [79, 3],
            [73, 7]
        ])
        next_state, reward, done = env.step(actions)
        ns_exp = torch.cat([
            torch.reshape(puzzle_from_string('100007500020090040800206000200008491051400008000009352015604789092005160040901200'), (1, 81)),
            torch.reshape(puzzle_from_string('090380105318500974065179238006900712001203000024751306100690007652007400080402631'), (1, 81)),
            torch.reshape(puzzle_from_string('183745692746921835295368741467853219831296457529174368912637584658412973374589126'), (1, 81))
        ], dim = 0)
        self.assertTrue(torch.equal(next_state, ns_exp))
        self.assertTrue(torch.equal(reward, torch.tensor([[16], [-1024], [1024]])))
        self.assertTrue(torch.equal(done, torch.tensor([[False], [True], [True]])))

class TestSudokuLoader(tst.TestCase):
    def test_loader_next(self):
        ''' Note: will take several minutes to run '''
        chunksize = 128
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loader = SudokuLoader(db_path='postgresql://chris:@/Sudoku.db', n=3, chunksize=chunksize)
        for i in range(2):
            puzzles, solns = loader.next(device)
            puzzles = puzzles.to('cpu')
            solns = solns.to('cpu')
            self.assertEqual(puzzles.size(), solns.size())
            self.assertEqual(puzzles.size(), torch.Size((chunksize * 288, 81)))
            for p, s in zip(puzzles, solns):
                p = torch.reshape(p, (9, 9, 1))
                s = torch.reshape(s, (9, 9, 1))
                self.assertTrue(torch.equal(s, solve_sudoku(p)[0]))
        
if __name__ == 'main':
    tst.main()
