import numpy as np
from random import randint

xInitial = np.array([
    ['W', 'W', 'W'],
    ['W', 'W', 'W'],
    ['W', 'W', 'W'],
    ['G', 'G', 'G'],
    ['G', 'G', 'G'],
    ['G', 'G', 'G'],
    ['R', 'R', 'R'],
    ['R', 'R', 'R'],
    ['R', 'R', 'R'],
    ['B', 'B', 'B'],
    ['B', 'B', 'B'],
    ['B', 'B', 'B'],
    ['O', 'O', 'O'],
    ['O', 'O', 'O'],
    ['O', 'O', 'O'],
    ['Y', 'Y', 'Y'],
    ['Y', 'Y', 'Y'],
    ['Y', 'Y', 'Y']
])


def restore(x):
    x[0:18, 0:3] = xInitial[0:18, 0:3]


def FrontCW(x):  # action 1
    x[6:9, 0:3] = np.fliplr(x[6:9, 0:3].transpose())
    temp1 = np.array(x[2, 0:3])
    temp2 = np.array(x[9:12, 0])
    temp3 = np.array(x[15, 0:3])
    temp4 = np.array(x[3:6, 2])
    x[2, 0:3] = np.fliplr([temp4])[0]
    x[9:12, 0] = temp1
    x[15, 0:3] = np.fliplr([temp2])[0]
    x[3:6, 2] = temp3


def FrontACW(x):  # action 2
    FrontCW(x)
    FrontCW(x)
    FrontCW(x)


def UpCW(x):  # action 3
    x[0:3, 0:3] = np.fliplr(x[0:3, 0:3].transpose())
    temp1 = np.array(x[12, 0:3])
    temp2 = np.array(x[9, 0:3])
    temp3 = np.array(x[6, 0:3])
    temp4 = np.array(x[3, 0:3])
    x[12, 0:3] = temp4
    x[9, 0:3] = temp1
    x[6, 0:3] = temp2
    x[3, 0:3] = temp3


def UpACW(x):  # acion 4
    UpCW(x)
    UpCW(x)
    UpCW(x)


def DownCW(x):  # action 5 Front down clock wise
    x[15:18, 0:3] = np.fliplr(x[15:18, 0:3].transpose())
    temp1 = np.array(x[8, 0:3])
    temp2 = np.array(x[11, 0:3])
    temp3 = np.array(x[14, 0:3])
    temp4 = np.array(x[5, 0:3])
    x[8, 0:3] = temp4
    x[11, 0:3] = temp1
    x[14, 0:3] = temp2
    x[5, 0:3] = temp3


def DownACW(x):  # action 6
    DownCW(x)
    DownCW(x)
    DownCW(x)


def LeftCW(x):  # action 7

    x[3:6, 0:3] = np.fliplr(x[3:6, 0:3].transpose())
    temp1 = np.array(x[0:3, 0])
    temp2 = np.array(x[6:9, 0])
    temp3 = np.array(x[15:18, 0])
    temp4 = np.array(x[12:15, 2])
    x[0:3, 0] = np.fliplr([temp4])[0]
    x[6:9, 0] = temp1
    x[15:18, 0] = temp2
    x[12:15, 2] = np.fliplr([temp3])[0]


def LeftACW(x):  # action 8
    LeftCW(x)
    LeftCW(x)
    LeftCW(x)


def RightCW(x):  # action 9 Front right clock wise

    x[9:12, 0:3] = np.fliplr(x[9:12, 0:3].transpose())
    temp1 = np.array(x[0:3, 2])
    temp2 = np.array(x[12:15, 0])
    temp3 = np.array(x[15:18, 2])
    temp4 = np.array(x[6:9, 2])
    x[0:3, 2] = temp4
    x[12:15, 0] = np.fliplr([temp1])[0]
    x[15:18, 2] = np.fliplr([temp2])[0]
    x[6:9, 2] = temp3


def RightACW(x):  # action 10
    RightCW(x)
    RightCW(x)
    RightCW(x)


def BackCW(x):  # action 11 Front  back clock wise

    x[12:15, :] = np.fliplr(x[12:15, :].transpose())
    temp1 = np.array(x[0, 0:3])
    temp2 = np.array(x[3:6, 0])
    temp3 = np.array(x[17, 0:3])
    temp4 = np.array(x[9:12, 2])
    x[0, 0:3] = temp4
    x[3:6, 0] = np.fliplr([temp1])[0]
    x[17, 0:3] = temp2
    x[9:12, 2] = np.fliplr([temp3])[0]


def BackACW(x):  # action 12
    BackCW(x)
    BackCW(x)
    BackCW(x)


def PrintCube(x, print_cube=False):
    str_cube = f"""
             {x[0, 0:3]}
             {x[1, 0:3]}
             {x[2, 0:3]}
{x[3, 0:3]}{x[6, 0:3]}{x[9, 0:3]}{x[12, 0:3]}
{x[4, 0:3]}{x[7, 0:3]}{x[10, 0:3]}{x[13, 0:3]}
{x[5, 0:3]}{x[8, 0:3]}{x[11, 0:3]}{x[14, 0:3]}
             {x[15, 0:3]}
             {x[16, 0:3]}
             {x[17, 0:3]}
    """
    if print_cube:
        print(str_cube)
    return str_cube


def make_move(x, move, reverse, print_cube=True):
    chosen_move = None
    if reverse == 1:
        if move % 2 == 0:
            move = move - 1
        else:
            move = move + 1
    if move == 1:
        FrontCW(x)
        chosen_move = "FrontCW"

    elif move == 2:
        FrontACW(x)
        chosen_move = "FrontACW"

    elif move == 3:
        UpCW(x)
        chosen_move = "UpCW"

    elif move == 4:
        UpACW(x)
        chosen_move = "UpACW"

    elif move == 5:
        DownCW(x)
        chosen_move = "DownCW"

    elif move == 6:
        DownACW(x)
        chosen_move = "DownACW"

    elif move == 7:
        LeftCW(x)
        chosen_move = "LeftCW"

    elif move == 8:
        LeftACW(x)
        chosen_move = "LeftACW"

    elif move == 9:
        RightCW(x)
        chosen_move = "RightCW"

    elif move == 10:
        RightACW(x)
        chosen_move = "RightACW"

    elif move == 11:
        BackCW(x)
        chosen_move = "BackCW"

    elif move == 12:
        BackACW(x)
        chosen_move = "BackACW"
    else:
        raise Exception('Wrong Move')
    if print_cube:
        PrintCube(x)
    return chosen_move


def scramble(x, num_scrambles=6):
    random_moves = [randint(1, 12) for _ in range(num_scrambles)]
    scramble_moves = [make_move(x, move, 0, False) for move in random_moves]
    str_cube = PrintCube(x, False)
    return scramble_moves, str_cube


if __name__ == '__main__':
    scramble_seqs = []
    for num_scrambles in range(1, 30, 3):
        x = np.array(xInitial)
        scramble_seq, str_cube = scramble(x, num_scrambles)
        scramble_seqs.append(scramble_seq)
        with open(f'inputs/input_{num_scrambles}.txt', 'w') as f:
            f.write(str_cube)
    with open('scramble_seqs.txt', 'w') as f:
        for scramble_seq in scramble_seqs:
            f.write(' '.join(scramble_seq))
            f.write('\n')
