import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader

import data
import models
import utils
import pandas as pd
from timeit import default_timer as timer


defaultColors = 'YGORWB'
ops = 'LRUDFB'
invOps = {'L': 'R', 'R': 'L', 'U': 'D', 'D': 'U', 'F': 'B', 'B': 'F'}


class CubeLoss(nn.Module):
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean', 'sum')
        super(CubeLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, output, target):
        output = output.view(output.shape[0], -1)  # NSC -> N(SC)
        target = 2 * target[:, 0] + target[:, 1]

        loss = self.criterion(output, target)

        pred = torch.argmax(output, dim=1)
        acc = (pred == target).float()
        if self.reduction == 'mean':
            acc = torch.mean(acc)
        if self.reduction == 'sum':
            acc = torch.sum(acc)
        return loss, acc


class Stats:
    def __init__(self):
        self.n = 0
        self.loss = 0
        self.acc = 0

    def accumulate(self, _n, _loss, _acc):
        self.n += _n
        self.loss += torch.sum(_loss).item()
        self.acc += torch.sum(_acc).item()

    def getLoss(self):
        return self.loss / max(1, self.n)

    def getAcc(self):
        return self.acc / max(1, self.n)


class PerClassStats:
    def __init__(self, maxScrambles):
        self.stats = [Stats() for _ in range(maxScrambles + 1)]

    def accumulate(self, scrambles, loss, acc):
        for scr, l, a in zip(scrambles, loss, acc):
            self.stats[scr].accumulate(1, l, a)

    def lossStr(self):
        return ''.join([f'{i}: {s.getLoss():.3f}  ' for i, s in enumerate(self.stats[1:], start=1)])

    def accStr(self):
        return ''.join([f'{i}: {100 * s.getAcc():.2f}%  ' for i, s in enumerate(self.stats[1:], start=1)])


def opToIndices(op):
    opIdx = ops.index(op[0])
    if len(op) == 1:
        amountIdx = 0
    if len(op) == 2 and op[1] == 'i':
        amountIdx = 1
    y = torch.tensor([opIdx, amountIdx], dtype=torch.long)
    return y


def indicesToOp(indices):
    op, amount = indices
    op, amount = op.item(), amount.item()
    op = ops[op]

    assert amount in (0, 1)
    if amount == 0:
        return op
    if amount == 1:
        return f'{op}i'


def strToIndices(cubeStr, colors=defaultColors):
    lines = cubeStr.split('\n')
    indices = torch.zeros((6, 3, 3), dtype=torch.long)  # SHW

    # x coordinates have stride 4, y coordinates have 3. See Cube.__str__() method for details
    _setIndices(indices[0], lines, 0, 3, colors)  # L
    _setIndices(indices[1], lines, 8, 3, colors)  # R
    _setIndices(indices[2], lines, 4, 0, colors)  # U
    _setIndices(indices[3], lines, 4, 6, colors)  # D
    _setIndices(indices[4], lines, 4, 3, colors)  # F
    _setIndices(indices[5], lines, 12, 3, colors)  # B
    return indices


def indicesToStr(indices, colors=defaultColors):
    lines = [' ' * 15] * 9
    _setString(indices[0], lines, 0, 3, colors)  # L
    _setString(indices[1], lines, 8, 3, colors)  # R
    _setString(indices[2], lines, 4, 0, colors)  # U
    _setString(indices[3], lines, 4, 6, colors)  # D
    _setString(indices[4], lines, 4, 3, colors)  # F
    _setString(indices[5], lines, 12, 3, colors)  # B

    cubeStr = '\n'.join(lines)
    return cubeStr


def indicesToOneHot(indices):
    eye = torch.eye(6, dtype=torch.float32)
    oneHot = eye[indices]  # SHWC
    oneHot = oneHot.permute(0, 3, 1, 2)  # SCHW
    return oneHot


def _setIndices(indices, lines, x0, y0, colors=defaultColors):
    for y in range(3):
        for x in range(3):
            indices[y][x] = colors.index(lines[y0 + y][x0 + x])


def _setString(indices, lines, x0, y0, colors=defaultColors):
    for y in range(3):
        for x in range(3):
            char = colors[indices[y, x].item()]
            lines[y0 + y] = lines[y0 + y][:x0 + x] + char + lines[y0 + y][x0 + x + 1:]



class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def getModel(mul=1):
    return nn.Sequential(
        Flatten(),
        nn.Linear(6 * 6 * 3 * 3, round(mul * 4096)),
        nn.ReLU(inplace=True),
        nn.Linear(round(mul * 4096), round(mul * 2048)),
        nn.ReLU(inplace=True),
        nn.Linear(round(mul * 2048), round(mul * 512)),
        nn.ReLU(inplace=True),
        nn.Linear(round(mul * 512), 6 * 2),
    )


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
maxScrambles = 20
modelDir = '../models'


def main(start_epoch=1, end_epoch=1000, save_frequency=50, mode='train', numSolves=20, scrambles_start=1, scrambles_range=30, scrambles_step=3):
    tests = {}
    print(f'Using device {device}')
    assert mode in ('train', 'test')
    assert save_frequency > 0

    ds = data.RubikDataset(60000, maxScrambles)
    dl = DataLoader(ds, batch_size=64, num_workers=1)

    if start_epoch == 0:
        net = models.getModel(mul=1).to(device)
    else:
        net = torch.load(getModelPath(start_epoch), map_location=device)

    if mode == 'train':
        train(net, dl, start_epoch, end_epoch, save_frequency)
    if mode in ('train', 'test'):
        for scrambles in range(scrambles_start, scrambles_range, scrambles_step):
            print(f'solving for {scrambles} scrambles')
            scrambles_sum = 0
            start = timer()
            for i in range(numSolves):
                scrambles_sum += testSolve(net, scrambles=scrambles)
            end = timer() - start
            avg_time = round(end / numSolves, 2)
            avg = scrambles_sum / numSolves
            tests[scrambles] = [avg, avg_time]
            print(f'Average solution steps: {avg}\nTime it took: {avg_time} seconds on average')
    return tests


def train(net, dl, start_epoch, end_epoch, save_frequency):
    optim = torch.optim.Adam(net.parameters())
    criterion = utils.CubeLoss('none').to(device)

    for e in range(start_epoch + 1, end_epoch + 1):
        stats = utils.Stats()
        perClass = utils.PerClassStats(maxScrambles)

        for input, target, scrambles in dl:
            optim.zero_grad()
            input, target = input.to(device), target.to(device)
            output = net(input)
            loss, acc = criterion(output, target)
            torch.mean(loss).backward()
            optim.step()
            stats.accumulate(len(target), loss, acc)
            perClass.accumulate(scrambles, loss, acc)

        print(f'Epoch {e}/{end_epoch}:')
        print(f'acc={100 * stats.getAcc():.2f}%, loss={stats.getLoss():.3f}')
        print(f'acc= {perClass.accStr()}')
        print()
        if e % save_frequency == 0:
            os.makedirs(modelDir, exist_ok=True)
            filePath = getModelPath(e)
            print(f'Saving to {filePath}')
            torch.save(net, filePath)


def getModelPath(epoch):
    return f'{modelDir}/net.{epoch:04}.pt'


def testSolve(net, scrambles):
    env = data.RubikEnv(scrambles=scrambles)

    numSteps = 0
    for i in range(500):  # Randomize if solving fails
        pastStates = set()
        for i in range(100):  # Try solving for a short amount of time
            obs, done, hsh = env.getState()
            pastStates.add(hsh)
            if done:
                # print(f'Test with {scrambles} scrambles solved in {numSteps} steps')
                return numSteps

            obs = obs.view(1, *obs.shape).to(device)  # batch and GPU
            logits = net(obs)
            logits = logits.to('cpu').view(-1)  # CPU and debatch

            while True:  # Compute action
                action = torch.argmax(logits).item()
                envAction = torch.tensor([action // 2, action % 2], dtype=torch.long)
                env.step(envAction)
                hsh = env.getState()[2]
                if hsh in pastStates:
                    if logits[action] < -999:
                        break
                    logits[action] = -1000

                    envAction[1] = {0: 1, 1: 0}[envAction[1].item()]  # invert action
                    env.step(envAction)
                else:
                    break
            numSteps += 1

        for j in range(20):  # Randomize
            action = torch.LongTensor([random.randint(0, 5), random.randint(0, 1)])
            env.step(action)
            numSteps += 1

    print(f'did not solve {scrambles} scrambles')
    return scrambles


if __name__ == '__main__':
    res = main()
    df = pd.DataFrame(res).T.reset_index()
    df.columns=['scrambles', 'avg_steps', 'avg_time_sec']
    df.to_csv('DL_res.csv', index=False)
    print(df)
