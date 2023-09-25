#Ziman David zdim1981 524/2
import numpy as np
import random
import math
from copy import deepcopy

HUMAN = -1
BOT = 1
MAX_DEPTH = 1
positions = {
        -1: "H",
        +1: "B",
        -2: "X",
        0: ' '
    }

class Position(object):
    def __init__(self,   x,   y):
        self.x = x
        self.y = y


def getNeighbours(pos, board):
        neighbours = []
        x, y = pos.x, pos.y
        if x == 0 and y == 0:       #LEFT TOP
            if board[x+1][y] == 0:
                neighbours.append(Position(x+1, y))
            if board[x][y+1] == 0:
                neighbours.append(Position(x, y+1))
            if board[x+1][y+1] == 0:
                neighbours.append(Position(x+1, y+1))
        elif x == 0 and y != boardSize-1:       #LEFT 
            if board[x][y-1] == 0:
                neighbours.append(Position(x, y-1))
            if board[x+1][y-1] == 0:
                neighbours.append(Position(x+1, y-1))
            if board[x+1][y] == 0:
                neighbours.append(Position(x+1, y))
            if board[x+1][y+1] == 0:
                neighbours.append(Position(x+1, y+1))
            if board[x][y+1] == 0:
                neighbours.append(Position(x, y+1))
        elif x == 0 and y == boardSize-1:   #LEFT BOT
            if board[x][y-1] == 0:
                neighbours.append(Position(x, y-1))
            if board[x+1][y-1] == 0:
                neighbours.append(Position(x+1, y-1))
            if board[x+1][y] == 0:
                neighbours.append(Position(x+1, y))
        elif x != (boardSize-1) and y == (boardSize-1):     #BOT
            if board[x-1][y] == 0:
                neighbours.append(Position(x-1, y))
            if board[x-1][y-1] == 0:
                neighbours.append(Position(x-1, y-1))
            if board[x][y-1] == 0:
                neighbours.append(Position(x, y-1))
            if board[x+1][y-1] == 0:
                neighbours.append(Position(x+1, y-1))
            if board[x+1][y] == 0:
                neighbours.append(Position(x+1, y))
        elif x == (boardSize-1) and y == (boardSize-1):     #RIGHT BOT
            if board[x][y-1] == 0:
                neighbours.append(Position(x, y-1))
            if board[x-1][y-1] == 0:
                neighbours.append(Position(x-1, y-1))
            if board[x-1][y] == 0:
                neighbours.append(Position(x-1, y))
        elif x == (boardSize-1) and y != 0:     #RIGHT
            if board[x][y+1] == 0:
                neighbours.append(Position(x, y+1))
            if board[x-1][y+1] == 0:
                neighbours.append(Position(x-1, y+1))
            if board[x-1][y] == 0:
                neighbours.append(Position(x-1, y))
            if board[x-1][y-1] == 0:
                neighbours.append(Position(x-1, y-1))
            if board[x][y-1] == 0:
                neighbours.append(Position(x, y-1))
        elif x == (boardSize-1) and y == 0:     #RIGHT TOP
            if board[x-1][y] == 0:
                neighbours.append(Position(x-1, y))
            if board[x-1][y+1] == 0:
                neighbours.append(Position(x-1, y+1))
            if board[x][y+1] == 0:
                neighbours.append(Position(x, y+1))
        elif y == 0:        #TOP
            if board[x+1][y] == 0:
                neighbours.append(Position(x+1, y))
            if board[x+1][y+1] == 0:
                neighbours.append(Position(x+1, y+1))
            if board[x][y+1] == 0:
                neighbours.append(Position(x, y+1))
            if board[x-1][y+1] == 0:
                neighbours.append(Position(x-1, y+1))
            if board[x-1][y] == 0:
                neighbours.append(Position(x-1, y))
        else:       #INSIDE
            if board[x+1][y] == 0:
                neighbours.append(Position(x+1, y))
            if board[x+1][y+1] == 0:
                neighbours.append(Position(x+1, y+1))
            if board[x][y+1] == 0:
                neighbours.append(Position(x, y+1))
            if board[x-1][y+1] == 0:
                neighbours.append(Position(x-1, y+1))
            if board[x-1][y] == 0:
                neighbours.append(Position(x-1, y))
            if board[x-1][y-1] == 0:
                neighbours.append(Position(x-1, y-1))
            if board[x][y-1] == 0:
                neighbours.append(Position(x, y-1))
            if board[x+1][y-1] == 0:
                neighbours.append(Position(x+1, y-1))

        return neighbours


def getAvailablePositions(board):
    pos = []
    for i in range(boardSize):
        for j in range(boardSize):
            if board[i][j] == 0:
                pos.append(Position(i, j))
    return pos


def getPlayerPosition(player, board):
    for i in range(boardSize):
        for j in range(boardSize):
            if board[i][j] == player:
                return Position(i, j)

def isLoser(player, board):
    pos = getPlayerPosition(player, board)
    return len(getNeighbours(pos, board)) == 0


def movePlayer(x, y, player, board):
    if board[x][y] == 0:
        pos = getPlayerPosition(player, board)
        board[pos.x][pos.y] = 0
        board[x][y] = player
        return True
    return False

def blockPosition(x, y, board):
    if board[x][y] == 0:
        board[x][y] = -2
        return True
    return False

def getResult(player, board):
    if isLoser(player, board):
        return -player
    if isLoser(-player, board):
        return player


def elem(list, filter):
    for x in list:
        if filter(x):
            return True
    return False


def printBoard():
    line = boardSize*'----'
    print('\n' + line + '-')
    for row in board:
        for pos in row:
            symbol = positions[pos]
            print('|', symbol, end=' ')
        print('|\n' + line + '-')


def getScore(board):        #HEURISTIC
    botPos = getPlayerPosition(BOT, board)
    humanPos = getPlayerPosition(HUMAN, board)
    botVal = 0
    for pos in getNeighbours(botPos, board):
        botVal += len(getNeighbours(pos, board))
    humanVal = 0
    for pos in getNeighbours(humanPos, board):
        humanVal += len(getNeighbours(pos, board))
    return botVal - humanVal


def getSteps(player, board):        #ALL POSSIBLE OUTCOMES
    steps = []
    playerPos = getPlayerPosition(player, board)
    neighbours = getNeighbours(playerPos, board)
    for pos in neighbours:
        x, y = pos.x, pos.y
        movePlayer(x, y, player, board)
        allPos = getAvailablePositions(board)
        for block in allPos:
            bX, bY = block.x, block.y
            board[bX][bY] = -2
            steps.append(deepcopy(board))
            board[bX][bY] = 0
        movePlayer(playerPos.x, playerPos.y, player, board)
    return steps

def minMax(player, depth, board, alpha, beta):
    if isLoser(BOT, board) or isLoser(HUMAN, board):
        score = getResult(player, board)
        if score == 1:
            return math.inf
        else:
            return -math.inf
    if depth == MAX_DEPTH:
        score = getScore(board)
        return score
    if player == HUMAN:
        mini = math.inf
        for step in getSteps(player, board):
            mini = min(mini, minMax(-player, depth + 1, step, alpha, beta))
            beta = min(beta, mini)
            if beta <= alpha:
                break
        return mini
    else:
        maxi = -math.inf
        for step in getSteps(player, board):
            maxi = max(maxi, minMax(-player, depth + 1, step, alpha, beta))
            alpha = max(alpha, maxi)
            if beta <= alpha:
                break
        return maxi


def step(player, board):
    maxi = -math.inf
    pos = getPlayerPosition(player, board)
    for nbs in getNeighbours(pos, board):
        movePlayer(nbs.x, nbs.y, player, board)
        for block in getAvailablePositions(board):
            board[block.x][block.y] = -2
            value = minMax(-player, 0, board, -math.inf, math.inf)
            if value >= maxi:
                maxi = value
                move = [nbs, block]
            board[block.x][block.y] = 0
        movePlayer(pos.x, pos.y, player, board)
    return move


def botPlays():
    global board
    depth = len(getAvailablePositions(board))
    if depth == 0 or isLoser(BOT, board) or isLoser(HUMAN, board):
        return
    print(f'Computer turn: ')
    printBoard()
    nextStep, nextBlock = step(BOT, board)
    print("BOT STEPPED ON:", nextStep.x, nextStep.y)
    print("BOT BLOCKED:",nextBlock.x, nextBlock.y)
    movePlayer(nextStep.x, nextStep.y, BOT, board)
    board[nextBlock.x][nextBlock.y] = -2


def humanPlays():
    depth = len(getAvailablePositions(board))
    if depth == 0 or isLoser(BOT, board) or isLoser(HUMAN, board):
        return
    pos = getPlayerPosition(HUMAN, board)
    neighbours = getNeighbours(pos, board)
    print('Human turn: ')
    printBoard()
    moveX = int(input('ROW : '))
    moveY = int(input('COLUMN : '))
    while not elem(neighbours, lambda nb: nb.x == moveX and nb.y == moveY):
        print("Invalid position!")
        moveX = int(input('ROW : '))
        moveY = int(input('COLUMN : '))
    movePlayer(moveX, moveY, HUMAN, board)
    print("BLOCKING:")
    blockX = int(input('ROW : '))
    blockY = int(input('COLUMN : '))
    poses = getAvailablePositions(board)
    while not elem(poses, lambda nb: nb.x == blockX and nb.y == blockY):
        print("Invalid position!")
        print("BLOCKING:")
        blockX = int(input('ROW : '))
        blockY = int(input('COLUMN : '))
    blockPosition(blockX, blockY, board)

boardSize = int(input("Size of board = "))
board = np.zeros((boardSize, boardSize), dtype=int)

board[0][boardSize//2] = -1
board[boardSize-1][boardSize//2] = 1

starts = random.randint(0, 1)

while len(getAvailablePositions(board)) > 0 and not (isLoser(BOT, board) or isLoser(HUMAN, board)):
    if starts == 0:
        botPlays()
        who = ""
    humanPlays()
    if getResult(BOT, board):
        printBoard()
        print("you won")
        break
    botPlays()
    if getResult(HUMAN, board):
        printBoard()
        print("you lost")
        break







