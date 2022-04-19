"""
Tic Tac Toe Player
"""

import math
import numpy as np
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count = {"X": 0, "O": 0}
    
    for i, row in enumerate(board):
        for j in range(len(row)):
            if board[i][j] == "X":
                count["X"] += 1
            elif board[i][j] == "O":
                count["O"] += 1
                
    if count["X"] == count["O"]:
        return "X"
    else:
        return "O"


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    act = set()
    
    for i, row in enumerate(board):
        for j in range(len(row)):
            if board[i][j] == EMPTY:
                act.add((i,j))
    return act


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    board_copy = copy.deepcopy(board)
    if player(board) == "X":
        board_copy[i][j] = "X"
    elif player(board) == "O":
        board_copy[i][j] = "O"
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    
    #horizontals
    for i, row in enumerate(board):
        if row[0] == row[1] == row[2] == player(board):
            return player(board)
        
    
    #verticals
    for j in range(len(board)):
        if board[0][j] == board[1][j] == board[2][j] == player(board):
            return player(board)
        
    #diagonals
    
    if board[0][0] == board[1][1] == board[2][2] == player(board) or board[2][0] == board[1][1] == board[0][2] == player(board):
        return player(board)
            


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    
    for i, row in enumerate(board):
        for j in range(len(row)):
            if board[i][j] == EMPTY:
                return False
            
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if terminal(board):
        if winner(board) == "X":
            return 1
        elif winner(board) == "O":
            return -1
        else:
            return 0
    else:
        return None
        
def max_value(board):
    
    if terminal(board):
        return utility(board), None
    
    value = -np.inf
    action = None
    for action in actions(board):
        v, a = min_value(result(board, action))
        print(v, a)
        if v > value:
            value = v
            
            if v == 1:
                return value, action
            
    return value, action


def min_value(board):
    
    if terminal(board):
        return utility(board), None
    
    value = np.inf
    action = None
    for action in actions(board):
        v, a = max_value(result(board, action))
        print(v, a)
        if v < value:
            value = v
            
            if v == -1:
                return value, action
            
    return value, action


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    
    if terminal(board):
        return None
    
    else:
        if player(board) == "X":
            value, action = max_value(board)
            return action
        else:
            value, action = min_value(board)
            return action
        
#debug