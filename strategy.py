import numpy as np


def find_best_move(chessboard):
    """
    A simple strategy that finds the highest value swap on the chessboard.
    This is a simple game that eliminate every 3 consecutive pieces.
    Player can choose to switch any two adjacent pieces to make 3 or more consecutive pieces.

    chessboard: A 2D numpy array representing the chessboard, the size is 8 * 8 by default.
    Returns: A tuple representing the best swap.
    """
    swaps = []
    for i in range(8):
        for j in range(7):
            swaps.append(((i, j), (i, j + 1)))
    for i in range(7):
        for j in range(8):
            swaps.append(((i, j), (i + 1, j)))

    best_swap = None
    best_value = 0
    for swap in swaps:
        value = evaluate_swap(chessboard, swap)
        if value is not None and value > best_value:
            best_value = value
            best_swap = swap

    return best_swap


def evaluate_swap(chessboard, swap):
    """
    Evaluate a swap on the chessboard.
    chessboard: A 2D numpy array representing the chessboard, the size is 8 * 8 by default.
    swap: A tuple representing the swap to evaluate.
    Returns: An integer representing the value of the swap.
    """
    newboard = np.copy(chessboard)
    newboard[swap[0]], newboard[swap[1]] = newboard[swap[1]], newboard[swap[0]]
    value = calculate_value(newboard)

    return value


def check_consecutive_in_line(arr, threshhold=3):
    """
    >>> check_consecutive_in_line([1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
    [(0, 3), (5, 5)]
    >>> check_consecutive_in_line([2, 2, 1, 3, 4, 4, 4, 4, 3, 5])
    [(4, 4)]
    >>> check_consecutive_in_line([1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 5)
    [(4, 6)]
    >>> check_consecutive_in_line([0, 3, 3, 3, 0, 2, 2, 1, 1, 1])
    [(1, 3), (7, 3)]
    """
    res = []
    count = 0
    for i in range(0, len(arr)):
        if arr[i] == 0:
            count = 0
        else:
            if i == 0:
                count = 1
            elif arr[i] == arr[i - 1]:
                count += 1
                if count >= threshhold:
                    res.append((i - count + 1, count))
            else:
                count = 1
    d = {k: v for k, v in res}
    res = [(k, d[k]) for k in d]
    return res


def check_consecutive_in_board(chessboard):
    row_pair = []
    for i in range(8):
        temp = check_consecutive_in_line(chessboard[i])
        if temp:
            for j, length in temp:
                row_pair.append(((i, j), length))
    col_pair = []
    for j in range(8):
        temp = check_consecutive_in_line(chessboard[:, j])
        if temp:
            for i, length in temp:
                col_pair.append(((i, j), length))
    return row_pair, col_pair


def updateable(row_pair, col_pair):
    return row_pair or col_pair


def prepare_board(chessboard, row_pair, col_pair):
    for (i, j), length in row_pair:
        chessboard[i, j : j + length] = -1
    for (i, j), length in col_pair:
        chessboard[i : i + length, j] = -1


def board_value(chessboard):
    return (chessboard == -1).sum().item()


def update_col(col_arr):
    for i in range(len(col_arr)):
        if col_arr[i] == -1:
            if i == 0:
                col_arr[i] = 0
            else:
                col_arr[1 : i + 1] = col_arr[:i]
                col_arr[0] = 0


def update_board(chessboard):
    """
    >>> chessboard = np.array(
    ...     [[1, 1, 1, 0, 0, 1, 1, 1],
    ...      [2, 2, 1, 3, 4, 4, 4, 4],
    ...      [1, 1, 1, 0, 1, 1, 1, 1],
    ...      [0, 3, 3, 3, 0, 2, 2, 1],
    ...      [1, 1, 1, 0, 0, 1, 1, 1],
    ...      [2, 2, -1, -1, -1, 4, 4, 4],
    ...      [1, 1, -1, -1, -1, 1, 1, 1],
    ...      [0, 3, 3, 3, 0, 2, 2, 1]])
    >>> update_board(chessboard)
    >>> chessboard
    array([[1, 1, 0, 0, 0, 1, 1, 1],
           [2, 2, 0, 0, 0, 4, 4, 4],
           [1, 1, 1, 0, 0, 1, 1, 1],
           [0, 3, 1, 3, 4, 2, 2, 1],
           [1, 1, 1, 0, 1, 1, 1, 1],
           [2, 2, 3, 3, 0, 4, 4, 4],
           [1, 1, 1, 0, 0, 1, 1, 1],
           [0, 3, 3, 3, 0, 2, 2, 1]])
    """
    for j in range(8):
        update_col(chessboard[:, j])


def calculate_value(chessboard):
    row_pair, col_pair = check_consecutive_in_board(chessboard)
    if not updateable(row_pair, col_pair):
        return None
    else:
        value = 0
        tempboard = np.copy(chessboard)
        while row_pair or col_pair:
            prepare_board(tempboard, row_pair, col_pair)
            value += board_value(tempboard)
            update_board(tempboard)
            row_pair, col_pair = check_consecutive_in_board(tempboard)
        return value
