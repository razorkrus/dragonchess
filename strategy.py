import numpy as np


def find_best_move(chessboard):
    """
    A simple strategy that finds the highest value move on the chessboard.
    This is a simple game that eliminate every 3 consecutive pieces.
    Player can choose to switch any two adjacent pieces to make 3 or more consecutive pieces.

    chessboard: A 2D numpy array representing the chessboard, the size is 8 * 8 by default.
    Returns: A tuple representing the best move.
    """
    moves = []
    for i in range(8):
        for j in range(7):
            moves.append(((i, j), (i, j + 1)))
    for i in range(7):
        for j in range(8):
            moves.append(((i, j), (i + 1, j)))

    best_move = None
    best_value = 0
    for move in moves:
        value = evaluate_move(chessboard, move)
        if value > best_value:
            best_value = value
            best_move = move

    return best_move


def evaluate_move(chessboard, move):
    """
    Evaluate a move on the chessboard.
    chessboard: A 2D numpy array representing the chessboard, the size is 8 * 8 by default.
    move: A tuple representing the move to evaluate.
    Returns: An integer representing the value of the move.
    """
    new_chessboard = np.copy(chessboard)
    new_chessboard[move[0]], new_chessboard[move[1]] = (
        new_chessboard[move[1]],
        new_chessboard[move[0]],
    )
    weight = 0
    for coordinates in move:
        weight += coordinate_value(new_chessboard, coordinates) ** 3
    return weight


def coordinate_value(chessboard, coordinates):
    """
    Get the value of the piece at the given coordinates.
    chessboard: A 2D numpy array representing the chessboard, the size is 8 * 8 by default.
    coordinates: A tuple representing the coordinates of the piece.
    Returns: An integer representing the count of the piece.
    """
    count_left = count_by_direction(chessboard, coordinates, left)
    count_right = count_by_direction(chessboard, coordinates, right)
    count_horizontal = count_left + count_right - 1
    value_horizontal = count_horizontal if count_horizontal >= 3 else 0

    count_up = count_by_direction(chessboard, coordinates, up)
    count_down = count_by_direction(chessboard, coordinates, down)
    count_vertical = count_up + count_down - 1
    value_vertical = count_vertical if count_vertical >= 3 else 0

    if value_horizontal >= 3 and value_vertical >= 3:
        value = value_horizontal + value_vertical - 1
    else:
        value = max(value_horizontal, value_vertical)
    return value


def count_by_direction(chessboard, coordinates, direction):
    """
    Check the count of the consecutive pieces in the given direction.
    chessboard: A 2D numpy array representing the chessboard, the size is 8 * 8 by default.
    coordinates: A tuple representing the coordinates of the piece.
    direction: A function representing the direction to check.
    Returns: An integer representing the count of the consecutive pieces.
    """
    count = 1
    while (direction(coordinates) is not None) and (
        chessboard[coordinates] == chessboard[direction(coordinates)]
    ):
        coordinates = direction(coordinates)
        count += 1
    return count


def left(coordinates):
    return (coordinates[0], coordinates[1] - 1) if coordinates[1] > 0 else None


def right(coordinates):
    return (coordinates[0], coordinates[1] + 1) if coordinates[1] < 7 else None


def up(coordinates):
    return (coordinates[0] - 1, coordinates[1]) if coordinates[0] > 0 else None


def down(coordinates):
    return (coordinates[0] + 1, coordinates[1]) if coordinates[0] < 7 else None
