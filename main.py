import os
import time
import cv2
import numpy as np
import pyautogui
import win32gui
import win32con
from strategy import find_best_move
import keyboard
from model_test import generate_chessboard


left, top = (694, 400)
right, bottom = (1894, 1600)

cols = 8
rows = 8

x_seris = np.linspace(left, right, cols + 1).astype(int)
y_seris = np.linspace(top, bottom, rows + 1).astype(int)
grid_width = int(x_seris[1] - x_seris[0])
grid_height = int(y_seris[1] - y_seris[0])

x_midpoints = (x_seris[1:] + x_seris[:-1]) // 2
y_midpoints = (y_seris[1:] + y_seris[:-1]) // 2


def coordinate_to_midposition(coordinates):
    """Coordinates means the indices of the chessboard."""
    i, j = coordinates
    x = x_midpoints[j]
    y = y_midpoints[i]
    return x, y


def move_piece(move):
    pyautogui.moveTo(*coordinate_to_midposition(move[0]), 0.1)
    pyautogui.mouseDown()
    pyautogui.moveTo(*coordinate_to_midposition(move[1]), 0.2)
    pyautogui.mouseUp()
    pyautogui.moveTo(300, 300)


def action():
    chessboard = generate_chessboard()
    move = find_best_move(chessboard)
    move_piece(move)


def main():
    keyboard.add_hotkey("c", action)
    keyboard.add_hotkey("q", lambda: os._exit(0))
    keyboard.wait("q")


if __name__ == "__main__":
    main()
