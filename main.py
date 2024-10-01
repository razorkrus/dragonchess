import time
import cv2
import numpy as np
import pyautogui
import win32gui
import win32con
from strategy import find_best_move
from myutils import timeit_decorator


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


@timeit_decorator
def read_chessboard():
    # Load template images
    templates = [
        cv2.imread(f"template{i}.png", cv2.IMREAD_GRAYSCALE) for i in range(1, 7)
    ]

    # Initialize the chessboard
    chessboard = np.zeros((rows, cols), dtype=int)

    # screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
    # screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

    for i in range(rows):
        for j in range(cols):
            cell_region = (int(x_seris[j]), int(y_seris[i]), grid_width, grid_height)
            cellshot = pyautogui.screenshot(region=cell_region)
            cell_image = cv2.cvtColor(np.array(cellshot), cv2.COLOR_RGB2GRAY)

            # Find the best matching template
            best_match = match_template(cell_image, templates)

            # Assign the best match index to the chessboard
            chessboard[i, j] = best_match

    draw_chessboard(chessboard)
    return chessboard


def draw_chessboard(chessboard):
    grid_width = 50
    grid_height = 50
    # Create an image to draw the chessboard
    board_image = np.zeros((grid_height * rows, grid_width * cols, 3), dtype=np.uint8)

    # Draw the grid
    for i in range(rows):
        for j in range(cols):
            top_left = (j * grid_width, i * grid_height)
            bottom_right = ((j + 1) * grid_width, (i + 1) * grid_height)
            cv2.rectangle(board_image, top_left, bottom_right, (255, 255, 255), 1)

            # Draw the pieces (for simplicity, using text to represent pieces)
            piece = chessboard[i, j]
            if piece != 0:
                cv2.putText(
                    board_image,
                    str(piece),
                    (top_left[0] + grid_width // 2, top_left[1] + grid_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    # Display the chessboard
    cv2.namedWindow("Chessboard", cv2.WINDOW_NORMAL)
    cv2.imshow("Chessboard", board_image)
    set_window_on_top("Chessboard")
    set_window_position("Chessboard", 2100, 500)
    cv2.waitKey(1)  # Add a small delay to allow the window to update


def set_window_on_top(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
        )


def set_window_position(window_name, x, y):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOP,
            x,
            y,
            0,
            0,
            win32con.SWP_NOSIZE,
        )


def match_template(cell_image, templates):
    best_match = -1
    best_val = float("inf")  # Initialize with a large value for minimum comparison

    for idx, template in enumerate(templates):
        # Resize the template to match the cell size if necessary
        if template.shape != cell_image.shape:
            resized_template = cv2.resize(
                template, (cell_image.shape[1], cell_image.shape[0])
            )
        else:
            resized_template = template

        # Perform template matching
        result = cv2.matchTemplate(cell_image, resized_template, cv2.TM_SQDIFF_NORMED)

        # Find the minimum squared difference
        min_val, _, _, _ = cv2.minMaxLoc(result)

        # Update the best match if the current one is better
        if min_val < best_val:
            best_val = min_val
            best_match = idx

    return best_match


def coordinate_to_midposition(coordinates):
    """Coordinates means the indices of the chessboard."""
    i, j = coordinates
    x = x_midpoints[j]
    y = y_midpoints[i]
    return x, y


def move_piece(move):
    pyautogui.moveTo(*coordinate_to_midposition(move[0]), 0.1)
    pyautogui.mouseDown()
    pyautogui.moveTo(*coordinate_to_midposition(move[1]), 0.3)
    pyautogui.mouseUp()
    pyautogui.moveTo(300, 300)


def main():
    time.sleep(3)
    while True:
        chessboard = read_chessboard()
        move = find_best_move(chessboard)
        move_piece(move)


if __name__ == "__main__":
    main()
