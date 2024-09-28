import pyautogui
import os
import time
import keyboard


left, top = 694, 400
width, height = 1200, 1200
region = (left, top, width, height)

# Define the number of columns and rows
cols, rows = 8, 8

# Calculate the width and height of each grid cell
cell_width = width // cols
cell_height = height // rows

# Create the output directory if it doesn't exist
output_dir = "dataset/train"
os.makedirs(output_dir, exist_ok=True)


def shot_and_crop(round=0):
    screenshot = pyautogui.screenshot(region=region)

    # Divide the screenshot into grid cells and save each cell
    for i in range(cols):
        for j in range(rows):
            # Calculate the coordinates of the current cell
            left = i * cell_width
            top = j * cell_height
            right = (i + 1) * cell_width
            bottom = (j + 1) * cell_height

            # Crop the cell from the screenshot
            cell = screenshot.crop((left, top, right, bottom))

            # Save the cell as an image file
            cell_filename = os.path.join(output_dir, f"round_{round}_cell_{i}_{j}.jpg")
            cell.save(cell_filename)

    print(f"Round {round} completed.")


r = 0


def capture_screen():
    global r
    shot_and_crop(r)
    time.sleep(1)
    r += 1


def quit_script():
    print("Quitting script.")
    os._exit(0)


# Set up hotkeys
keyboard.add_hotkey("c", capture_screen)
keyboard.add_hotkey("q", quit_script)

print("Press 'c' to capture the screen or 'q' to quit.")

# Keep the script running
keyboard.wait("q")
