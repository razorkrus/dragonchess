import pyautogui
import os
import time
import keyboard
import argparse


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for the chess AI."
    )
    parser.add_argument("rounds", type=int, help="Counter for the number of rounds.")
    args = parser.parse_args()

    r = args.rounds
    print("Ready to capture the screen.")

    while True:
        if keyboard.is_pressed("c"):
            shot_and_crop(r)
            r += 1
            while keyboard.is_pressed("c"):
                time.sleep(0.1)
        if keyboard.is_pressed("q"):
            break
        time.sleep(0.1)


if __name__ == "__main__":
    main()
