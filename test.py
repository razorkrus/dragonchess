import tkinter as tk
import threading
import time
import numpy as np


def create_grid(data):
    # Create the main window
    root = tk.Tk()
    root.title("chessboard")

    rows, cols = data.shape

    # Create a grid of labels
    for i in range(rows):
        for j in range(cols):
            label = tk.Label(
                root, text=data[i][j], borderwidth=1, relief="solid", width=10, height=2
            )
            label.grid(row=i, column=j, padx=5, pady=5)

    # Run the Tkinter event loop
    root.mainloop()


data = np.arange(64).reshape((8, 8))

for i in range(10):
    data = np.random.randint(low=1, high=10, size=(8, 8))

# Create and start a new thread to run the Tkinter main loop
thread = threading.Thread(target=create_grid, args=(data,))
thread.start()

# Continue with the rest of your script
print("The Tkinter window is running in a separate thread.")

# Call the function to create the grid
# create_grid(data)
