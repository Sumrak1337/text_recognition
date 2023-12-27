import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image


class Painter:
    def __init__(
        self,
        application: tk.Tk,
        save_root: str,
        image_name: str,
        canvas_width: int,
        canvas_height: int,
        default_size: int = 4,
        default_color: str = "black",
    ):
        self.application = application
        self.save_root = Path(save_root)
        self.image_name = image_name

        self.canvas = tk.Canvas(
            self.application,
            width=canvas_width,
            height=canvas_height,
            bg="white",
            bd=3,
            relief=tk.SUNKEN,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.setup_tools()
        self.setup_events()
        self.prev_x = None
        self.prev_y = None

        self.default_size = default_size
        self.default_color = default_color
        self.selected_tool = "pen"
        self.selected_color = self.default_color
        self.selected_size = self.default_size

    def setup_tools(self):
        tool_frame = ttk.LabelFrame(self.application)
        tool_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.Y)

        pen_button = ttk.Button(tool_frame, text="Pen", command=self.select_pen_tool)
        pen_button.pack(side=tk.TOP, padx=5, pady=5)

        eraser_button = ttk.Button(
            tool_frame, text="Eraser", command=self.select_eraser_tool
        )
        eraser_button.pack(side=tk.TOP, padx=5, pady=5)

        clear_button = ttk.Button(
            tool_frame, text="Clear all", command=self.clear_canvas
        )
        clear_button.pack(side=tk.TOP, padx=5, pady=5)

        save_button = ttk.Button(tool_frame, text="Save", command=self.save_picture)
        save_button.pack(side=tk.TOP, padx=5, pady=5)

        exit_button = ttk.Button(tool_frame, text="Exit", command=self.application.quit)
        exit_button.pack(side=tk.TOP, padx=5, pady=5)

    def setup_events(self):
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.release)

    def select_pen_tool(self):
        self.selected_tool = "pen"

    def select_eraser_tool(self):
        self.selected_tool = "eraser"

    def draw(self, event: tk.Event):
        x1 = event.x - self.selected_size
        y1 = event.y - self.selected_size
        x2 = event.x + self.selected_size
        y2 = event.y + self.selected_size

        if self.selected_tool == "pen":
            self.selected_size = self.default_size
            self.selected_color = self.default_color
        elif self.selected_tool == "eraser":
            self.selected_size = self.default_size + 20
            self.selected_color = "white"

        if self.prev_x is not None and self.prev_y is not None:
            self.canvas.create_oval(
                x1,
                y1,
                x2,
                y2,
                fill=self.selected_color,
                outline=self.selected_color,
                width=self.selected_size,
            )

        self.prev_x = event.x
        self.prev_y = event.y

    def release(self, *args):
        self.prev_x = None
        self.prev_y = None

    def clear_canvas(self):
        self.canvas.delete("all")

    def save_picture(self):
        file_name = str(self.save_root / self.image_name)
        self.canvas.postscript(file=file_name)
        img = Image.open(file_name)
        img.save(file_name, "png")
        self.application.quit()
