from tkinter import Canvas
from classes.board import Property

class MonopolyBoard:
    def __init__(self, root, board):
        self.root = root
        self.root.title("Monopoly Board")
        
        self.canvas = Canvas(root, width=800, height=800, bg="white")
        self.canvas.pack()
        
        self.game_board = board # Initialize the game board
        self.draw_board()
    
    def draw_board(self, position=0, player_name = ""):
        """Draws the Monopoly board using the actual board data."""
        cell_size = 60
        board_positions = self.get_board_positions()
        groups = ["Brown", "Pink", "Orange", "Red", "Yellow", "Green", "Indigo"]
        for i, cell in enumerate(self.game_board.cells):
            x, y = board_positions[i]
            # if position == i:
            #     self.add_player_position(x, y, cell_size, player_name)
            if isinstance(cell, Property):
                if cell.group in groups:
                    self.draw_cell(x, y, cell_size, cell.name, cell.group)
                else:
                    self.draw_cell(x, y, cell_size, cell.name, "White")
            else:
                self.draw_cell(x, y, cell_size, cell.name, "White")
            
    
    def draw_cell(self, x, y, size, name, color_group):
        """Draw a single square cell with a name on the board."""
        self.canvas.create_rectangle(x, y, x + size, y + size, outline="black", fill = color_group, width=2)
        self.canvas.create_text(x + size/2, y + size/2, text=name, font=("Arial", 8))
        
    def add_player_position (self, x, y, size, player_name):
        self.canvas.create_text(x + size/2, y + size/2, text= player_name, font=("Arial", 18))

    def get_board_positions(self):
        """Returns a mapping of board indices to positions on the canvas."""
        positions = []
        cell_size = 60
        board_size = 11
        
        # Bottom row (left to right)
        for i in range(board_size):
            positions.append((i * cell_size, (board_size - 1) * cell_size))
        
        # Right column (bottom to top)
        for i in range(1, board_size):
            positions.append(((board_size - 1) * cell_size, (board_size - 1 - i) * cell_size))
        
        # Top row (right to left)
        for i in range(1, board_size):
            positions.append(((board_size - 1 - i) * cell_size, 0))
        
        # Left column (top to bottom)
        for i in range(1, board_size - 1):
            positions.append((0, i * cell_size))
        
        return positions
