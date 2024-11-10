from player import Player
from board import Board
from log import Log
import numpy as np

class State:
    '''
    To represent Monopoly as a MDP we first represent the full set of
    knowledge a real human player would have, as the (observed) state
    of the agent. We formulate the state st as a 3-dimensional vector of
    objects containing information about the game’s area, position and
    finance current status at time t.
    The area object, contains information about the game’s properties, 
    meaning the properties possessed from the current player and
    his opponents at time t. More specifically, it is a 10×2 matrix where
    the first column represents the agent-player, the second the rest of
    its opponents and each row corresponds to one of the game’s colourgroups 
    (8 property groups, the group of all utilities and the group of all rail-roads). 
    Each element (x,y) specifies the percentage that the 
    player y owns from group x, where value 0 indicates that the player
    does not own anything and 1 that at least one hotel has been built.
    
    The position variable determines the player’s current position on
    the board in relation to its colour-group, scaled to [0,1] e.g. If the
    player is in the fourth colour group this value would be 0.4.
    
    The finance vector consists of 2 values, specifying the current
    player’s number of properties in comparison to those of his opponents’ 
    as well as a calculation of his current amount of money.
    Specifically concerning the first value, given players a, b and c and
    the number of properties they own as pa, pb, pc it will be pa
    pa+pb+pc. For the second value, since the maximum amount of money owned
    by a player, x, varies significantly, the corresponding variable is
    transformed to a bounded one with the use of the sigmoid function
    f(x) = x_1+|x|.
    
    A sample state s at a given time t would be as follows :
    st = {0.3529, 0, 0, 0.2352, 0, 0, ..., 0, 0.6543, 0.3319, 0.3432}
    where the first 20 values represent the area vector, the next one the
    player’s position on board and the last 2 the player’s current financial
    status. In this specific example, the player has completed 0.3529%
    of the maximum available development (building) in area 0, and his
    current financial status is equal to 0.3432.
    '''    
    groups = ['Brown', 'Railroads', 'Lightblue', 'Pink', 'Utilities', 'Orange', 'Red', 'Yellow', 'Green' , 'Indigo']
    players = ['Self', 'Other']
    
    area = np.zeros((2, 10))
    position = 0
    finance = np.zeros(2)    
    state = np.zeros(23)
    
    def __init__ (self, player: Player, board):
        self.get_area(board)
        self.get_position(player.position)
        self.get_finance(player.properties)
        self.consolidate_state_vectors(self.area, self.position, self.finance)

    def get_area (self, board :Board, player: Player) -> np.ndarray:
        """Reads the board for properties belonging to the agent and other players.

        Args:
            board (Board): _description_

        Returns:
            np.ndarray: _description_
        """
        area_array = []
        
        self.area = np.array(area_array)

    def get_position(self, position_int : int) -> float:
        """Converts a position in [0,39] to one in [0,1]
        
        Args:
            position_int (int): _description_
        """
        position_float = (position_int + 1) / 40
        self.position = position_float

        
    def get_finance(self, money: float, properties: list) -> np.ndarray:
        """Gets the finance state vector from the player's money and properties

        Args:
            money (_type_): _description_
            properties (_type_): _description_
        """
        self.finance = np.zeros(2) 
        
    def consolidate_state_vectors(self) -> np.ndarray:
        """converts the 3 vectors/integers into a new, 1*23 vector.

        Args:
            area (_type_): _description_
            position (_type_): _description_
            finance (_type_): _description_
        """
        # Flatten the 2x10 area array to 1x20
        flattened_area = self.area.flatten()
        # Combine all into a 1x23 array
        self.state  = np.concatenate((flattened_area, [self.position], self.finance))