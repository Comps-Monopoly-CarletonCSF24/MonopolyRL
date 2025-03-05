from classes.player_logistics import Player
import numpy as np
from settings import GameSettings
from classes.board import Board, Property, Cell
from classes.log import Log


# the number of properties on the board in each group
num_property_per_group = {'Brown': 2, 'Railroads': 4, 'Lightblue': 3, 'Pink': 3, 'Utilities': 2, 'Orange': 3, 'Red': 3, 'Yellow': 3, 'Green': 3, 'Indigo': 2}
# tn arbitrary index for groups between 1 - 9
group_indices = {'Brown': 0, 'Railroads': 1, 'Lightblue': 2, 'Pink': 3, 'Utilities': 4, 'Orange': 5, 'Red': 6, 'Yellow': 7, 'Green': 8, 'Indigo': 9}
# the cells in each group, indexed by the indices above
group_cell_indices = [[1, 3], [5, 15, 25, 35], [6,8,9], [11,13,14], [12,28], [16,18,19], [21,23,24], [26,27,29], [31,32,34], [37, 39]]


# number of groups ob the board
Num_Groups = 10
# least common multiple for the number of property per group on the board
LCM_Property_Per_Group = 12
# number of cells on the board
Num_Total_Cells = 40
# a number to represent how much property the player owns within one color, max 17
Total_Property_Points = 17
class State:
    state = None
    area = None
    position = None
    finance = None
    has_monopoly =None
    is_property =None
    has_more_money=None
    
    
    def __init__(self, current_player: Player, players: list):
        self.has_monopoly = False
        self.is_property = False
        self.has_more_money = False
        self.state = get_state(self.has_monopoly, self.is_property, self.has_more_money)
        

def is_property(current_player: Player, position_int: int) -> bool:
    """Determines if a position is/is not a player's property

    Args:
        position_int(int): a number between 0 and 39 representing the players position
    
    Returns:
        bool: true/false value that indicates if a certain position is a property
    
    """
    if position_int in current_player.owned:
        return True
    return False

def has_monopoly(current_player: Player, board: Board, current_position: int) -> bool:
    """ Returns if the player has monopoly over the current property color they are on
    Args:
        board(Board): the game board
        player(Player): the player to check for monopoly
        current_position:/
    
    Returns:
        A bool value indicating true/false indicating if the player has monopoly
    """
    if not isinstance(current_position, Property): #check if the current position is a property
        return False
    
    current_color = board.get_property(current_position).color
    color_group_properties = [prop for prop in board.properties
                              if prop.color == current_color]
    
    for property in color_group_properties:
        if property not in current_player.owned:
            return False
        
    return True

def has_more_money(current_player: Player, players: list) -> bool:
    """Determines if a player has more money than any other 2 players
    Args:
        current_player (Player): /
        players (list): a list of Player objects representing players that are alive
    Returns: 
        bool : a true/false value """

    # Get number of active players (not bankrupt)
    active_players = [p for p in players if not p.is_bankrupt]
    num_active_players = len(active_players)

    if num_active_players <= 2:
        #in 2 player games, just check if we have more money than the opponent
        for player in active_players:
            if player != current_player and not player.is_bankrupt:
                return current_player.money > player.money

        return False
    else:
        #case for 3+ alive players
        current_player_money = current_player.money
        count = 0
        for player in active_players:
            if not player.is_bankrupt:
                if current_player_money > player.money:
                    count += 1
                if count >= 2:
                    return True
    return False

def get_state(has_more_money: bool, has_monopoly:bool, is_property: bool) -> np.ndarray:
    """converts the 3 vectors/integers into a new one-dimensional vector

    Returns:
        state(np.ndarray): a 1 * 23 vector representing the state
    """
    #convert boolean to float values (True = 1.0, False = 0.0)
    state = np.array([
        float(has_monopoly),
        float(is_property),
        float(has_more_money)
    ], dtype = np.float64)

    return state