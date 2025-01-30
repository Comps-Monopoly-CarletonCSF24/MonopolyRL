from classes.player_logistics import Player
from classes.board import Property
import numpy as np
import copy

# the number of properties on the board in each group
num_property_per_group = {'Brown': 2, 'Railroads': 4, 'Lightblue': 3, 'Pink': 3, 'Utilities': 4, 'Orange': 3, 'Red': 3, 'Yellow': 3, 'Green': 3, 'Indigo': 2}
# tn arbitrary index for groups between 1 - 9
group_indices = {'Brown': 0, 'Railroads': 1, 'Lightblue': 2, 'Pink': 3, 'Utilities': 4, 'Orange': 5, 'Red': 6, 'Yellow': 7, 'Green': 8, 'Indigo': 9}
# number of groups ob the board
Num_Groups = 10
# least common multiple for the number of property per group on the board
LCM_Property_Per_Group = 12
# number of cells on the board
Num_Total_Cells = 40
# a number to represent how much property the player owns within one color, max 17
group_cell_indices = [[1, 3], [5, 15, 25, 35], [6,8,9], [11,13,14], [12,28], [16,18,19], [21,23,24], [26,27,29], [31,32,34], [37, 39]]
Total_Property_Points = 17

class State:
    state = None
    area = None
    position = None
    finance = None
    
    def __init__(self, current_player: Player, players: list):
        self.area = get_area(current_player, players)
        self.position = get_position(current_player.position)
        self.finance = get_finance(current_player, players)
        self.state = get_state(self.area, self.position, self.finance)
       
def get_area(current_player: Player, players: Player) -> np.ndarray:
    """ returns the area vector describing property owning percentage for each color
    Args:
        board (Board): _description_

    Returns:
        np.ndarray: each index (according to the assigned group indices) represents 
        the percentage of property points earned in each color group
    """
    self_property_points = get_property_points_by_group(current_player)
    others_property_points = np.zeros(Num_Groups)
    for player in players:
        if not (player.is_bankrupt or player.name == current_player.name):
            others_property_points += get_property_points_by_group(player)
    area = np.vstack((self_property_points, others_property_points)) / Total_Property_Points
    return area

def get_property_points_by_group(player:Player) -> np.ndarray:
    """Gets the number of property of each group that a player has

    Args:
        player (Player): 

    Returns:
        np.ndarray: each index (according to the assigned group indices) represents 
        property points that a player owns, max 17. For all land in the group the 
        player gets 12 (take fractions if not all owned), and for each house on 
        any property in that group, the player gets 1 point
    """
    property_by_group = [0] * Num_Groups
    for property in player.owned:
        group_index = group_indices[property.group]
        if property.has_hotel > 0:
            property_by_group[group_index] = Total_Property_Points
        elif property.has_houses > 0:
            property_by_group[group_index] = max(property_by_group[group_index], LCM_Property_Per_Group + property.has_houses)
        elif property_by_group[group_index] < LCM_Property_Per_Group:
            property_by_group[group_index] += LCM_Property_Per_Group / num_property_per_group[property.group]
            
    return np.array(property_by_group)

def get_position(position_int : int) -> float:
    """Converts a position in [0,39] to one in [0,1] by dividing 39

    Args:
        position_int (int): a number between 0 and 39 representing the players position

    Returns:
        float: a number between 0 and 1 representing the players position
    """
    position_float = (position_int) / (Num_Total_Cells - 1)
    return position_float
    
def get_finance(current_player: Player, players: list) -> np.ndarray:
    """Gets the finance state vector from the player's money and properties

    Args:
        current_player (Player): /
        players (list): a list of Player objects representing players that are alive

    Returns:
        np.ndarray: _description_
    """

    property_owned_total = 0
    for player in players:
        if not player.is_bankrupt:
            property_owned_total += get_num_property(player)
    if property_owned_total == 0:
        property_ratio = 0
    else:
        property_ratio = get_num_property(current_player) / property_owned_total
    
    money_normalized = sigmoid_money(current_player.money)
    finance = np.array([property_ratio, money_normalized])
    return finance

def get_num_property(player: Player, houses = False) -> int:
    """returns the number of properties a player has

    Args:
        player (Player): /
        houses (bool, optional): whether to count houses. Defaults to False.

    Returns:
        int: total number of property the player has
    """
    total_property = 0
    for property in player.owned:
        total_property += 1 
        if houses:
            total_property += property.has_hotel + property.has_houses
    return total_property

def sigmoid_money(money: int) -> float:
    """normalizes the amount of money a player has with a sigmoid function

    Args:
        money (_type_): _description_

    Returns:
        _type_: _description_
    """
    return money / ( 1 + abs(money))
    
def get_state(area: np.ndarray, position: int, finance: np.ndarray) -> np.ndarray:
    """converts the 3 vectors/integers into a new one-dimensional vector

    Returns:
        state(np.ndarray): a 1 * 23 vector representing the state
    """
    flattened_area = area.flatten()
    state = np.concatenate((flattened_area, [position], finance))
    return state