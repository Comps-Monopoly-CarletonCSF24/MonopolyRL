from player import Player
from board import Board, Property, Cell
from log import Log
import numpy as np

property_by_group_zeros = {'Brown': 0, 'Railroads': 0, 'Lightblue':0, 
                        'Pink': 0, 'Utilities': 0, 'Orange': 0,
                        'Red': 0, 'Yellow': 0, 'Green': 0, 'Indigo': 0}
num_property_per_group = {'Brown': 2, 'Railroads': 4, 'Lightblue': 3, 
                        'Pink': 3, 'Utilities': 4, 'Orange': 3,
                        'Red': 3, 'Yellow': 3, 'Green': 3, 'Indigo': 2}
group_indices = {'Brown': 0, 'Railroads': 1, 'Lightblue': 2, 'Pink': 3, 
                    'Utilities': 4, 'Orange': 5, 'Red': 6, 'Yellow': 7, 
                    'Green': 8, 'Indigo': 9}
Num_Groups = 10
#least common multiple
LCM_Property_Per_Group = 12
Num_Total_Cells = 40

class State:
    state = None
    def __init__(self, current_player: Player, players: list, board):
        area = self.get_area(current_player, players)
        position = self.get_position(current_player.position)
        finance = self.get_finance(current_player, players)
        self.state = self.get_state(area, position, finance)

    def get_area(self, current_player: Player, players: Player) -> np.ndarray:
        """Reads the board for properties belonging to the agent and other players.

        Args:
            board (Board): _description_

        Returns:
            np.ndarray: _description_
        """
        self_property_by_group = self.get_property_by_group(current_player)
        others_property_by_group = np.zeros(Num_Groups)
        for player in players:
            if not (player.is_bankrupt or player.name == current_player.name):
                others_property_by_group += self.get_property_by_group(player)
        area = np.vstack(self_property_by_group, others_property_by_group)
        self.area = area
    
    def get_property_by_group(player:Player) -> np.ndarray:
        """Gets the number of property of each group that a player has

        Args:
            player (Player): _description_

        Returns:
            property_by_group: _description_
        """
        property_by_group = [0] * Num_Groups
        for property in player.owned:
            group_index = group_indices[property.group]
            property_by_group[group_index] += LCM_Property_Per_Group / num_property_per_group[property.group]
            if property.has_houses:
                property_by_group[group_index] = LCM_Property_Per_Group + property.has_houses
            if property.has_hotel:
                property_by_group[group_index] = LCM_Property_Per_Group + property.has_hotel
        return np.array(property_by_group)
    
    def get_position(self, position_int : int) -> float:
        """Converts a position in [0,39] to one in [0,1)
        
        Args:
            position_int (int): _description_
        """
        position_float = (position_int) / (Num_Total_Cells - 1)
        self.position = position_float
        
    def get_finance(self, current_player: Player, players: list) -> np.ndarray:
        """Gets the finance state vector from the player's money and properties

        Args:
            money (_type_): _description_
            properties (_type_): _description_
        """
        property_others_accumulated = 0
        for player in players:
            if not (player.is_bankrupt or player.name == current_player.name):
                property_others_accumulated += self.get_num_property(player)
        property_ratio = self.get_num_property(current_player) / property_others_accumulated
        money_normalized = self.sigmoid_money(current_player.money)
        finance = np.array([property_ratio, money_normalized])
        self.finance = finance

    def get_num_property(player: Player):
        """returns the number of properties a player has

        Args:
            player (Player): _description_
        """
        total_property = 0
        for property in player.owned:
            total_property += 1 + property.has_hotel + property.has_houses
        return total_property

    def sigmoid_money(money):
        """normalizes the amount of money a player has with a sigmoid function

        Args:
            money (_type_): _description_

        Returns:
            _type_: _description_
        """
        return money / ( 1 + abs(money))
       
    def get_state(self) -> np.ndarray:
        """converts the 3 vectors/integers into a new, 1*23 vector.

        Returns:
            np.ndarray: _description_
        """
        # Flatten the 2x10 area array to 1x20
        flattened_area = self.area.flatten()
        # Combine all into a 1x23 array
        self.state  = np.concatenate((flattened_area, [self.position], self.finance))