from classes.player import Player
from classes.board import Board
from classes.log import Log
import numpy as np

class Action:
    def __init__(self):
        self.properties = list(range(28))  # Property indices from 0 to 27
        self.actions = ['buy_all', 'do_nothing']  # Available actions for each property
        self.total_actions = len(self.properties) * len(self.actions)  # 1x56 action space

    def map_action_index(self, action_index):
        """
        Maps a flattened action index to a specific property and action type.

        Parameters:
            - action_index (int): The index of the action in the flattened action space.
        
        Returns:
            - tuple: A tuple containing the property index and the action type.
        """
        if not isinstance(action_index, (int, np.integer)):
            try:
                action_index = int(action_index)
            except ValueError:
                raise ValueError(
                    f"action_index must be an integer, got {type(action_index)}."
                )
    
        if action_index < 0 or action_index >= self.total_actions:
            raise ValueError(
                f"The action index must be an integer between 0 and {self.total_actions - 1}."
            )
        property_idx = action_index // len(self.actions)
        action_type = self.actions[action_index % len(self.actions)]
        return property_idx, action_type

    def execute_action(self, player, board, property_idx, action_type):
        """
        Executes the action on the given property for the specified player.
        """
        if action_type == 'buy_all':
            while board[property_idx].is_owned() and player.can_afford(board[property_idx].price): #using loop to buy all the property a player can
                player.buy_property(board[property_idx])
        elif action_type == 'do_nothing':
            pass  # No action is taken