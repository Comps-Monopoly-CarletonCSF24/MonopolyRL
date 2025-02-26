from classes.player_logistics import Player
from classes.board import Board
from classes.log import Log
from classes.board import Property
from classes.state import group_cell_indices
import numpy as np

class Action:
    def __init__(self):
        self.properties = list(range(28))
        self.actions = ['buy', 'sell', 'do_nothing']  # Available actions for each property
        self.total_actions = len(self.properties) * len(self.actions)  # 1x84 action space

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

    def is_excutable(self, player, board, property_idx, action_idx):
        """
        Checks if the player can take the action that they are attempting to take.
        Returns true of the player can and False otherwise. 
        """
        try:
            #map the 0-27 index to actual board position
            actual_positions = [idx for group in group_cell_indices for idx in group]
            
            #check if property_idx is a valid index
            if property_idx >= len(actual_positions):
                return False
            
            #get the actual board position for this property
            board_position = actual_positions[property_idx]
            
            if board_position != player.position:
                #print(f"Cannot buy property at {board_position} when player is at position {player.position}")
                return False
            property = board.cells[board_position]
            if not isinstance(property, Property):
                return False
            
            if action_idx == 0: #buy
                return (property.owner is None and 
                        player.can_afford(property.cost_base))
            return True #do_nothing is always executable
        except Exception as e:
            #print(f"Error in is_excutable: {e}")
            return False