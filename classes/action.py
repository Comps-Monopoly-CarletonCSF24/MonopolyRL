from classes.player_logistics import Player
from classes.board import Board
from classes.log import Log
from classes.board import Property
import numpy as np

class Action:
    def __init__(self):
        self.properties = list(range(28))  # Property indices from 0 to 27
        self.actions = ['buy', 'do_nothing']  # Available actions for each property
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
        property = board.get_property(property_idx)
    
        if action_idx == 0:  # buy
            return property.owner == player
        #elif action_idx == 0:  # buy
           #return (property.owner is None and 
                    #player.can_afford(property.cost_base))  # Changed price to cost_base
        return True  # do_nothing is always executable
    
    def execute_action(self, player, board, property_idx, action_type, log, players):

        """
        Executes the action on the given property for the specified player.
        Args:
            player: the player who is taking the action
            board: the board object
            property_idx: the index of the property to be acted on
            action_type: the type of action to be taken
            log: the log object
            players: the list of players (for auction)
        """
       
        #property = board.get_property(property_idx)
        current_property = board.cells[player.position]
        
        if action_type == 'buy':  # Changed from 'buy_all' to 'buy'

            
            if (isinstance(current_property, Property) and 
                current_property.owner is None and 
                player.money >= current_property.cost_base):    
                
                player.buy_property(current_property, log)


        elif action_type == 'do_nothing':
            if isinstance(current_property, Property) and current_property.owner is None and player.money >= current_property.cost_base:
                log.add(f">>> {player.name} chose not to buy {current_property.name} (${current_property.cost_base})")
                #trigger auction
                player.auction_property(current_property, players, log)
