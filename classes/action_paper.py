import numpy as np

property_count = 28
actions = ['buy', 'sell', 'do_nothing']
total_actions = property_count * len(actions)

class Action:
    def __init__(self, action_index: int):
        self.action_idx = action_index
        self.property_idx, self.action_type = map_action_index(action_index)
        
def map_action_index(action_index):
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
    if action_index < 0 or action_index >= total_actions:
        raise ValueError(
            f"The action index must be an integer between 0 and {total_actions - 1}."
        )
    property_idx = action_index // len(actions)
    action_type = actions[action_index % len(actions)]
    return property_idx, action_type

def is_excutable (self, player, board, property_idx, action_idx):
    '''
    Checks if the player can take the action that they are attempting to take.
    Returns true of the player can and False otherwise. 
    '''
    if action_idx == 1:
        return True
    elif action_idx == 0:
        if not board[property_idx].is_owned() and player.can_afford(board[property_idx].price):
            return True
        return False    

