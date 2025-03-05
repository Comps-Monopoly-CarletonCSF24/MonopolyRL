from classes.board import Property
from classes.action import Action
import pickle
import os


def initialize_q_table(filename, actions):
    """Initialize Q-table with all possible states and actions using pickle"""
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Check if file already exists
        if os.path.exists(filename):
            print(f"Q-table already exists at {filename}")
            return
        
        print(f"Initializing Q-table at: {filename}")
        
        # Initialize the Q-table as a dictionary
        q_table = {}
        
        # Populate the Q-table with all possible state-action pairs
        for s1 in [0.0, 1.0]:
            for s2 in [0.0, 1.0]:
                for s3 in [0.0, 1.0]:
                    state = (s1, s2, s3)
                    q_table[state] = {action: 0.0 for action in actions}
        
        # Serialize the Q-table to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump(q_table, f)
        
        print(f"Successfully initialized Q-table at {filename}")
        
    except Exception as e:
        print(f"Error initializing Q-table: {e}")
        raise

def get_q_value(q_table, state, action, action_obj):
    """gets the q-value from q-table"""
    #if this is a buy action, start with a higher intial q value
    if action_obj.actions[action % len(action_obj.actions)] == 'buy':
        return q_table.get((state, action), 100.0)
    return q_table.get((state, action), 0.0)


def save_q_table(q_table, filename ="q_table.plk"):
    with open(filename, "wb") as f:
        pickle.dump(q_table, f)
    print(f"Q-table saved to {filename}")

def load_q_table(filename="q_table.pkl"):
    try:
        with open(filename,"rb") as f:
            q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")
        return q_table
    except FileNotFoundError:
        print(f"No Q-table found at {filename}, initializing new Q-table.")
        return {} #return an empty Q-table if the file does not exist




def update_q_table(filename, state, action_idx, reward, next_state, next_available_actions, alpha, gamma, action_obj):
    # Load the existing Q-table
    try:
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
    except (EOFError, FileNotFoundError):
        print(f"Error loading Q-table from {filename}, skipping update")
        return  

    # Convert action index to action type string
    _, action_type = action_obj.map_action_index(action_idx)

    # Ensure the state exists in the Q-table
    if state not in q_table:
        q_table[state] = {action: 0.0 for action in action_obj.actions}

    # Get the current Q-value
    old_q = q_table[state].get(action_type, 0.0)

    # Calculate max Q-value for the next state
    next_max_q = 0.0
    if next_available_actions:
        next_q_values = [q_table.get(next_state, {}).get(action_obj.map_action_index(a)[1], 0.0) for a in next_available_actions]
        next_max_q = max(next_q_values) if next_q_values else 0.0
    # Ensure all variables are floats
    print(f"reward is {reward}, and its type is {type(reward)}")
    old_q = float(old_q)
    reward = float(reward)
    gamma = float(gamma)
    next_max_q = float(next_max_q)
    # Q-learning update formula
    new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
    q_table[state][action_type] = new_q

    # Save the updated Q-table back to the file
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)

    print(f"Updated Q-value for state {state}, action {action_type} to {new_q}")

def get_q_value_from_pkl(filename, state, action_idx, action_obj):
    try:
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")
    except FileNotFoundError:
        print(f"No Q-table found at {filename}, returning default Q-value.")
        return 0.0  # Return a default Q-value if the file does not exist

    # Convert action index to action type string
    _, action_type = action_obj.map_action_index(action_idx)

    # Retrieve the Q-value from the Q-table
    if state in q_table and action_type in q_table[state]:
        return q_table[state][action_type]
    else:
        print(f"State {state} or action {action_type} not found in Q-table, returning default Q-value.")
        return 0.0  # Return a default Q-value if the state-action pair is not found

def calculate_q_reward(player, board, players):
    """Calculate the reward based on the player's current state."""
    reward = 0
    
    current_state = player.get_state(board, players)
    has_monopoly, is_property, has_more_money = current_state

    #base reward
    reward += player.money * 0.001
    reward += len(player.owned) * 20

    current_property = board.cells[player.position]
    if (isinstance(current_property, Property) and current_property.owner == None and
        current_property.cost_base <= player.money - player.settings.unspendable_cash):
        penalty =- 300
        if has_monopoly == 1.0:
            penalty *= 2
        if has_more_money == 1.0:
            penalty *= 2
        reward += penalty

    #reward for properties owned    
    reward += len(player.owned) * 20 #increase base property reward
    color_groups = {}
    for prop in player.owned:
        if hasattr(prop, 'color'):  # Only count colored properties
            if prop.color not in color_groups:
                color_groups[prop.color] = 0
            color_groups[prop.color] += 1

            #progressive rewards for more properties in the same color
            reward += (color_groups[prop.color] **2 ) * 30
        
    # Huge reward for monopolies
    for group in board.groups.values():
        if all(prop.owner == player for prop in group):
            reward += 200

        #extra reward for developed monopolies
        if hasattr(group[0], 'houses'):
            houses = sum(prop.houses for prop in group)
            reward += houses * 50        
    
    #strategic positioning rewards
    try:
        opponent = [p for p in players if p!= player and not p.is_bankrupt][0]
    except IndexError:
        # Handle case where all other players are bankrupt
        # Return a default state or end game state
        return (0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0)
    
    #reward for having more properties than opponent
    prop_difference = len(player.owned) - len(opponent.owned)
    reward += prop_difference * 30

    #penalty for having fewer properties than opponent
    if len(player.owned) < len(opponent.owned):
        reward -= 500

    return reward    