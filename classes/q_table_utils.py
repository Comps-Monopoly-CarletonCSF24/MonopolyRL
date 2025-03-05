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
        print(f"hello")
        print(f"type is {type(q_table[state][action_type])}, with value {q_table[state][action_type]}")
        return q_table[state][action_type]
    else:
        print(f"State {state} or action {action_type} not found in Q-table, returning default Q-value.")
        return 0.0  # Return a default Q-value if the state-action pair is not found

def calculate_q_reward(player, board, players):
    """Calculate the reward based on the player's current state."""
    reward = 0.0
    
    current_state = player.get_state(board, players)
    has_monopoly, is_property, has_more_money = current_state

    # Smaller base reward for money to reduce do_nothing preference
    reward += player.money * 0.0001  # Reduced from 0.001

    current_property = board.cells[player.position]
    # Check if the current position is a property
    if isinstance(current_property, Property):
        if is_property:
            # Reward for landing on own property
            reward += 5
        elif current_property.owner and current_property.owner != player:
            # Penalty for landing on opponent's property
            reward -= 10

    # Increase reward for buying properties
    if (isinstance(current_property, Property) and current_property.owner == None and
        current_property.cost_base <= player.money - player.settings.unspendable_cash):
        # Convert penalty to reward for buying
        buy_reward = 50  # Base reward for buying
        if has_monopoly == 1.0:
            buy_reward *= 1.5  # 50% more reward if it could complete monopoly
        if has_more_money == 1.0:
            buy_reward *= 1.2  # 20% more reward if player has more money
        reward += buy_reward
        
        # Add extra reward if property would complete a color group
        if hasattr(current_property, 'color'):
            same_color_owned = sum(1 for prop in player.owned 
                                 if hasattr(prop, 'color') and prop.color == current_property.color)
            total_in_color = sum(1 for prop in board.properties 
                               if hasattr(prop, 'color') and prop.color == current_property.color)
            if same_color_owned == total_in_color - 1:  # Would complete the set
                reward += 100  # Big reward for completing monopoly

    # Increased rewards for owned properties
    reward += len(player.owned) * 5  # Increased from 20
    
    # Higher rewards for color groups
    color_groups = {}
    for prop in player.owned:
        if hasattr(prop, 'color'):
            if prop.color not in color_groups:
                color_groups[prop.color] = 0
            color_groups[prop.color] += 1
            # Exponential reward for more properties in same color
            reward += (color_groups[prop.color] ** 3) * 5  # Increased multiplier and power
    
    # Much bigger reward for monopolies
    for group in board.groups.values():
        if all(prop.owner == player for prop in group):
            reward += 100  # Increased from 200
            # Extra reward for developed monopolies
            if hasattr(group[0], 'houses'):
                houses = sum(prop.houses for prop in group)
                reward += houses * 20  # Increased from 5

    # Strategic rewards
    try:
        opponent = [p for p in players if p != player and not p.is_bankrupt][0]
        
        # Bigger rewards/penalties based on property difference
        prop_difference = len(player.owned) - len(opponent.owned)
        if prop_difference > 0:
            reward += prop_difference * 10  # Increased from 30
        else:
            reward -= abs(prop_difference) * 20  # Increased penalty
            
    except IndexError:
        return 200.0  # Increased winning reward

    # Add small penalty for doing nothing
    if not player.owned:
        reward -= 10  # Penalty for having no properties

    return float(reward)

def log_to_file(message, filename="q_learning_progress.txt"):
    """Write log message to file with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(filename, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

def update_q_table(filename, state, action_idx, reward, next_state, next_available_actions, alpha, gamma, action_obj):
    """Update Q-table using Q-learning algorithm"""
    try:
        with open(filename, 'rb') as f:
            q_table = pickle.load(f)
    except (EOFError, FileNotFoundError):
        message = f"Creating new Q-table at {filename}"
        print(message)
        log_to_file(message)
        q_table = {}

    # Convert action index to action type
    _, action_type = action_obj.map_action_index(action_idx)

    # Initialize state in Q-table if not present
    if state not in q_table:
        q_table[state] = {action: 0.0 for action in action_obj.actions}

    # Get current Q-value
    current_q = float(q_table[state].get(action_type, 0.0))

    # Calculate max Q-value for next state
    next_max_q = 0.0
    if next_available_actions and next_state in q_table:
        next_q_values = []
        for next_action in next_available_actions:
            _, next_action_type = action_obj.map_action_index(next_action)
            next_q_values.append(float(q_table[next_state].get(next_action_type, 0.0)))
        next_max_q = max(next_q_values) if next_q_values else 0.0

    # Ensure reward is a float
    reward = float(reward)
    
    # Q-learning update formula
    new_q = current_q + alpha * (reward + gamma * next_max_q - current_q)
    
    # Store the new Q-value
    q_table[state][action_type] = new_q

    # Save updated Q-table
    try:
        with open(filename, 'wb') as f:
            pickle.dump(q_table, f)
        message = f"Updated Q-value for state {state}, action {action_type} to {new_q}"
        print(message)
        log_to_file(message)
    except Exception as e:
        message = f"Error saving Q-table to {filename}: {e}"
        print(message)
        log_to_file(message)

    # Monitor Q-value stability
    q_change = abs(new_q - current_q)
    if q_change < 0.01:
        message = f"Q-value stabilizing for state {state}, action {action_type}"
        print(message)
        log_to_file(message)
    elif q_change > 100:
        message = f"Warning: Large Q-value change ({q_change}) for state {state}"
        print(message)
        log_to_file(message)
    
    return new_q    