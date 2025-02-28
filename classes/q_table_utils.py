from classes.board import Property
from classes.action import Action
import numpy as np


def get_q_value(q_table, state, action, action_obj):
    """gets the q-value from q-table"""
    #if this is a buy action, start with a higher intial q value
    if action_obj.actions[action % len(action_obj.actions)] == 'buy':
        return q_table.get((state, action), 100.0)
    return q_table.get((state, action), 0.0)

def initialize_q_table(filename, actions):
    """Initialize Q-table with all possible states and actions"""
    try:
        with open(filename, 'w') as f:
            # Iterate over all possible states
            for state1 in [0.0, 1.0]:
                for state2 in [0.0, 1.0]:
                    for state3 in [0.0, 1.0]:
                        state = (state1, state2, state3)
                        # Iterate over all possible actions
                        for action in actions:
                            line = f"({state[0]}, {state[1]}, {state[2]}),{action},0.0\n"
                            f.write(line)
                            print(f"Writing to file: {line.strip()}")  # Debug statement
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
    finally:
        print(f"Q-table initialization completed in {filename}")

def get_q_value_from_file(filename, state, action_idx, action_obj):
    _, action_type = action_obj.map_action_index(action_idx)
    with open(filename, 'r') as f:
        for line in f:
            #split line into state-action-q value parts
            state_action, q = line.strip().rsplit(',',1)
            state_str, action_str = state_action.strip('()').rsplit(')', 1)

            #convert the state and action from strings to their respective types
            parsed_state = tuple(map(float, state_str.split(',')))
            parsed_action = action_str.strip().strip('"').strip(',').strip()

            #debug statements
            
            #check if the parsed state and action match the input
            if parsed_state == state and parsed_action == action_type:
                #print(f"parsed_state is {parsed_state} and state is {state}  ")
                #print(f"parsed_action is {parsed_action} and action is {action_type}")
                return float(q)

    return 0.0 #default q-value if not found

def update_q_table(filename, state, action_idx, reward, next_state, next_available_actions, alpha, gamma, action_obj):
    # Ensure next_state is a tuple of floats
    if not isinstance(next_state, tuple):
        raise ValueError("next_state must be a tuple of floats.")

    # Ensure next_available_actions is iterable
    if isinstance(next_available_actions, (int, np.integer)):
        next_available_actions = [next_available_actions]
    if isinstance(reward, tuple):
        reward = float(reward[0])
    

    # Ensure next_available_actions is always a list of regular integers
    next_available_actions = [int(action) for action in next_available_actions]
    #print(f"Normalized next available actions: {next_available_actions}")
    
    # Read current Q-value
    old_q = get_q_value_from_file(filename, state, action_idx, action_obj)
    
    if not isinstance(old_q, float):
        raise ValueError("old_q must be a float.")

    # Calculate max Q-value for the next state
    if not next_available_actions:
        next_max_q = 0.0
    else:
        next_q_values = [get_q_value_from_file(filename, next_state, next_action, action_obj) for next_action in next_available_actions]
        # Ensure all values in next_q_values are floats
        for q in next_q_values:
            if not isinstance(q, float):
                raise ValueError(f"Expected float in next_q_values, got {type(q)}")
        next_max_q = max(next_q_values) if next_q_values else 0.0

    # Debugging: Print variable types
    '''print(f"old_q: {old_q}, type: {type(old_q)}")
    print(f"alpha: {alpha}, type: {type(alpha)}")
    print(f"reward: {reward}, type: {type(reward)}")
    print(f"gamma: {gamma}, type: {type(gamma)}")
    print(f"next_max_q: {next_max_q}, type: {type(next_max_q)}")
'''
    # Q-learning update formula
    new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
    #print(f"new Q value: {new_q}. ")
    # Update the Q-value in the file
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
        
        with open(filename, "w") as f:
            for line in lines:
                #make sure the line contains a comma before processing 
                if ',' not in line:
                    continue #skip lines that do not have the expected format
                # Parse the line to extract state, action, and Q-value
                state_action, q = line.strip().rsplit(',', 1)

                #check if the state_action part contains the expected delimiter
                if '),' not in state_action:
                    continue #skip lines that do not have the expected format
                state_str, action_str = state_action.strip('()').rsplit('),', 1)
                
                # Convert the state and action from strings to their respective types
                parsed_state = tuple(map(float, state_str.split(',')))
                parsed_action = action_str.strip().strip('"')
                #print(f"Parsed state:{parsed_state}, State; {state}")
                _, action_type = action_obj.map_action_index(action_idx)
                #print(f"Parsed action:{parsed_action}, Action: {action_type}")
                #print(f"Action type is {action_type}")
                #check if the parsed state and action match the input
                print(f"{parsed_state == state and parsed_action == action_type}")
                if parsed_state == state and parsed_action == action_type:
                    new_q_rounded = round(new_q, 2)
                    new_line = f'({state[0]}, {state[1]}, {state[2]}),{action_type},{new_q_rounded:.2f}\n'
                    f.write(new_line)
                    print(f"Updated Q-value in file: {new_line.strip()}")
                else:
                    f.write(line)
    except IOError as e:
        print(f"Error updating Q-table in {filename}: {e}")

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