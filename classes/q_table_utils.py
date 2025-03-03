from classes.board import Property
from classes.action import Action
import numpy as np
import os


def get_q_value(q_table, state, action, action_obj):
    """gets the q-value from q-table"""
    #if this is a buy action, start with a higher intial q value
    if action_obj.actions[action % len(action_obj.actions)] == 'buy':
        return q_table.get((state, action), 100.0)
    return q_table.get((state, action), 0.0)

def initialize_q_table(filename, actions):
    """Initialize Q-table with all possible states and actions"""
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
        
        # Write directly to the file (no need for temp file)
        with open(filename, 'w', encoding='utf-8') as f:
            # Write all possible state-action pairs
            for s1 in [0.0, 1.0]:
                for s2 in [0.0, 1.0]:
                    for s3 in [0.0, 1.0]:
                        state = (s1, s2, s3)
                        for action in actions:
                            line = f"{state[0]:.1f},{state[1]:.1f},{state[2]:.1f},{action},0.0\n"
                            f.write(line)
        
        print(f"Successfully initialized Q-table at {filename}")
        
    except Exception as e:
        print(f"Error initializing Q-table: {e}")
        raise

def update_q_table(filename, state, action_idx, reward, next_state, next_available_actions, alpha, gamma, action_obj):
    """Updates Q-value using Q-learning formula"""
    try:
        # Input validation
        if not isinstance(state, tuple) or len(state) != 3:
            raise ValueError(f"Invalid state format: {state}")
            
        # Get action type
        _, action_type = action_obj.map_action_index(action_idx)
        
        # Read current Q-values
        current_q_values = {}
        updated = False
        
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            try:
                # Parse line
                parts = line.strip().split(',')
                if len(parts) != 5:  # Should have 3 state values, action, and q-value
                    continue
                    
                s0, s1, s2, act, q_val = parts
                current_state = (float(s0), float(s1), float(s2))
                
                if current_state == state and act == action_type:
                    # Calculate new Q-value
                    old_q = float(q_val)
                    if not next_available_actions:
                        next_max_q = 0.0
                    else:
                        next_q_values = []
                        for next_act in next_available_actions:
                            _, next_action_type = action_obj.map_action_index(next_act)
                            for l in lines:
                                ns0, ns1, ns2, nact, nq = l.strip().split(',')
                                if (float(ns0), float(ns1), float(ns2)) == next_state and nact == next_action_type:
                                    next_q_values.append(float(nq))
                                    break
                        next_max_q = max(next_q_values) if next_q_values else 0.0
                    
                    new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
                    line = f"{state[0]:.1f},{state[1]:.1f},{state[2]:.1f},{action_type},{new_q:.2f}\n"
                    updated = True
                
                new_lines.append(line)
                
            except Exception as e:
                print(f"Error processing line '{line.strip()}': {e}")
                new_lines.append(line)
                continue
        
        if not updated:
            print(f"Warning: No matching state-action pair found for update")
            return False
            
        # Write back to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        return True
        
    except Exception as e:
        print(f"Error updating Q-table: {e}")
        return False

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
    """Updates Q-value using Q-learning formula"""
    try:
        # Input validation
        if not isinstance(state, tuple) or len(state) != 3:
            raise ValueError(f"Invalid state format: {state}")
            
        # Get action type
        _, action_type = action_obj.map_action_index(action_idx)
        
        # Read the entire file
        with open(filename, 'r', encoding='utf-8') as f:
            header = f.readline()  # Skip header
            lines = f.readlines()
            
        # Find and update the matching line
        updated = False
        new_lines = [header]  # Start with header
        
        for line in lines:
            try:
                state_0, state_1, state_2, act, q_val = line.strip().split(',')
                current_state = (float(state_0), float(state_1), float(state_2))
                
                if current_state == state and act == action_type:
                    # Calculate new Q-value
                    old_q = float(q_val)
                    if not next_available_actions:
                        next_max_q = 0.0
                    else:
                        next_q_values = [get_q_value_from_file(filename, next_state, next_action, action_obj)
                                        for next_action in next_available_actions]
                        next_max_q = max(next_q_values) if next_q_values else 0.0

                    # Q-learning update formula
                    new_q = old_q + alpha * (reward + gamma * next_max_q - old_q)
                    new_q_rounded = round(new_q, 2)
                    
                    new_line = f'({state[0]:.1f}, {state[1]:.1f}, {state[2]:.1f}),{action_type},{new_q_rounded:.2f}\n'
                    new_lines.append(new_line)
                    updated = True
                    print(f"Updated Q-value in file: {new_line.strip()}")
                else:
                    new_lines.append(line)
            except (ValueError, IndexError) as e:
                print(f"Error parsing line '{line.strip()}': {e}")
                new_lines.append(line)
                continue
        
        # Write all lines back to file
        with open(filename, 'w') as f:
            f.writelines(new_lines)
        
        if not updated:
            print(f"Warning: No matching state-action pair found for update")
        return updated
    
    except Exception as e:
        print(f"Error updating Q-table: {e}")
        return False

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