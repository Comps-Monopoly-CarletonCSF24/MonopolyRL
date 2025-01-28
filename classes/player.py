from typing import List
import random
import numpy as np
from classes.board import Board, Property
from classes.dice import Dice
from classes.log import Log
from classes.action import Action
from settings import GameSettings
from classes.state import State
from classes.player_logistics import Player

class Fixed_Policy_Player(Player):
    def handle_action(self, board, players, dice, log):
        # Trade with other players. Keep trading until no trades are possible
        while self.do_a_two_way_trade(players, board, log):
            pass
        # Unmortgage a property. Keep doing it until possible
        while self.unmortgage_a_property(board, log):
            pass
        # Improve all properties that can be improved
        self.improve_properties(board, log)
        # Player lands on a property
        if isinstance(board.cells[self.position], Property):
            self.handle_buying_property(board, players, log)

    def do_a_two_way_trade(self, players, board, log):
        ''' Look for and perform a two-way trade
        '''

        def get_price_difference(gives, receives):
            ''' Calculate price difference between items player
            is about to give minus what he is about to receive.
            >0 means player gives away more
            Return both absolute (in $), relative for a giver, relative for a receiver
            '''

            cost_gives = sum(cell.cost_base for cell in gives)
            cost_receives = sum(cell.cost_base for cell in receives)

            diff_abs = cost_gives - cost_receives

            diff_giver, diff_receiver = float("inf"), float("inf")
            if receives:
                diff_giver = cost_gives / cost_receives
            if gives:
                diff_receiver = cost_receives / cost_gives

            return diff_abs, diff_giver, diff_receiver

        def remove_by_color(cells, color):
            new_cells = [cell for cell in cells if cell.group != color]
            return new_cells

        def fair_deal(player_gives, player_receives, other_player):
            ''' Remove properties from to_sell and to_buy to make it as fair as possible
            '''

            # First, get all colors in both sides of the deal
            color_receives = [cell.group for cell in player_receives]
            color_gives = [cell.group for cell in player_gives]

            # If there are only properties from size-2 groups, no trade
            both_colors = set(color_receives + color_gives)
            if both_colors.issubset({"Utilities", "Indigo", "Brown"}):
                return [], []

            # Look at "Indigo", "Brown", "Utilities". These have 2 properties,
            # so both players would want to receive them
            # If they are present, remove it from the guy who has longer list
            # If list has the same length, remove both questionable items

            for questionable_color in ["Utilities", "Indigo", "Brown"]:
                if questionable_color in color_receives and questionable_color in color_gives:
                    if len(player_receives) > len(player_gives):
                        player_receives = remove_by_color(player_receives, questionable_color)
                    elif len(player_receives) < len(player_gives):
                        player_gives = remove_by_color(player_gives, questionable_color)
                    else:
                        player_receives = remove_by_color(player_receives, questionable_color)
                        player_gives = remove_by_color(player_gives, questionable_color)

            # Sort, starting from the most expensive
            player_receives.sort(key=lambda x: -x.cost_base)
            player_gives.sort(key=lambda x: -x.cost_base)

            # Check the difference in value and make sure it is not larger that player's preference
            while player_gives and player_receives:

                diff_abs, diff_giver, diff_receiver = \
                    get_price_difference(player_gives, player_receives)

                # This player gives too much
                if diff_abs > self.settings.trade_max_diff_abs or \
                diff_giver > self.settings.trade_max_diff_rel:
                    player_gives.pop()
                    continue
                # Other player gives too much
                if -diff_abs > other_player.settings.trade_max_diff_abs or \
                diff_receiver > other_player.settings.trade_max_diff_rel:
                    player_receives.pop()
                    continue
                break

            return player_gives, player_receives

        for other_player in players:
            # Selling/buying thing matches
            if self.wants_to_buy.intersection(other_player.wants_to_sell) and \
               self.wants_to_sell.intersection(other_player.wants_to_buy):
                player_receives = list(self.wants_to_buy.intersection(other_player.wants_to_sell))
                player_gives = list(self.wants_to_sell.intersection(other_player.wants_to_buy))

                # Work out a fair deal (don't trade same color,
                # get value difference within the limit)
                player_gives, player_receives = \
                    fair_deal(player_gives, player_receives, other_player)

                # If their deal is not empty, go on
                if player_receives and player_gives:

                    # Price difference in traded properties
                    price_difference, _, _ = \
                        get_price_difference(player_gives, player_receives)

                    # Player gives await more expensive item, other play has to pay
                    if price_difference > 0:
                        # Other guy can't pay
                        if other_player.money - price_difference < \
                           other_player.settings.unspendable_cash:
                            return False
                        other_player.money -= price_difference
                        self.money += price_difference

                    # Player gives cheaper stuff, has to pay
                    if price_difference < 0:
                        # This player can't pay
                        if self.money - abs(price_difference) < \
                           self.settings.unspendable_cash:
                            return False
                        other_player.money += abs(price_difference)
                        self.money -= abs(price_difference)

                    # Property changes hands
                    for cell_to_receive in player_receives:
                        cell_to_receive.owner = self
                        self.owned.append(cell_to_receive)
                        other_player.owned.remove(cell_to_receive)
                    for cell_to_give in player_gives:
                        cell_to_give.owner = other_player
                        other_player.owned.append(cell_to_give)
                        self.owned.remove(cell_to_give)

                    # Log the trade and compensation payment
                    log.add(f"Trade: {self} gives {[str(cell) for cell in player_gives]}, " +
                            f"receives {[str(cell) for cell in player_receives]} " +
                            f"from {other_player}")

                    if price_difference > 0:
                        log.add(f"{self} received " +
                                f"price difference compensation ${abs(price_difference)} " + 
                                f"from {other_player}")
                    if price_difference < 0:
                        log.add(f"{other_player} received " +
                                f"price difference compensation ${abs(price_difference)} " +
                                f"from {self}")

                    # Recalculate monopoly and improvement status
                    board.recalculate_monopoly_coeffs(player_gives[0])
                    board.recalculate_monopoly_coeffs(player_receives[0])

                    # Recalculate who wants to buy what
                    # (for all players, it may affect their decisions too)
                    for player in players:
                        player.update_lists_of_properties_to_trade(board)

                    # Return True, to run trading function again
                    return True
        return False
    
    def unmortgage_a_property(self, board, log):
        ''' Go through the list of properties and unmortgage one,
        if there is enough money to do so. Return True, if any unmortgaging
        took place (to call it again)
        '''

        for cell in self.owned:
            if cell.is_mortgaged:
                cost_to_unmortgage = \
                    cell.cost_base * GameSettings.mortgage_value + \
                    cell.cost_base * GameSettings.mortgage_fee
                if self.money - cost_to_unmortgage >= self.settings.unspendable_cash:
                    log.add(f"{self} unmortgages {cell} for ${cost_to_unmortgage}")
                    self.money -= cost_to_unmortgage
                    cell.is_mortgaged = False
                    self.update_lists_of_properties_to_trade(board)
                    return True

        return False
    
    def improve_properties(self, board, log):
        ''' While there is money to spend and properties to improve,
        keep building houses/hotels
        '''

        def get_next_property_to_improve():
            ''' Decide what is the next property to improve:
            - it should be eligible for improvement (is monopoly, not mortgaged,
            has not more houses than other cells in the group)
            - start with cheapest
            '''
            can_be_improved = []
            for cell in self.owned:
                # Property has to be:
                # - not maxed out (no hotel)
                # - not mortgaged
                # - a part of monopoly, but not railway or utility (so the monopoly_coef is 2)
                if cell.has_hotel == 0 and not cell.is_mortgaged and cell.monopoly_coef == 2 \
                    and not (cell.group == "Railroads" or cell.group == "Utilities") :
                    # Look at other cells in this group
                    # If they have fewer houses, this cell can not be improved
                    # If any cells in the group is mortgaged, this cell can not be improved
                    for other_cell in board.groups[cell.group]:
                        if other_cell.has_houses < cell.has_houses or other_cell.is_mortgaged:
                            break
                    else:
                        # Make sure there are available houses/hotel for this improvement
                        if cell.has_houses != 4 and board.available_houses > 0 or \
                           cell.has_houses == 4 and board.available_hotels > 0:
                            can_be_improved.append(cell)
            # Sort the list by the cost of house
            can_be_improved.sort(key = lambda x: x.cost_house)

            # Return first (the cheapest) property that can be improved
            if can_be_improved:
                return can_be_improved[0]
            return None

        while True:
            cell_to_improve = get_next_property_to_improve()

            # Nothing to improve anymore
            if cell_to_improve is None:
                break

            improvement_cost = cell_to_improve.cost_house

            # Don't do it if you don't have money to spend
            if self.money - improvement_cost < self.settings.unspendable_cash:
                break

            # Building a house
            ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4:"4th"}

            if cell_to_improve.has_houses != 4:
                cell_to_improve.has_houses += 1
                board.available_houses -= 1
                # Paying for the improvement
                self.money -= cell_to_improve.cost_house
                log.add(f"{self} built {ordinal[cell_to_improve.has_houses]} " +
                        f"house on {cell_to_improve} for ${cell_to_improve.cost_house}")

            # Building a hotel
            elif cell_to_improve.has_houses == 4:
                cell_to_improve.has_houses = 0
                cell_to_improve.has_hotel = 1
                board.available_houses += 4
                board.available_hotels -= 1
                # Paying for the improvement
                self.money -= cell_to_improve.cost_house
                log.add(f"{self} built a hotel on {cell_to_improve}")
    
    def handle_buying_property(self, board, players, log):
        ''' Landing on property: either buy it or pay rent
        '''

        def is_willing_to_buy_property(property_to_buy):
            ''' Check if the player is willing to buy an unowned property
            '''
            # Player has money lower than unspendable minimum
            if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
                return False

            # Player does not have enough money
            # If unspendable_cash >= 0 this check is redundant
            # However we'll need to think if a "mortgage to buy" situation
            if property_to_buy.cost_base > self.money:
                return False

            # Property is in one of the groups, player chose to ignore
            if property_to_buy.group in self.settings.ignore_property_groups:
                return False

            # Nothing stops the player from making a purchase
            return True

        def buy_property(property_to_buy):
            ''' Player buys the property
            '''
            property_to_buy.owner = self
            self.owned.append(property_to_buy)
            self.money -= property_to_buy.cost_base
            
        # This is the property a player landed on
        landed_property = board.cells[self.position]

        # Property is not owned by anyone
        if landed_property.owner is None:
            
            # Does the player want to buy it?
            if is_willing_to_buy_property(landed_property):
                # Buy property
                buy_property(landed_property)
                log.add(f"Player {self.name} bought {landed_property} " +
                        f"for ${landed_property.cost_base}")

                # Recalculate all monopoly / can build flags
                board.recalculate_monopoly_coeffs(landed_property)

                # Recalculate who wants to buy what
                # (for all players, it may affect their decisions too)
                for player in players:
                    player.update_lists_of_properties_to_trade(board)

            else:
                log.add(f"Player {self.name} landed on a {landed_property}, he refuses to buy it")
                # TODO: Bank auctions the property
        
class BasicQPlayer(Player):

    def __init__(self, name, settings, position=0, money=1500):
        super().__init__(name, settings)
        self.qTable = {}
        self.alpha = 0.4  # increase
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.4  # increased
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = 0

        #desperate to get this working
        self.is_willing_to_buy_property = True
        self.action_obj=Action() #create an action object
        self.min_money = 100 #set a lowest amount of money
    
    def get_state(self, board, players):
        """gets the state of the player
        """
    
        current_property = board.cells[self.position]

        is_property = isinstance(current_property, Property)
        can_afford = is_property and self.money >= current_property.cost_base
        property_value = current_property.cost_base if is_property else 0
        money_ratio = min(self.money / 2000, 1.0)

        #count properties owned in the same color group
        same_color_properties = 0
        if is_property and hasattr(current_property, 'color'):
            for prop in self.owned:
                if hasattr(prop, 'color') and prop.color == current_property.color:
                    same_color_properties += 1
        return (float(same_color_properties), float(can_afford), round(money_ratio,2), same_color_properties)
    
    
    def choose_action(self, board,state, available_actions):
        '''
        chooses an action from the available actions. Big pereference buying properties
        '''
        if not available_actions:
            return None
        current_property = board.cells[self.position]
        is_property = isinstance(current_property, Property)
        # Always try to buy if:
        # 1. It's a property
        # 2. We can afford it
        # 3. We have more than minimum cash reserve
        if (is_property and 
            current_property.owner is None and 
            self.money >= current_property.cost_base + self.min_money):
            
            buy_actions = [a for a in available_actions 
                          if self.action_obj.actions[a % len(self.action_obj.actions)] == 'buy']
            if buy_actions:
                return buy_actions[0]
        
        # Normal exploration/exploitation for other cases
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in available_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(available_actions, q_values) 
                          if q == max_q]
            return np.random.choice(best_actions)
    
    
    def get_q_value(self, state, action):
        '''
        gets the qvalues from the qtable
        '''
        
        return self.qTable.get((state,action), 0.0)
    
    def update_q_value(self, state, action, reward, next_state, next_available_actions):
        '''
        updates the qtable by adding better qvalues depending on the most recent move, 
        if it had a better reward than the previous one
        '''
        
        #convert state and nextState to tuples if they are not already
        if not next_available_actions:
            next_max_q = 0
        else:
            next_q_values = [self.get_q_value(next_state, next_action) for next_action in next_available_actions]
            next_max_q = max(next_q_values) if next_q_values else 0
        
        #q-learning update formula
        old_q = self.get_q_value(state, action)

        #if old_q is None, initialize it to 0
        if old_q is None:
            old_q = 0.0
        reward = reward if reward is not None else 0.0
        next_max_q = next_max_q if next_max_q is not None else 0.0
        
        if next_max_q is None or reward is None:
            raise ValueError("next_max_q or reward must not be None")
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.qTable[(state, action)] = new_q
        
    def calculate_reward(self, board, players):
        """Calculate the reward based on the player's current state."""
        reward = 0
        #reward for money
        reward += self.money * 0.01

        
        for prop in self.owned:
            #reward for properties owned
            reward += len(self.owned) * 0.5
            if hasattr(prop, 'color'):  # Only count colored properties
                same_color_count = sum(
                    1 for other_prop in self.owned
                    if hasattr(other_prop, 'color') and other_prop.color == prop.color
                )
                reward += same_color_count * 100
                reward += 50 #immediate reward for owning a property        
        # Huge reward for monopolies
        for group in board.groups.values():
            if all(prop.owner == self for prop in group):
                reward += 1000  # Increased from 50
                
        # Penalty for having very low cash reserves
        #if self.money < 100:  # Add penalty if cash is too low
           #reward -= 200
            
        return reward
    
    def make_a_move(self, board, players, dice, log):
        #call parent's make_a_move to handle the dice roll
        result = super().make_a_move(board, players, dice, log)
        if result == "bankrupt" or result == "move is over":
            return result
   
        # Q learning execution
        # Exploration: choose random action
    
        current_state = self.get_state(board, players)
        #get available actions
        action_obj = Action()
        available_actions = []

        for action_idx in range(len(action_obj.actions)):
            property_idx, action_type = action_obj.map_action_index(action_idx)
            if action_obj.is_excutable(self, board, property_idx, action_type):
                available_actions.append(action_idx)
        
        #choose action
        chosen_action = self.choose_action(board,current_state, available_actions)
        
        #execute action
        property_idx, action_type = action_obj.map_action_index(chosen_action)
        action_obj.execute_action(self, board, property_idx, action_type, log)

        #calculate reward
        reward = self.calculate_reward(board, players)

       #update q-table if we have a previous state-action pair
        if self.previous_state is not None:
            self.update_q_value(
                self.previous_state,
                self.previous_action,
                self.previous_reward,
                current_state,
                available_actions
            )
       

       #store current state and action pair for next iteration
        self.previous_state = current_state
        self.previous_action = chosen_action
        self.previous_reward = reward

        self.log_q_table()

        return "continue"
    def should_buy_property(self, property, board):
        """Determine if the agent should buy a property"""
        if not property or not isinstance(property, Property):
            return False
            
        if property.owner is not None:
            return False
            
        if self.money < property.cost_base + self.min_cash_reserve:
            return False
            
        # Count properties in same color group
        same_color_count = 0
        if hasattr(property, 'color'):
            for prop in self.owned:
                if hasattr(prop, 'color') and prop.color == property.color:
                    same_color_count += 1
                    
        # More likely to buy if we have properties of same color
        if same_color_count > 0:
            return True
            
        # Always buy railroads and utilities
        if property.group in ['Railroads', 'Utilities']:
            return True
            
        # Buy if price is reasonable compared to our money
        return property.cost_base <= self.money * 0.4
    def buy_property(self, property_to_buy, log):
        """Buy a property and update the game state
    
    Args:
        property_to_buy (Property): The property being purchased
        log (Log): Game log instance
    """
        if self.is_willing_to_buy_property:
            # Update property ownership
            property_to_buy.owner = self
            self.owned.append(property_to_buy)
            self.money -= property_to_buy.cost_base
            
            # Log the purchase
            log.add(f"Player {self.name} bought {property_to_buy} " +
                        f"for ${property_to_buy.cost_base}")
            
            # Recalculate monopoly / can build flags
            #board.recalculate_monopoly_coeffs(property_to_buy)

            # Recalculate who wants to buy what
            #for player in players:
                #player.update_lists_of_properties_to_trade(board)
        else:
            log.add(f"Player {self.name} landed on {property_to_buy}, but refuses to buy it")
            # TODO: Bank auctions the property

    def log_q_table(self):
        """log q table to a file"""
        with open(f"qtable_{self.name}.txt", "w") as f:
            f.write(f"Q-table for {self.name}:\n\n")
            f.write("State-Action Pairs with non-zero Q-values:\n\n")
            
            # Sort Q-table by Q-value for better readability
            sorted_q = sorted(self.qTable.items(), key=lambda x: x[1], reverse=True)
            
            for (state, action), q_value in sorted_q:
                if q_value != 0:  # Only show non-zero Q-values
                    f.write(f"State: {state}\n")
                    property_idx = action // 3  # Assuming 3 actions per property
                    action_type = ['buy', 'sell', 'do_nothing'][action % 3]
                    f.write(f"Action: Property {property_idx}, {action_type}\n")
                    f.write(f"Q-value: {q_value:.2f}\n\n")
            

class DQAPlayer(Player):
    pass