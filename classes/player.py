from typing import List
import random
import numpy as np
from classes.board import Board,Property
from classes.dice import Dice
from classes.log import Log
from classes.action import Action
from settings import GameSettings, StandardPlayer
from classes.state import get_state, is_property, has_monopoly, has_more_money, State, group_cell_indices
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
                if diff_abs > StandardPlayer.trade_max_diff_abs or \
                diff_giver > StandardPlayer.trade_max_diff_rel:
                    player_gives.pop()
                    continue
                # Other player gives too much
                if -diff_abs > StandardPlayer.trade_max_diff_abs or \
                diff_receiver > StandardPlayer.trade_max_diff_rel:
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
                           StandardPlayer.unspendable_cash:
                            return False
                        other_player.money -= price_difference
                        self.money += price_difference

                    # Player gives cheaper stuff, has to pay
                    if price_difference < 0:
                        # This player can't pay
                        if self.money - abs(price_difference) < \
                           StandardPlayer.unspendable_cash:
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
                if self.money - cost_to_unmortgage >= StandardPlayer.unspendable_cash:
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
            if self.money - improvement_cost < StandardPlayer.unspendable_cash:
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
            if self.money - property_to_buy.cost_base < StandardPlayer.unspendable_cash:
                return False

            # Player does not have enough money
            # If unspendable_cash >= 0 this check is redundant
            # However we'll need to think if a "mortgage to buy" situation
            if property_to_buy.cost_base > self.money:
                return False

            # Property is in one of the groups, player chose to ignore
            if property_to_buy.group in StandardPlayer.ignore_property_groups:
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
                # Trigger auction
                self.auction_property(landed_property, players, log)
                return "end turn"

class BasicQPlayer(Player):

    def __init__(self, name, settings, position=0, money=1500):
        super().__init__(name, settings)
        self.qTable = {}
        self.alpha = 0.1  
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.4  # increased
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = 0

       
        self.is_willing_to_buy_property = True
        self.action_obj= Action() #create an action object
        self.min_money = 100 #set a lowest amount of money

    def get_state(self, board, players):
        """gets the state of the player
        """
        current_property = board.cells[self.position]
        #property feature
        is_property_val = is_property(self,current_property)
        has_monopoly_val = has_monopoly(self,board, self.position)
        has_more_money_val = has_more_money(self,players)

        state = tuple(get_state(
            has_monopoly = has_monopoly_val,
            is_property = is_property_val,
            has_more_money = has_more_money_val)

        )

        return state
    
    def train_agent_with_one_action(self, board: Board, players: List[Player], current_state: State, current_action: Action):
        """update q-value based on action results"""
        #calculate reward for the action taken
        reward = self.calculate_reward(board, players)
        
        #get next state and available actions for q-learning update
        next_state = self.get_state(board, players)
        next_available_actions = self.get_available_actions(board)
        
        #update Q-table if we have a previous state-action pair
        if self.previous_state is not None:
            self.update_q_value(
                self.previous_state,
                self.previous_action,
                self.previous_reward,
                current_state,
                next_available_actions
            )
        
        #store current state/action/reward for next iteration
        self.previous_state = current_state
        self.previous_action = current_action
        self.previous_reward = reward

    def get_available_actions(self, board: Board):
        """get available actions for the current state"""
        available_actions = []

        for action_idx in range(self.action_obj.total_actions):
            try:
                property_idx, action_type = self.action_obj.map_action_index(action_idx)

                if property_idx >= len(board.cells):
                    print(f"Invalid property index: {property_idx}")
                    continue
                #check if the action is executable
                if self.action_obj.is_excutable(self, board, property_idx, action_type):
                    available_actions.append(action_idx)
                
            except Exception as e:
                print(f"Error checking action {action_idx}: {e}")
                continue
        return available_actions
            
    def would_complete_monopoly(self, property):
        
        #checks if the property would complete a monopoly
        
        if not hasattr(property, 'color'):
            return False
        color = property.color
        same_color_props = sum(1 for prop in self.owned if hasattr(prop, 'color') and prop.color == color)
        total_in_color = sum(1 for prop in board.properties if hasattr(prop, 'color') and prop.color == color)
        
        return len(same_color_props) == total_in_color - 1
    
    def buy_in_group(self, group_idx: int, board: Board, players: List[Player], log: Log):
        cells_in_group = []
        for cell_idx in group_cell_indices[group_idx]:
            cells_in_group.append(board.cells[cell_idx])
        
        def get_next_property_to_unmortgage():
            for cell in cells_in_group:
                if not cell.is_mortgaged:
                    continue
                if not cell.owner:
                    continue
                cost_to_unmortage = \
                    cell.cost_base * GameSettings.mortgage_value + \
                    cell.cost_base * GameSettings.mortgage_fee
                if not self.money - cost_to_unmortage < self.settings.unspendable_cash:
                    continue
                return cell, cost_to_unmortage
            return None, None
        
        def unmortgage_property(property_to_unmortgage, cost_to_unmortage):
            log.add(f"{self.name} unmortgages {property_to_unmortgage} for ${cost_to_unmortage}")
            self.money -= cost_to_unmortage
            property_to_unmortgage.is_mortgaged = False
            self.update_lists_of_properties_to_trade(board)
            return True
        
        def can_buy_property():
            '''check if the player can buy a property'''
            property_to_buy = board.cells[self.position]
            if not self.position in group_cell_indices[group_idx]:
                return False, None
            if not isinstance(property_to_buy, Property):
                return False, None
            if property_to_buy.owner != None:
                return False, None
            if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
                return False, None
            return True, property_to_buy
        
        def buy_property(property_to_buy):
            '''Player buys the property'''
            if property_to_buy is None:
                return False
            property_to_buy.owner = self
            self.owned.append(property_to_buy)
            self.money -= property_to_buy.cost_base
            log.add(f"Player {self.name} bought {property_to_buy} " +
                   f"for ${property_to_buy.cost_base}")
            board.recalculate_monopoly_coeffs(property_to_buy)
            #recalculate who wants to buy what
            for player in players:
                player.update_lists_of_properties_to_trade(board)
            return True
        
        # First priority: unmortgage
        cell_to_unmortgage, cost_to_unmortgage = get_next_property_to_unmortgage()
        if cell_to_unmortgage:
            return unmortgage_property(cell_to_unmortgage, cost_to_unmortgage)
        
        # Second priority: buy property
        can_buy, property_to_buy = can_buy_property()
        if can_buy:
            return buy_property(property_to_buy)
        
        return False
            
    def sell_in_group(self, group_idx: int, board: Board, log: Log):
        cells_in_group = []
        for cell_idx in group_cell_indices[group_idx]:
            cells_in_group.append(board.cells[cell_idx])
        
        def get_next_property_to_sell():
            """see if there is a property to sell"""
            for cell in cells_in_group:
                if not isinstance(cell, Property):
                    continue
                if cell.owner != self:
                    continue
                if cell.is_mortgaged:
                    continue
                return cell
            return None
        
        def mortage_property(property_to_mortgage):
            """mortage a property"""
            mortage_price = property_to_mortgage.cost_base * GameSettings.mortgage_value
            self.money += mortage_price
            log.add(f"{self.name} mortages {property_to_mortgage}, raising ${mortage_price}")
            return True
        
        def get_next_property_to_downgrade():
            """decide what is the next property to downgrade:
                - start with most developed properties
                - must maintain even building"""
            can_be_downgraded = []
            for cell in cells_in_group:
                if cell.ownder == self:
                    if cell.has_hotel == 1 or cell.has_houses > 0:
                        #look at other cells in the group to maintain even building
                        for other_cell in board.groups[cell.group]:
                            if other_cell.has_houses > cell.has_houses:
                                break
                        else:
                            can_be_downgraded.append(cell)
            #Sort by development level (hotel first, then most houses)
            can_be_downgraded.sort(key = lambda x: (x.has_hotel * 5 + x.has_houses), reverse=True)
            return can_be_downgraded[0] if can_be_downgraded else None
        
        def downgrade_property(property_to_downgrade):
            if not property_to_downgrade:
                return False
            
            if property_to_downgrade.has_hotel == 1:
                #convert hotel back to 4 houses if possible
                if board.available_houses >= 4:
                    property_to_downgrade.has_hotel = 0
                    property_to_downgrade.has_houses = 4
                    board.available_hotels += 1
                    board.available_houses -= 4
                    sell_price = 
    


    
    def is_group_actionable(self, group_idx: int, board: Board):
        cell_indices_in_group = group_cell_indices[group_idx]
        for cell_idx in cell_indices_in_group:
            cell = board.cells[cell_idx]
            #can sell
            if cell.owner == self:
                return True
            #call buy
            if self.position == cell_idx and not cell.owner:
                return True
            #can improve
            if cell.monopoly_coef == 2:
                return True
        return False
    
    def select_action(self, board: Board, players: List[Player]) -> tuple:
        """high level method that gets state and selects an action"""
        current_state = self.get_state(board, players)
        available_actions = self.get_available_actions(board)
        action = self._choose_action_strategy(board, current_state, available_actions)
        return current_state, action
    
    def _choose_action_strategy(self, board, state, available_actions):
        '''
        chooses an action from the available actions. Big pereference buying properties
        '''
        if not available_actions:
            return None
        # Debug prints to see what's happening
        

        current_property = board.cells[self.position]
        #print(f"Available actions: {available_actions}")
        #print(f"Current position: {self.position}")
        #print(f"Current money: {self.money}")
        is_property = isinstance(current_property, Property)
        #print(f"is_property: {is_property}")
        #has_monopoly, _, has_more_money = state
        if is_property and current_property.owner is None:
            #print(f"just checked no one owns: {current_property}")
            buy_actions = [a for a in available_actions 
                          if self.action_obj.actions[a % len(self.action_obj.actions)] == 'buy']
            #print(f"Buy actions available: {buy_actions}")
            if buy_actions and self.money - current_property.cost_base < self.settings.unspendable_cash:
                #always buy if it completes a monopoly
                if (
                    has_monopoly == 1.0 or has_more_money == 1.0 or 
                    
                    self.would_complete_monopoly(current_property) or 

                #always buy railraods or utilities of affordable
                    current_property.group in ['Railroads', 'Utilities'] and self.money - current_property.cost_base < self.settings.unspendable_cash or
                #property is relatively cheap to our money
                current_property.cost_base <= self.money * 0.2 or
                #buy if we have properties of the same color
                    (hasattr(current_property, 'color') and 
                    any(prop.color == current_property.color for prop in self.owned))
                ):
                    return buy_actions[0]
                

        # Normal exploration/exploitation for other cases
        if np.random.rand() < self.epsilon:
            if np.random.rand() < 0.8:
                buy_actions = [a for a in available_actions 
                               if self.action_obj.actions[a % len(self.action_obj.actions)] == 'buy']
                if buy_actions:
                    return np.random.choice(buy_actions)
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
        #if this is a buy action, start with a higher intial q value
        if self.action_obj.actions[action % len(self.action_obj.actions)] == 'buy':
            return self.qTable.get((state, action), 2000.0)
        
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
        
        current_state = self.get_state(board, players)
        has_monopoly, is_property, has_more_money = current_state

        #base reward 
        reward += self.money * 0.001
        reward += len(self.owned) * 20

        current_property = board.cells[self.position]
        if (isinstance(current_property, Property) and current_property.owner == None and 
        current_property.cost_base <= self.money - self.settings.unspendable_cash):
            penalty = -300
            if has_monopoly == 1.0:
                penalty *= 2
            if has_more_money == 1.0:
                penalty *= 2
            reward += penalty
            
        #reward for properties owned
        reward += len(self.owned) * 20 #increased base property reward
        #extra reward for properies in same color groups
        color_groups = {}
        for prop in self.owned:
            if hasattr(prop, 'color'):  # Only count colored properties
                if prop.color not in color_groups:
                    color_groups[prop.color] = 0
                color_groups[prop.color] += 1

                #progressive rewards for more properties in the same color
                reward += (color_groups[prop.color] **2 ) * 30
        
        # Huge reward for monopolies
        for group in board.groups.values():
            if all(prop.owner == self for prop in group):
                reward += 200

        #extra reward for developed monopolies
            if hasattr(group[0], 'houses'):
                houses = sum(prop.houses for prop in group)
                reward += houses * 50        
        #strategic positioning rewards
        try:
            opponent = [p for p in players if p!= self and not p.is_bankrupt][0]
        except IndexError:
            # Handle case where all other players are bankrupt
            # Return a default state or end game state
            return (0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0)
        #reward for having more properties than opponent
        prop_difference = len(self.owned) - len(opponent.owned)
        reward += prop_difference * 30

        #penalty for having fewer properties than opponent
        if len(self.owned) < len(opponent.owned):
            reward -= 500

        return reward    
    
    def calculate_max_bid(self, property_to_auction, current_bid):
        #more conservative for basic q-learning player
        return min(self.money * 0.5, property_to_auction.cost_base)
    
    def execute_action(self, board, players, log, property_idx, action_type):
        """executes a single action and returns success/failure"""
        
        self.action_successful = True
        try:
            # First check if player is in jail
            if self.in_jail:
                print(f"{self.name} is in jail, cannot execute actions")
                return False

            # Convert property_idx to actual board position
            actual_positions = [pos for group in group_cell_indices for pos in group]
            if property_idx >= len(actual_positions):
                print(f"Invalid property index: {property_idx}")
                return False
            
            board_position = actual_positions[property_idx]
            current_property = board.cells[board_position]

            if action_type == 'buy':
                # Check if the property is at player's current position
                if board_position != self.position:
                    print(f"Cannot buy property at position {board_position} when player is at position {self.position}")
                    return False

                # Find which group this property belongs to
                group_idx = None
                for idx, group in enumerate(group_cell_indices):
                    if board_position in group:
                        group_idx = idx
                        break
                
                if group_idx is not None:
                    success = self.buy_in_group(group_idx, board, players, log)
                    if success:
                        return True
                    else:
                        return False
                else:
                    print(f"Property at position {board_position} not found in any group!")
                    return False
                
            elif action_type == 'do_nothing':
                return True
            
            return False

        except Exception as e:
            self.action_successful = False
            log.add(f"{self.name} failed to execute action {action_type} on property {property_idx}: {e}")
            return False
    def make_a_move(self, board, players, dice, log):
        # Call parent's make_a_move to handle the dice roll
        result = super().make_a_move(board, players, dice, log)
        
        if result == "bankrupt" or result == "move is over":
            return result
        # Get state and action
        current_state, chosen_action = self.select_action(board, players)
        
        if chosen_action is None:
            return "continue"
            
        # 2.execute action
        property_idx, action_type = self.action_obj.map_action_index(chosen_action)
        #print(f"called execute_action {property_idx} {action_type}")
        success = self.execute_action(board, players, log, property_idx, action_type)

        # 3.train agent if action was successful
        if success:
            #print(f"called train_agent_with_one_action {current_state} {chosen_action}")
            self.train_agent_with_one_action(board, players, current_state, chosen_action)
        
        self.log_q_table()
        return "continue"
    
    def log_q_table(self):
        """log q table to a file"""
        with open(f"FILE:qtable_{self.name}.txt", "w") as f:
            f.write(f"Q-table for {self.name}:\n\n")
            f.write("State-Action Pairs with non-zero Q-values:\n\n")
            
            #group q values by proeprty idx
            property_groups = {}
            for (state,action), q_value in self.qTable.items():
                if q_value != 0:
                    property_idx = action //3
                    if property_idx not in property_groups:
                        property_groups[property_idx] = []
                    property_groups[property_idx].append((state, action, q_value))

            #write q values sorted by property index
            for property_idx in sorted(property_groups.keys()):
                    f.write(f"Property index: Property {property_idx}\n")
                    f.write("-" * 20 + "\n")

                    #sort actions for this property by q value
                    property_actions = sorted(property_groups[property_idx], key=lambda x: x[2], reverse=True)
            
                    for state, action, q_value in property_actions:
                        f.write(f"State: {state}\n")
                        action_type = ['buy','sell', 'do_nothing'][action % 3]
                        f.write(f"Action: {action_type}\n")
                        f.write(f"Q-value: {q_value:.2f}\n\n")

class DQAPlayer(Player):
    pass