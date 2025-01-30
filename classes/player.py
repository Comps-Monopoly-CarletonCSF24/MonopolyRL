from typing import List
from classes.action import Action
from classes.board import Board, Property
from classes.dice import Dice
from classes.log import Log
from settings import GameSettings
from classes.state import State, group_cell_indices
from classes.player_logistics import Player
from classes.approx_ql import ApproxQLearningAgent
from classes.rewards import Reward

class Fixed_Policy_Player(Player):
    def handle_action(self, board, players, dice, log):
        # Trade with other players. Keep trading until no trades are possible
        # while self.do_a_two_way_trade(players, board, log):
        #     pass
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
                log.add(f"{self.name} built {ordinal[cell_to_improve.has_houses]} " +
                        f"house on {cell_to_improve} for ${cell_to_improve.cost_house}")

            # Building a hotel
            elif cell_to_improve.has_houses == 4:
                cell_to_improve.has_houses = 0
                cell_to_improve.has_hotel = 1
                board.available_houses += 4
                board.available_hotels -= 1
                # Paying for the improvement
                self.money -= cell_to_improve.cost_house
                log.add(f"{self.name} built a hotel on {cell_to_improve}")

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

class Approx_q_agent(Player):
    def __init__(self, name, settings):
        super().__init__(name, settings)
        self.action_object = Action()
        self.agent = ApproxQLearningAgent(name = name, settings=GameSettings, feature_size=150) 
        pass
    
    def handle_action(self, board: Board, players: List[Player], dice: Dice, log: Log):
        for group_idx in range(len(group_cell_indices)):
            action = self.take_one_action(board,players, group_idx)
            self.execute_action(board, log, action, group_idx)
        # self.agent.plot_q_values()

    def take_one_action(self, board: Board, players: List[Player], group_idx):
        """Moved agent.take_turn to here. The agent takes a turn and performs 
        all possible actiosn according to the NN.

        Args:
            board (Board): _description_
            players (_type_): _description_
            dice (_type_): _description_
            log (_type_): _description_
        """
        
        current_player = self
        current_state = State(current_player=current_player, players= players)
        action_index_in_small_list, action_index_in_bigger_list = self.agent.select_action(current_state)

        next_state = self.agent.simulate_action(board, current_state, current_player, players, action_index_in_bigger_list, group_idx)
        # If the action is not doable, you need to choose a different action and simulate that.
        # In fact, choose the next best action and simulate that. 

        reward = Reward().get_reward(current_player, players)
        self.agent.update(current_state, action_index_in_bigger_list, reward, next_state)
        actions = self.action_object.actions
        return actions[action_index_in_small_list]
        
    def execute_action(self, board: Board, log: Log, action: Action, group_idx):
        """Executes the action on the given property for the specified player.

        Args:
            board (Board): _description_
            log (Log): _description_
            action (Action): _description_
            group_idx (_type_): _description_
        """
        if action == 'buy':
            self.buy_in_group(group_idx, board, log)
            pass
        elif action == 'sell':
            self.sell_in_group(group_idx, board, log)
            pass
        elif action == 'do_nothing':
            pass

        return
    
    def buy_in_group(self, group_idx: int, board: Board, log: Log):

        self.unmortgage_a_property(board, log)
        cells_in_group = []
        for cell_idx in group_cell_indices[group_idx]:
            cells_in_group.append(board.cells[cell_idx])
            
        def can_buy_property(property_to_buy):
            '''
            Check if the player can buy a property
            '''
            if not isinstance(property_to_buy, Property):
                return False
            if property_to_buy.owner != None:
                return False
            if self.money - property_to_buy.cost_base < self.settings.unspendable_cash:
                return False
            return True
        
        def buy_property(property_to_buy):
            ''' Player buys the property'''
            log.add(f"Agent bought property : {property_to_buy.name}")
            property_to_buy.owner = self
            self.owned.append(property_to_buy)
            self.money -= property_to_buy.cost_base
            return True
        
        def get_next_property_to_improve():
            ''' Decide what is the next property to improve:
            - it should be eligible for improvement (is monopoly, not mortgaged,
            has not more houses than other cells in the group)
            - start with cheapest
            '''
            can_be_improved = []
            for cell in cells_in_group:
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
                            if cell.owner.name == self.name:
                                can_be_improved.append(cell)
            # Sort the list by the cost of house
            can_be_improved.sort(key = lambda x: x.cost_house)
            
            # Return first (the cheapest) property that can be improved
            if can_be_improved:
                return can_be_improved[0]
            return None
    
        def improve_property(cell_to_improve):
            if not cell_to_improve or cell_to_improve.owner != self:
                return False
            
            improvement_cost = cell_to_improve.cost_house

            # Don't do it if you don't have money to spend
            if self.money - improvement_cost < self.settings.unspendable_cash:
                return False

            # Building a house
            ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4:"4th"}

            if cell_to_improve.has_houses != 4 and cell_to_improve.owner.name == self.name:
                cell_to_improve.has_houses += 1
                board.available_houses -= 1
                # Paying for the improvement
                self.money -= cell_to_improve.cost_house
                log.add(f"{self.name} built {ordinal[cell_to_improve.has_houses]} " +
                        f"house on {cell_to_improve} for ${cell_to_improve.cost_house}")

            # Building a hotel
            elif cell_to_improve.has_houses == 4 and cell_to_improve.owner == self:
                print (f"building a hotel: {cell_to_improve.owner} with {self.name}")
                cell_to_improve.has_houses = 0
                cell_to_improve.has_hotel = 1
                board.available_houses += 4
                board.available_hotels -= 1
                # Paying for the improvement
                self.money -= cell_to_improve.cost_house
                log.add(f"{self.name} built a hotel on {cell_to_improve}")
            return True
        
        # if landed on an unowned property: buy it

        property_to_buy = board.cells[self.position]
        if can_buy_property(property_to_buy):
            buy_property(property_to_buy)
            board.recalculate_monopoly_coeffs(property_to_buy)
        
        while True:
           
            cell_to_improve = get_next_property_to_improve()
            if not improve_property(cell_to_improve):
                break
    
    def sell_in_group(self, group_idx: int, board: Board, log: Log):
        cells_in_group = []
        for cell_idx in group_cell_indices[group_idx]:
            cells_in_group.append(board.cells[cell_idx])

        def get_next_property_to_downgrade(cells_in_group):
            ''' Decide what is the next property to downgrade:
            - start with most developed properties
            - must maintain even building
            '''
            can_be_downgraded = []
            for cell in cells_in_group:
                if cell.owner == self:
                    if cell.has_hotel > 0 or cell.has_houses > 0:
                        for other_cell in board.groups[cell.group]:
                            if cell.has_hotel == 0 and (other_cell.has_houses > cell.has_houses or \
                            other_cell.has_hotel > 0):
                                break
                        else:
                            can_be_downgraded.append(cell)
                            
            # Sort by development level (hotel first, then most houses)
            can_be_downgraded.sort(key=lambda x: (x.has_hotel * 5 + x.has_houses), reverse=True)
            return can_be_downgraded[0] if can_be_downgraded else None
        
        def downgrade_property(cell_to_downgrade):
            if not cell_to_downgrade or cell_to_downgrade.owner != self:
                return False

            if cell_to_downgrade.has_hotel:
                # Convert hotel back to 4 houses if enough houses are available
                if board.available_houses >= 4:
                    cell_to_downgrade.has_hotel = False
                    cell_to_downgrade.has_houses = 4
                    board.available_hotels += 1
                    board.available_houses -= 4
                    sell_price = cell_to_downgrade.cost_house // 2
                    self.money += sell_price
                    return True
                return False  # Not enough houses to downgrade the hotel

            elif cell_to_downgrade.has_houses > 0:
                if any(prop.has_houses > cell_to_downgrade.has_houses for prop in cells_in_group):
                    return False  # Prevent unbalanced house selling

                cell_to_downgrade.has_houses -= 1
                board.available_houses += 1
                sell_price = cell_to_downgrade.cost_house // 2
                self.money += sell_price
                return True
            return False

        # First try to sell buildings if any exist
        cell_to_downgrade = get_next_property_to_downgrade(cells_in_group)
        if cell_to_downgrade:
            return downgrade_property(cell_to_downgrade)
        
        # If no buildings to sell, try to sell property
        
        return True  

    def is_group_actionable(self, group_idx: int, board: Board):
        cell_indices_in_group = group_cell_indices[group_idx]
        for cell_idx in cell_indices_in_group:
            cell = board.cells[cell_idx]
            # can sell
            if cell.owner == self: 
                return True
            # can buy
            if self.position == cell_idx and not cell.owner:
                return True
            # can improve
            if cell.monopoly_coef == 2:
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
        
class BasicQPlayer(Player):
    pass