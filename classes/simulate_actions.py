from classes.board import Board, Property
from classes.state import State, group_cell_indices
from classes.player_logistics import Player
import classes.state as s
import copy

def buy_in_group_simulation(group_idx: int, board: Board, player:Player, property: Property, Players):

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
            if player.money - property_to_buy.cost_base < player.settings.unspendable_cash:
                return False
            return True
        
        def buy_property(property_to_buy):
            ''' Player buys the property'''
            
            property_to_buy.owner = player
            player.owned.append(property_to_buy)
            player.money -= property_to_buy.cost_base
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
                            can_be_improved.append(cell)
            # Sort the list by the cost of house
            can_be_improved.sort(key = lambda x: x.cost_house)
            
            # Return first (the cheapest) property that can be improved
            if can_be_improved:
                return can_be_improved[0]
            return None
    
        def improve_property(cell_to_improve):
            if not cell_to_improve:
                return False
            
            improvement_cost = cell_to_improve.cost_house

            # Don't do it if you don't have money to spend
            if player.money - improvement_cost < player.settings.unspendable_cash:
                return False

            # Building a house
            ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4:"4th"}

            if cell_to_improve.has_houses != 4:
                cell_to_improve.has_houses += 1
                board.available_houses -= 1
                # Paying for the improvement
                player.money -= cell_to_improve.cost_house

            # Building a hotel
            elif cell_to_improve.has_houses == 4:
                cell_to_improve.has_houses = 0
                cell_to_improve.has_hotel = 1
                board.available_houses += 4
                board.available_hotels -= 1
                # Paying for the improvement
                player.money -= cell_to_improve.cost_house
            return True
        
        # if landed on an unowned property: buy it
        
        if can_buy_property(property):
            buy_property(property)
        
        while True:
                cell_to_improve = get_next_property_to_improve()
                status = improve_property(cell_to_improve)
                if not status:
                    break
        return True


def update_state_after_spending (group_idx: int, original_board: Board, original_player:Player, original_property: Property, players):
    player = copy.deepcopy(original_player)
    property = copy.deepcopy(original_property)
    board = copy.deepcopy(original_board)
    actions_status = buy_in_group_simulation(group_idx, board, player, property, players)
    if actions_status:
        new_state = s.get_state(s.get_area(player, players), 
            s.get_position(player.position), 
            s.get_finance(player, players))
    return new_state


def sell_in_group_simulation(group_idx: int, board: Board, player: Player):
    cells_in_group = []
    for cell_idx in group_cell_indices[group_idx]:
        cells_in_group.append(board.cells[cell_idx])
        
        
    def get_next_property_to_downgrade():
        ''' Decide what is the next property to downgrade:
        - start with most developed properties
        - must maintain even building
        '''
        can_be_downgraded = []
        for cell in cells_in_group:
            if cell.owner and cell.owner.name == player.name:
                if cell.has_hotel == 1 or cell.has_houses > 0:
                    print ("passed this one")
                    # Look at other cells in group to maintain even building
                    for other_cell in board.groups[cell.group]:
                        if other_cell.has_houses > cell.has_houses:
                            break
                    else:
                        can_be_downgraded.append(cell)
                        
        # Sort by development level (hotel first, then most houses)
        can_be_downgraded.sort(key=lambda x: (x.has_hotel * 5 + x.has_houses), reverse=True)
        return can_be_downgraded[0] if can_be_downgraded else None
    
    def downgrade_property(cell_to_downgrade):
        if not cell_to_downgrade:
            return False
            
        if cell_to_downgrade.has_hotel == 1:
            # Convert hotel back to 4 houses if possible
            if board.available_houses >= 4:
                cell_to_downgrade.has_hotel = 0
                cell_to_downgrade.has_houses = 4
                board.available_hotels += 1
                board.available_houses -= 4
                sell_price = cell_to_downgrade.cost_house // 2
                player.money += sell_price
                return True
            
        elif cell_to_downgrade.has_houses > 0:
            cell_to_downgrade.has_houses -= 1
            board.available_houses += 1
            sell_price = cell_to_downgrade.cost_house // 2
            player.money += sell_price
            return True
        return False

    # First try to sell buildings if any exist
    cell_to_downgrade = get_next_property_to_downgrade()
    if cell_to_downgrade:
        return downgrade_property(cell_to_downgrade)   
    return False

def update_state_after_selling (group_idx: int, original_board: Board, original_player:Player, players):
    player = copy.deepcopy(original_player)
    # board = copy.deepcopy(original_board)
    actions_status = sell_in_group_simulation(group_idx, original_board, player)
    if actions_status:
        print ("Selling went okay")
        new_state = s.get_state(s.get_area(player, players), 
            s.get_position(player.position), 
            s.get_finance(player, players))
        return new_state
    
    return 0