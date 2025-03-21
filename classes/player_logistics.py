# A basic Player Class that only contains game logistics. 
# The handle_actions method, which is deined differently for each agent in player.py, 
# allows different agents to behave differently

from classes.board import Property, GoToJail, LuxuryTax, IncomeTax
from classes.board import FreeParking, Chance, CommunityChest
from settings import GameSettings, StandardPlayer


class Player:
    ''' Class to contain player-replated into and actions:
    - money, position, owned property
    - actions to buy property of handle Chance cards etc
    '''

    def __init__(self, name, settings):

        # Player's name and behavioral settings
        self.name = name
        self.settings = settings

        # Player's money (will be set up by the simulation)
        self.money = 0

        # Player's position
        self.position = 0

        # Person's roll double and jail status
        # Is the player currently in jail
        self.in_jail = False
        # How any doubles player thrown so far
        self.had_doubles = 0
        # How many days in jail player spent so far
        self.days_in_jail = 0
        # Does player have a GOOJF card
        self.get_out_of_jail_chance = False
        self.get_out_of_jail_comm_chest = False

        # Owned properties
        self.owned = []

        # List of properties player wants to sell / buy
        # through trading with other players
        self.wants_to_sell = set()
        self.wants_to_buy = set()

        # Bankrupt (game ended for thi player)
        self.is_bankrupt = False

        # Placeholder for various flags used throughout the game
        self.other_notes = ""

    def __str__(self):
        return self.name

    def net_worth(self, count_mortgaged_as_full_value=False):
        ''' Calculate player's net worth (cache + property + houses)
        count_mortgaged_as_full_value determines if we consider property mortgaged status:
        - True: count as full, for Income Tax calculation
        - False: count partially, for net worth statistics
        '''
        net_worth = int(self.money)

        for cell in self.owned:

            if cell.is_mortgaged and not count_mortgaged_as_full_value:
                # Partially count mortgaged properties
                net_worth += int(cell.cost_base * (1 - GameSettings.mortgage_value))
            else:
                net_worth += cell.cost_base
                net_worth += (cell.has_houses + cell.has_hotel) * cell.cost_house

        return net_worth
    def calculate_max_bid(self, property_to_auction, current_bid):
        #calculate max bid for the player
        return min(self.money * 0.7, property_to_auction.cost_base * 1.1)

    def make_a_move(self, board, players, dice, log):
        ''' REVISION 1/11/2025: this class is now purely game logic (except jail)
        Each agent will be a subclass of this class to add their own action functions
        Main function for a player to make a move
        Receives:
        - a board, with all cells and other things
        - other players (in case we need to make transactions with them)
        - dice (to roll)
        - log handle
        '''

        # Is player is bankrupt - do nothing
        if self.is_bankrupt:
            return None

        log.add(f"=== Player {self.name} (${self.money}, " +
                f"at {board.cells[self.position].name}) goes: ===")

        # Things to do before the throwing of the dice:

        # The move itself:
        # Player rolls the dice
        _, dice_roll_score, dice_roll_is_double = dice.cast()

        # Get doubles for the third time: just go to jail
        if dice_roll_is_double and self.had_doubles == 2:
            self.handle_going_to_jail("rolled 3 doubles in a row", log)
            return

        # Player is currently in jail
        if self.in_jail:
            # Will return True if player stays in jail and move ends
            if self.handle_being_in_jail(dice_roll_is_double, board, log):
                return

        # Player moves to a cell
        self.position += dice_roll_score
        # Get salary if we passed go on the way
        if self.position >= 40:
            self.handle_salary(board, log)
        # Get the correct position, if we passed GO
        self.position %= 40
        log.add(f"Player {self.name} goes to: {board.cells[self.position].name}")

        # For all agents, make actions here:
        self.handle_action(board, players, dice, log)

        # Handle various types of cells player may land on

        # Both cards are processed first, as they may send player to a property
        # and Chance is before "Community Chest" as Chance can send to Community Chest
        
        # Pay rent if landing on property
        if isinstance(board.cells[self.position], Property):
            self.handle_rent(board, dice, log)
            
        # Player lands on "Chance"
        if isinstance(board.cells[self.position], Chance):
            # returning "move is over" means the move is over (even if it was a double)
            if self.handle_chance(board, players, log) == "move is over":
                return

        # Player lands on "Community Chest"
        if isinstance(board.cells[self.position], CommunityChest):
            # returning "move is over" means the move is over (even if it was a double)
            if self.handle_community_chest(board, players, log) == "move is over":
                return

        # Player lands on "Go To Jail"
        if isinstance(board.cells[self.position], GoToJail):
            self.handle_going_to_jail("landed on Go To Jail", log)
            return

        # Player lands on "Free Parking"
        if isinstance(board.cells[self.position], FreeParking):
            # If Free Parking Money house rule is on: get the money
            if GameSettings.free_parking_money:
                log.add(f"{self} gets ${board.free_parking_money} from Free Parking")
                self.money += board.free_parking_money
                board.free_parking_money = 0

        # Player lands on "Luxury Tax"
        if isinstance(board.cells[self.position], LuxuryTax):
            self.pay_money(GameSettings.luxury_tax, "bank", board, log)
            if not self.is_bankrupt:
                log.add(f"{self} pays Luxury Tax ${GameSettings.luxury_tax}")

        # Player lands on "Income Tax"
        if isinstance(board.cells[self.position], IncomeTax):
            self.handle_income_tax(board, log)

        # Reset Other notes flag
        self.other_notes = ""

        # If player went bankrupt this turn - return string "bankrupt"
        if self.is_bankrupt:
            return "bankrupt"

        # If the roll was a double
        if dice_roll_is_double:
            # Keep track of doubles in a row
            self.had_doubles += 1
            # We already handled sending to jail, so player just goes again
            log.add(f"{self} rolled a double ({self.had_doubles} in a row) so they go again.")
            self.make_a_move(board, players, dice, log)
        # If now a double: reset double counter
        else:
            self.had_doubles = 0
    
    def handle_action(self, board, players, dice, log):
        pass

    def handle_salary(self, board, log):
        ''' Adding Salary to the player's money, according to the game's settings
        '''
        self.money += board.settings.salary
        log.add(f"Player {self.name} receives salary ${board.settings.salary}")

    def handle_going_to_jail(self, message, log):
        ''' Start the jail time
        '''
        log.add(f"{self} {message}, and goes to Jail.")
        self.position = 10
        self.in_jail = True
        self.had_doubles = 0
        self.days_in_jail = 0

    def handle_being_in_jail(self, dice_roll_is_double, board, log):
        ''' Handle player being in Jail
        Return True if the player stays in jail (to end his turn)
        '''
        # Get out of jail on rolling double
        if self.get_out_of_jail_chance or self.get_out_of_jail_comm_chest:
            log.add(f"{self} uses a GOOJF card")
            self.in_jail = False
            self.days_in_jail = 0
            # Return the card to the deck
            if self.get_out_of_jail_chance:
                board.chance.add("Get Out of Jail Free")
                self.get_out_of_jail_chance = False
            else:
                board.chest.add("Get Out of Jail Free")
                self.get_out_of_jail_comm_chest = False

        # Get out of jail on rolling double
        elif dice_roll_is_double:
            log.add(f"{self} rolled a double, a leaves jail for free")
            self.in_jail = False
            self.days_in_jail = 0
        # Get out of jail and pay fine
        elif self.days_in_jail == 2: # It's your third day
            log.add(f"{self} did not rolled a double for the third time, " +
                    f"pays {GameSettings.exit_jail_fine} and leaves jail")
            self.pay_money(GameSettings.exit_jail_fine, "bank", board, log)
            self.in_jail = False
            self.days_in_jail = 0
        # Stay in jail for another turn
        else:
            log.add(f"{self} stays in jail")
            self.days_in_jail  += 1
            return True
        return False

    def handle_chance(self, board, players, log):
        ''' Draw and act on a Chance card
        Return True if the move should be over (go to jail)
        '''
        card = board.chance.draw()
        log.add(f"{self} drew Chance card: '{card}'")

        # Cards that send you to a certain location on board

        if card == "Advance to Boardwalk":
            log.add(f"{self} goes to {board.cells[39]}")
            self.position = 39

        elif card == "Advance to Go (Collect $200)":
            log.add(f"{self} goes to {board.cells[0]}")
            self.position = 0
            self.handle_salary(board, log)

        elif card == "Advance to Illinois Avenue. If you pass Go, collect $200":
            log.add(f"{self} goes to {board.cells[24]}")
            if self.position > 24:
                self.handle_salary(board, log)
            self.position = 24

        elif card == "Advance to St. Charles Place. If you pass Go, collect $200":
            log.add(f"{self} goes to {board.cells[11]}")
            if self.position > 11:
                self.handle_salary(board, log)
            self.position = 11

        elif card == "Take a trip to Reading Railroad. If you pass Go, collect $200":
            log.add(f"{self} goes to {board.cells[5]}")
            if self.position > 5:
                self.handle_salary(board, log)
            self.position = 5

        # Going backwards

        elif card == "Go Back 3 Spaces":
            self.position -= 3
            log.add(f"{self} goes to {board.cells[self.position]}")

        # Sends to a type of location, and affects the rent amount

        elif card == "Advance to the nearest Railroad. " + \
                     "If owned, pay owner twice the rental to which they are otherwise entitled":
            nearest_railroad = self.position
            while (nearest_railroad - 5) % 10 != 0:
                nearest_railroad += 1
                nearest_railroad %= 40
            log.add(f"{self} goes to {board.cells[nearest_railroad]}")
            if self.position > nearest_railroad:
                self.handle_salary(board, log)
            self.position = nearest_railroad
            self.other_notes = "double rent"

        elif card == "Advance token to nearest Utility. " + \
             "If owned, throw dice and pay owner a total ten times amount thrown.":
            nearest_utility = self.position
            while nearest_utility not in  (12, 28):
                nearest_utility += 1
                nearest_utility %= 40
            log.add(f"{self} goes to {board.cells[nearest_utility]}")
            if self.position > nearest_utility:
                self.handle_salary(board, log)
            self.position = nearest_utility
            self.other_notes = "10 times dice"

        # Jail related (go to jail or GOOJF card)

        elif card == "Get Out of Jail Free":
            log.add(f"{self} now has a 'Get Out of Jail Free' card")
            self.get_out_of_jail_chance = True
            # Remove the card from the deck
            board.chance.remove("Get Out of Jail Free")

        elif card == "Go to Jail. Go directly to Jail, do not pass Go, do not collect $200":
            self.handle_going_to_jail("got GTJ Chance card", log)
            return "move is over"

        # Receiving money

        elif card == "Bank pays you dividend of $50":
            log.add(f"{self} gets $50")
            self.money += 50

        elif card == "Your building loan matures. Collect $150":
            log.add(f"{self} gets $150")
            self.money += 150

        # Paying money (+ depending on property + to other players)

        elif card == "Speeding fine $15":
            self.pay_money(15, "bank", board, log)

        elif card == "Make general repairs on all your property. For each house pay $25. " + \
                "For each hotel pay $100":
            repair_cost = sum(cell.has_houses * 25 + cell.has_hotel * 100 for cell in self.owned)
            log.add(f"Repair cost: ${repair_cost}")
            self.pay_money(repair_cost, "bank", board, log)

        elif card == "You have been elected Chairman of the Board. Pay each player $50":
            for other_player in players:
                if other_player != self and not other_player.is_bankrupt:
                    self.pay_money(50, other_player, board, log)
                    if not self.is_bankrupt:
                        log.add(f"{self} pays {other_player} $50")

        return ""

    def handle_community_chest(self, board, players, log):
        ''' Draw and act on a Community Chest card
        Return True if the move should be over (go to jail)
        '''

        card = board.chest.draw()
        log.add(f"{self} drew Community Chest card: '{card}'")

        # Moving to Go

        if card == "Advance to Go (Collect $200)":
            log.add(f"{self} goes to {board.cells[0]}")
            self.position = 0
            self.handle_salary(board, log)

        # Jail related

        elif card == "Get Out of Jail Free":
            log.add(f"{self} now has a 'Get Out of Jail Free' card")
            self.get_out_of_jail_comm_chest = True
            # Remove the card from the deck
            board.chest.remove("Get Out of Jail Free")

        elif card == "Go to Jail. Go directly to Jail, do not pass Go, do not collect $200":
            self.handle_going_to_jail("got GTJ Community Chest card", log)
            return "move is over"

        # Paying money

        elif card == "Doctor's fee. Pay $50":
            self.pay_money(50, "bank", board, log)

        elif card == "Pay hospital fees of $100":
            self.pay_money(100, "bank", board, log)

        elif card == "Pay school fees of $50":
            self.pay_money(50, "bank", board, log)

        elif card == "You are assessed for street repair. $40 per house. $115 per hotel":
            repair_cost = sum(cell.has_houses * 40 + cell.has_hotel * 115 for cell in self.owned)
            log.add(f"Repair cost: ${repair_cost}")
            self.pay_money(repair_cost, "bank", board, log)

        # Receive money

        elif card == "Bank error in your favor. Collect $200":
            log.add(f"{self} gets $200")
            self.money += 200

        elif card == "From sale of stock you get $50":
            log.add(f"{self} gets $50")
            self.money += 50

        elif card == "Holiday fund matures. Receive $100":
            log.add(f"{self} gets $100")
            self.money += 100

        elif card == "Income tax refund. Collect $20":
            log.add(f"{self} gets $20")
            self.money += 20

        elif card == "Life insurance matures. Collect $100":
            log.add(f"{self} gets $100")
            self.money += 100

        elif card == "Receive $25 consultancy fee":
            log.add(f"{self} gets $25")
            self.money += 25

        elif card == "You have won second prize in a beauty contest. Collect $10":
            log.add(f"{self} gets $10")
            self.money += 10

        elif card == "You inherit $100""You inherit $100":
            log.add(f"{self} gets $100")
            self.money += 100

        # Receiving money from other players

        elif card == "It is your birthday. Collect $10 from every player":
            for other_player in players:
                if other_player != self and not other_player.is_bankrupt:
                    other_player.pay_money(50, self, board, log)
                    if not other_player.is_bankrupt:
                        log.add(f"{other_player} pays {self} $10")

        return ""

    def handle_income_tax(self, board, log):
        ''' Handle Income tax: choose which option
        (fix or %) is less money and go with it
        '''
        # Choose smaller between fixed rate and percentage
        tax_to_pay = min(
            GameSettings.income_tax,
            int(GameSettings.income_tax_percentage *
            self.net_worth(count_mortgaged_as_full_value=True)))

        if tax_to_pay == GameSettings.income_tax:
            log.add(f"{self} pays fixed Income tax {GameSettings.income_tax}")
        else:
            log.add(f"{self} pays {GameSettings.income_tax_percentage * 100:.0f}% " +
                    f"Income tax {tax_to_pay}")
        self.pay_money(tax_to_pay, "bank", board, log)

    def raise_money(self, required_amount, board, log):
        ''' Part of "Pay money" method. If there is not enough cash, player has to 
        sell houses, hotels, mortgage property until you get required_amount of money
        '''

        def get_next_property_to_deimprove(required_amount):
            ''' Get the next property to sell houses/hotel from.
            Logic goes as follows:
            - if you can sell a house, sell a house (otherwise seel a hotel, if you have no choice)
            - sell one that would bring you just above the required amount (or the most expensive)
            '''

            # 1. let's see which properties CAN be de-improved
            # The house/hotel count is the highest in the group
            can_be_deimproved = []
            can_be_deimproved_has_houses = False
            for cell in self.owned:
                if cell.has_houses > 0 or cell.has_hotel > 0:
                    # Look at other cells in this group
                    # Do they have more houses or hotels?
                    # If so this property cannot be de-improved
                    for other_cell in board.groups[cell.group]:
                        if cell.has_hotel == 0 and (other_cell.has_houses > cell.has_houses or \
                           other_cell.has_hotel > 0):
                            break
                    else:
                        can_be_deimproved.append(cell)
                        if cell.has_houses > 0:
                            can_be_deimproved_has_houses = True

            # No further de-improvements possible
            if len(can_be_deimproved) == 0:
                return None

            # 2. If there are houses and hotels, remove hotels from the list
            # Selling a hotel is a last resort
            if can_be_deimproved_has_houses:
                can_be_deimproved = [x for x in can_be_deimproved if x.has_hotel == 0]

            # 3. Find one that's just above the required amount (or the most expensive one)
            # Sort potential de-improvements from cheap to expensive
            can_be_deimproved.sort(key = lambda x: x.cost_house // 2)
            while True:
                # Only one possible option left
                if len(can_be_deimproved) == 1:
                    return can_be_deimproved[0]
                # Second expensive option is not enough, sell most expensive
                if can_be_deimproved[-2].cost_house // 2 < required_amount:
                    return can_be_deimproved[-1]
                # Remove most expensive option
                can_be_deimproved.pop()

        def get_list_of_properties_to_mortgage():
            ''' Put together a list of properties a player can sell houses from.
            '''
            list_to_mortgage = []
            for cell in self.owned:
                if not cell.is_mortgaged:
                    list_to_mortgage.append(
                        (int(cell.cost_base * GameSettings.mortgage_value), cell))

            # It will be popped from the end, so first to sell should be last
            list_to_mortgage.sort(key = lambda x: -x[0])
            return list_to_mortgage

        # Cycle through all possible de-improvements until
        # all houses/hotels are sold or enough money is raised
        while True:
            money_to_raise = required_amount - self.money
            cell_to_deimprove = get_next_property_to_deimprove(money_to_raise)

            if cell_to_deimprove is None or money_to_raise <= 0:
                break

            sell_price = cell_to_deimprove.cost_house // 2

            # Selling a hotel
            if cell_to_deimprove.has_hotel:
                # Selling hotel: can replace with 4 houses
                if board.available_houses >= 4:
                    cell_to_deimprove.has_hotel = 0
                    cell_to_deimprove.has_houses = 4
                    board.available_hotels += 1
                    board.available_houses -= 4
                    log.add(f"{self} sells a hotel on {cell_to_deimprove}, raising ${sell_price}")
                    self.money += sell_price
                # Selling hotel, must tear down all 5 houses from one plot
                # TODO: I think we need to tear down all 3 hotels in this situation?
                else:
                    cell_to_deimprove.has_hotel = 0
                    cell_to_deimprove.has_houses = 0
                    board.available_hotels += 1
                    log.add(f"{self} sells a hotel and all houses on {cell_to_deimprove}, " +
                            f"raising ${sell_price * 5}")
                    self.money += sell_price * 5

            # Selling a house
            else:
                cell_to_deimprove.has_houses -= 1
                board.available_houses += 1
                ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4:"4th"}
                log.add(f"{self} sells {ordinal[cell_to_deimprove.has_houses + 1]} " +
                        f"house on {cell_to_deimprove}, raising ${sell_price}")
                self.money += sell_price


        # Mortgage properties
        list_to_mortgage = get_list_of_properties_to_mortgage()

        while list_to_mortgage and self.money < required_amount:
            # Pick property to mortgage from the list
            mortgage_price, cell_to_mortgage = list_to_mortgage.pop()

            # Mortgage this property
            cell_to_mortgage.is_mortgaged = True
            self.money += mortgage_price
            log.add(f"{self} mortgages {cell_to_mortgage}, raising ${mortgage_price}")

    def pay_money(self, amount, payee, board, log):
        ''' Function to pay money to another player (or bank)
        This is where Bankruptcy is triggered.
        '''

        def count_max_raisable_money():
            ''' How much cash a plyer can produce?
            Used to determine if they should go bankrupt or not.
            Max raisable money are 1/2 of houses cost + 1/2 of unmortgaged properties cost
            '''
            max_raisable = self.money
            for cell in self.owned:
                if cell.has_houses > 0:
                    max_raisable += cell.cost_house * cell.has_houses // 2
                if cell.has_hotel > 0:
                    max_raisable += cell.cost_house * 5 // 2
                if not cell.is_mortgaged:
                    max_raisable += int(cell.cost_base * GameSettings.mortgage_value)
            return max_raisable

        def transfer_all_properties(payee, board, log):
            ''' Part of bankruptcy procedure, transfer all mortgaged property to the creditor
            '''

            while self.owned:
                cell_to_transfer = self.owned.pop()

                # Transfer to a player
                # TODO: Unmortgage the property right away, or pay more
                if isinstance(payee, Player):
                    cell_to_transfer.owner = payee
                    payee.owned.append(cell_to_transfer)
                # Transfer to the bank
                # TODO: Auction the property
                else:
                    cell_to_transfer.owner = None
                    cell_to_transfer.is_mortgaged = False

                board.recalculate_monopoly_coeffs(cell_to_transfer)
                log.add(f"{self} transfers {cell_to_transfer} to {payee}")

        # Regular transaction
        if amount < self.money:
            self.money -= amount
            if payee != "bank":
                payee.money += amount
            elif payee == "bank" and GameSettings.free_parking_money:
                board.free_parking_money += amount
            return

        max_raisable_money = count_max_raisable_money()
        # Can pay but need to sell some things first
        if amount < max_raisable_money:
            log.add(f"{self} has ${self.money}, he can pay ${amount}, " +
                    "but needs to mortgage/sell some things for that")
            self.raise_money(amount, board, log)
            self.money -= amount
            if payee != "bank":
                payee.money += amount
            elif payee == "bank" and GameSettings.free_parking_money:
                board.free_parking_money += amount


        # Bunkruptcy (can't pay even after selling and mortgaging all)
        else:
            log.add(f"{self} has to pay ${amount}, max they can raise is ${max_raisable_money}")
            self.is_bankrupt = True
            log.add(f"{self} is bankrupt")

            # Raise as much cash as possible to give payee
            self.raise_money(amount, board, log)
            log.add(f"{self} gave {payee} all their remaining money (${self.money})")
            if payee != "bank":
                payee.money += self.money
            elif payee == "bank" and GameSettings.free_parking_money:
                board.free_parking_money += amount

            self.money = 0

            # Transfer all property (mortgaged at this point) to payee
            transfer_all_properties(payee, board, log)

            # Reset all trade settings
            self.wants_to_sell = set()
            self.wants_to_buy = set()

    def update_lists_of_properties_to_trade(self, board):
        ''' Update list of properties player is willing to sell / buy
        '''

        # If player is not willing to trade, he would
        # have not declare his offered and desired properties,
        # thus stopping any trade with them
        if not StandardPlayer.participates_in_trades:
            return

        # Reset the lists
        self.wants_to_sell = set()
        self.wants_to_buy = set()

        # Go through each group
        for group_cells in board.groups.values():

            # Break down all properties within each color group into
            # "owned by me" / "owned by others" / "not owned"
            owned_by_me = []
            owned_by_others = []
            not_owned = []
            for cell in group_cells:
                if cell.owner == self:
                    owned_by_me.append(cell)
                elif cell.owner is None:
                    not_owned.append(cell)
                else:
                    owned_by_others.append(cell)

            # If there properties to buy - no trades
            if not_owned:
                continue
            # If I own 1: I am ready to sell it
            if len(owned_by_me) == 1:
                self.wants_to_sell.add(owned_by_me[0])
            # If someone owns 1 (and I own the rest): I want to buy it
            if len(owned_by_others) == 1:
                self.wants_to_buy.add(owned_by_others[0])
    
    def handle_rent(self, board, dice, log):
        landed_property = board.cells[self.position]
        if hasattr(landed_property, 'skip_rent_this_turn') and landed_property.skip_rent_this_turn:
            log.add(f"Property was just auctioned, no rent charged")
            landed_property.skip_rent_this_turn = False
            return
        # It is mortgaged: no action
        if landed_property.is_mortgaged:
            log.add("Property is mortgaged, no rent")

        # It is player's own property
        elif landed_property.owner == self:
            log.add("Own property, no rent")

        # Handle rent payments
        elif landed_property.owner is not None:
            
            
            rent_amount = landed_property.calculate_rent(dice)
            if self.other_notes == "double rent":
                rent_amount *= 2
                log.add(f"Per Chance card, rent is doubled (${rent_amount}).")
            if self.other_notes == "10 times dice":
                # Divide by monopoly_coef to restore the dice throw
                # Multiply that by 10
                rent_amount = rent_amount // landed_property.monopoly_coef * 10
                log.add(f"Per Chance card, rent is 10x dice throw (${rent_amount}).")
            log.add(f"Player {self.name} landed on {landed_property}, " +
                    f"owned by {landed_property.owner}")
            self.pay_money(rent_amount, landed_property.owner, board, log)
            if not self.is_bankrupt:
                log.add(f"{self} pays {landed_property.owner} rent ${rent_amount}")
    
    def auction_property(self, property_to_auction, players, log):
        """Auction a property to the highest bidder once a player chooses not to buy"
        Args:
            property_to_auction: the property to auction
            players: the list of players
            log: the log object
        """
    
        if property_to_auction.owner is not None:
            log.add(f"DEBUG: Property already has owner, exiting auction")
            return
        
        #create a list of eligible bidders(excluding self and bankrupt players)
        eligible_bidders = [p for p in players if p.name != self.name and not p.is_bankrupt]
        current_bid = property_to_auction.cost_base // 2

        #log.add(f"DEBUG: Found {len(eligible_bidders)} eligible bidders, who is {[p.name for p in eligible_bidders]}")

        if len(eligible_bidders) == 0:
            log.add(f"No eligible bidders for {property_to_auction}")
            return
        elif len(eligible_bidders) == 1:
            log.add(f"\n===Auctioning {property_to_auction} starting at ${current_bid}===")
            #log.add(f"DEBUG: Only one eligible bidder, {eligible_bidders[0].name}, buying property")
            property_to_auction.owner = eligible_bidders[0]
            eligible_bidders[0].owned.append(property_to_auction)
            eligible_bidders[0].money -= current_bid
            current_winner = eligible_bidders[0]
            property_to_auction.skip_rent_this_turn = True
        
        else:
        #start auction at half the property's base cost
            
            current_winner = None
            #continue auction until no one wants to bid higher
            
            #Each eligible bidder has a chance to bid
            for player in range(len(eligible_bidders)): 
                log.add(f"DEBUG: player: {eligible_bidders[player].name} is bidding")
                
                if (eligible_bidders[player].money >  current_bid + 10 and 
                    current_bid < property_to_auction.cost_base):
                    #use player's calculate_max_bid method to determine max bid for different player types
                    max_bid=min(
                        eligible_bidders[player].calculate_max_bid(property_to_auction, current_bid),
                        base_price
                      )
                    log.add(f"DEBUG: max bid: {max_bid}")  #don't bid above the property's base cost
                    
                    
                    
                    if current_bid < max_bid:
                        new_bid = min(current_bid + 10, max_bid) #increment bid by $10
                        current_bid = new_bid
                        current_winner = eligible_bidders[player]
                        log.add(f"{eligible_bidders[player].name} bids ${new_bid}")
                    else:
                        log.add(f"{eligible_bidders[player].name} does not want to bid higher")

        #sell to highest bidder
        if current_winner is not None:
            current_winner.money -= current_bid
            property_to_auction.owner = current_winner
            current_winner.owned.append(property_to_auction)   #add property to the winner's owned list
            log.add(f"{current_winner.name} buys {property_to_auction} for ${current_bid}")
        else:
            log.add(f"No one buys {property_to_auction}")