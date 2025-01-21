from classes.player import Player
from classes.board import Board, Property, Cell
from classes.log import Log
import numpy as np
from classes.state import State

class Update(State):
    def __init__(self, current_player: Player, players: list):
        # Call the parent class constructor to initialize the state
        super().__init__(current_player, players)
        self.old_state = None

    def update_after_purchase(self, current_player: Player, players: list, property: Property, property_cost: int):
        """ Updates the state after a player buys a property. """
        # Save the previous state
        self.old_state = self.state
        
        # Update the player's finance (subtract the property cost)
        current_player.money -= property_cost
        
        # Add the property to the player's owned properties
        current_player.owned.append(property)
        
        # Recalculate the player's area, finance, and other attributes after the purchase
        self.state = self.get_state(self.get_area(current_player, players), 
                                    self.get_position(current_player.position), 
                                    self.get_finance(current_player, players))

    def update_after_sale(self, current_player: Player, players: list, property: Property, sale_price: int):
        """ Updates the state after a player sells a property. """
        # Save the previous state
        self.old_state = self.state
        
        # Update the player's finance (add the sale price)
        current_player.money += sale_price
        
        # Remove the property from the player's owned properties
        current_player.owned.remove(property)
        
        # Recalculate the player's area, finance, and other attributes after the sale
        self.state = self.get_state(self.get_area(current_player, players), 
                                    self.get_position(current_player.position), 
                                    self.get_finance(current_player, players))

    def update_after_trade(self, trade: dict, players: list):
        """ Updates the state after a player trades with another player. """
        # Save the previous state
        self.old_state = self.state
        
        # Process the trade logic (transferring properties and money)
        self.process_trade(trade)
        
        # Recalculate the area and finance for both players after the trade
        self.state = self.get_state(self.get_area(self.current_player, players), 
                                    self.get_position(self.current_player.position), 
                                    self.get_finance(self.current_player, players))

    def has_state_changed(self) -> bool:
        """ Checks if the state has changed since the last update. """
        return not np.array_equal(self.old_state, self.state)
