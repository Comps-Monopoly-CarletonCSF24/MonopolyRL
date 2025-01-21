class Action:
    def __init__(self):
        self.properties = list(range(28))  # Property indices from 0 to 27
        self.actions = ['buy', 'sell', 'trade', 'do_nothing']  # Added 'trade' to actions
        self.total_actions = len(self.properties) * len(self.actions)  # Now 1x112 action space (28 * 4)

    def map_action_index(self, action_index):
        """
        Maps a flattened action index to a specific property and action type.

        Parameters:
            - action_index (int): The index of the action in the flattened action space.
        
        Returns:
            - tuple: A tuple containing the property index and the action type.
        """
        if not isinstance(action_index, (int, np.integer)):
            try:
                action_index = int(action_index)
            except ValueError:
                raise ValueError(
                    f"action_index must be an integer, got {type(action_index)}."
                )
    
        if action_index < 0 or action_index >= self.total_actions:
            raise ValueError(
                f"The action index must be an integer between 0 and {self.total_actions - 1}."
            )
        property_idx = action_index // len(self.actions)
        action_type = self.actions[action_index % len(self.actions)]
        return property_idx, action_type

    def is_excutable(self, player, board, property_idx, action_idx):
        """
        Checks if the player can take the action that they are attempting to take.
        Returns true if the player can and False otherwise. 
        """
        property = board.get_property(property_idx)
        
        if action_idx == 1:  # buy
            return True
        elif action_idx == 0:  # sell
            if property.owner is None and player.can_afford(property.price):
                return True
        elif action_idx == 2:  # trade
            # Check if property is in player's wants_to_buy or wants_to_sell sets
            return (property in player.wants_to_buy) or (property in player.wants_to_sell)
        return False

    def execute_action(self, player, board, property_idx, action_type, log=None):
        """
        Executes the action on the given property for the specified player.
        """
        property = board.get_property(property_idx)
        
        if action_type == 'buy':
            if property.owner is None and player.money >= property.cost_base:
                player.buy_property(property, log)
                
        elif action_type == 'sell':
            if property.owner == player:
                player.sell_property(property, log)
                
        elif action_type == 'trade':
            # If property is in wants_to_buy, try to initiate trade
            if property in player.wants_to_buy:
                other_player = property.owner
                if other_player and other_player != player:
                    # Try to find a property to trade
                    for my_property in player.owned:
                        if my_property in other_player.wants_to_buy:
                            # Found potential trade, initiate it
                            player_gives = [my_property]
                            player_receives = [property]
                            
                            # Calculate price difference
                            price_diff = property.cost_base - my_property.cost_base
                            
                            # Execute trade if both players can afford it
                            if (price_diff > 0 and other_player.money >= price_diff) or \
                               (price_diff < 0 and player.money >= abs(price_diff)):
                                # Handle money transfer
                                if price_diff > 0:
                                    other_player.money -= price_diff
                                    player.money += price_diff
                                else:
                                    player.money -= abs(price_diff)
                                    other_player.money += abs(price_diff)
                                
                                # Transfer properties
                                property.owner = player
                                player.owned.append(property)
                                other_player.owned.remove(property)
                                
                                my_property.owner = other_player
                                other_player.owned.append(my_property)
                                player.owned.remove(my_property)
                                
                                if log:
                                    log.add(f"Trade: {player} gives {my_property}, " +
                                           f"receives {property} from {other_player}")
                                    if price_diff != 0:
                                        log.add(f"Price difference paid: ${abs(price_diff)}")
                                
                                # Update trade lists
                                player.update_lists_of_properties_to_trade(board)
                                other_player.update_lists_of_properties_to_trade(board)
                                break
                                
        elif action_type == 'do_nothing':
            pass