from classes.player import Player

class Reward:
    def __init__(self):
        pass
    
    def get_alive_players(self, players):
        """
        Creates a list of all the alive players based on their net worth.
        """
        return [player for player in players if player.net_worth() > 0]

    def get_reward(self, player, players):
        """
        Compute the reward based on the player's net worth compared to opponents' net worth.
        """
        player_networth = player.net_worth()
        alive_players = self.get_alive_players(players)

        rem_players = [_.name for _ in alive_players]
        # if (len(rem_players) == 1):
            # print("Survivor:", rem_players)
        
        all_players_worth = sum(player.net_worth() for player in alive_players)
        
        if all_players_worth == 0:  # Avoid division by zero
            return 0
        
        num_players = len(alive_players)  # Dynamic number of players
        smoothing_factor = 0.450
        
        # Adjusted calculations
        net_worth_difference = player_networth - (all_players_worth - player_networth)
        player_finance_percent = (player_networth / all_players_worth) * 100
        
        # Reward calculation
        reward = ((net_worth_difference / num_players) * smoothing_factor) / \
                 (1 + abs((net_worth_difference / num_players) * smoothing_factor) - (1 / num_players) * player_finance_percent)
        
        return reward
