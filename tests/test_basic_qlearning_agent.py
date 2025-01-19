import random
import unittest
import sys
import os
import numpy as np
# add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.state import State

class Test_Basic_QLearning(unittest.TestCase):

    def test_state_space(self):
        """Plays multiple game for a number of rounds, checks if the state space 
        makes sense for a random player"""
        for i in range(50):
            _, players, _, _ = test_game(30, random.random())
            current_player = players[0]
            state = State(current_player, players).state
            
            # for player in players:
            #     print(get_property_points_by_group(player))
            #     for property in player.owned:
            #         print(f"    {property.group} {property.has_houses} {property.has_hotel}")
            self.assertTrue(np.all((state >= 0) & (state <= 1 + 1e-9)))
            self.assertTrue(len(state) == 23)
            
if __name__ == "__main__":
    unittest.main()