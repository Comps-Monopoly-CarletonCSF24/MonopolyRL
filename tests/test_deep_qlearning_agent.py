import random
import unittest
import sys
import os
from test_game import test_game

# add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.state import State
# class Test_Deep_QLearning(unittest.TestCase):

def test_is_state_similar():
    """Plays multiple game for a number of rounds, checks if the state space 
    makes sense for a random player"""
    _, players, _, _ = test_game(1, random.random())
    current_player = players[0]
    state = State(current_player, players)
    print(state.is_similar_to(state))
        
if __name__ == "__main__":
    pass