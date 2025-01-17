import random
import unittest
import sys
import os
import numpy as np

# add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from settings import GameSettings, LogSettings
from classes.player_logistics import Player
from classes.board import Board
from classes.dice import Dice
from classes.log import Log
from classes.state import State
from classes.state import get_property_points_by_group
def init_game(game_seed):
    
    # Initialize game log
    log = Log(LogSettings.game_log_file, disabled=not LogSettings.keep_game_log)

    # Initialize data log
    datalog = Log(LogSettings.data_log_file)

    # Initialize the board (plots, chance, community chest etc)
    board = Board(GameSettings)

    # Set up dice (it creates a separate random generator with initial "game_seed",
    # to have thread-safe shuffling and dice throws)
    dice = Dice(game_seed, GameSettings.dice_count, GameSettings.dice_sides, log)

    # Shuffle chance and community chest cards
    # (using thread-safe random generator)
    dice.shuffle(board.chance.cards)
    dice.shuffle(board.chest.cards)

    # Set up players with their behavior settings
    players = [Player(player_name, player_setting)
               for player_name, player_setting in GameSettings.players_list]

    if GameSettings.shuffle_players:
        # dice has a thread-safe copy of random.shuffle
        dice.shuffle(players)

    # Set up players starting money according to the game settings:
    # If starting money is a list, assign it to players in order
    if isinstance(GameSettings.starting_money, list):
        for player, starting_money in zip(players, GameSettings.starting_money):
            player.money = starting_money
    # If starting money is a single value, assign it to all players
    else:
        for player in players:
            player.money = GameSettings.starting_money

    return board, players, dice, log

def test_game(num_turns_to_play, game_seed):
    ''' Plays a game for a number of rounds and stop, returns all objects regarding the game'''
    print(f"seed used: {game_seed}")
    board, players, dice, log = init_game(game_seed)
    # Play for the required number of turns
    for turn_n in range(1, num_turns_to_play):
        # Players make their moves
        for player in players:
            # result will be "bankrupt" if player goes bankrupt
            result = player.make_a_move(board, players, dice, log)
            # If player goes bankrupt, log it in the data log file
    return board, players, dice, log

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