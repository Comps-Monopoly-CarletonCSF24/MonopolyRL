from test_settings import GameSettings, LogSettings
import sys ,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.log import Log
from classes.player_logistics import Player
from classes.board import Board
from classes.dice import Dice
from classes.player import DQAPlayer, Fixed_Policy_Player, BasicQPlayer

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

    players = []
    # Set up players with their behavior settings
    for player_name, player_type, player_setting in GameSettings.players_list:
        match player_type:
            case "Fixed Policy":
                players.append(Fixed_Policy_Player(player_name, player_setting))
            case "QLambda":
                players.append(DQAPlayer(player_name, player_setting))
            case "BasicQ":
                players.append(BasicQPlayer(player_name, player_setting))
                
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
