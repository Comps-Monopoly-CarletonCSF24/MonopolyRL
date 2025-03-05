''' Function, that wraps one game of monopoly:
from setting up boards, players etc to making moves by all players
'''
import os
from settings import SimulationSettings, GameSettings, LogSettings
import matplotlib.pyplot as plt

from classes.player import Fixed_Policy_Player, DQAPlayer, BasicQPlayer, ApproxQPlayer
from classes.board import Board
from classes.dice import Dice
from classes.log import Log
from classes.q_table_utils import initialize_q_table, load_q_table
'''
def plot_survival_rates(basic_q_survival, fixed_policy_survival):
    """Plot the survival rates of both agents."""
    plt.figure(figsize=(10, 5))

    # Plot survival rates
    plt.plot(basic_q_survival, label='Basic Q Agent', color='blue')
    plt.plot(fixed_policy_survival, label='Fixed Policy Agent', color='orange')
    plt.xlabel('Game Number')
    plt.ylabel('Survival Rate')
    plt.title('Survival Rate Comparison')
    plt.legend()

    # Save the plot to a file
    plt.savefig('survival_rates.png')
'''
def monopoly_game(data_for_simulation):
    ''' Simulation of one game.
    For convenience to set up a multi-thread,
    parameters are packed into a tuple: (game_number, game_seed):
    - "game number" is here to print out in the game log
    - "game_seed" to initialize random generator for the game
    '''
    game_number, game_seed = data_for_simulation

    # Initialize game log
    log = Log(LogSettings.game_log_file, disabled=not LogSettings.keep_game_log)

    # First line in the game log: game number and seed
    log.add(f"\n\n= GAME {game_number} of {SimulationSettings.n_games} " +
            f"(seed = {game_seed}) =")

    # Initialize data log
    datalog = Log(LogSettings.data_log_file)
    # Initialize Q-table once at the start of the game
    q_table_filename = "q_table.pkl"
    actions = ['buy','sell','do_nothing']

    # Initialize Q-table if it doesn't exist
    try:
        if not os.path.exists(q_table_filename):
            print(f"Initializing Q-table in {q_table_filename}")
            initialize_q_table(q_table_filename, actions)  # Create the Q-table if it doesn't exist
        else:
            print(f"Loading existing Q-table from {q_table_filename}")
            q_table = load_q_table(q_table_filename)  # Load the existing Q-table

    except Exception as e:
        print(f"Error with Q-table initialization: {e}")
        raise


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
        if player_type == "Fixed Policy":
            players.append(Fixed_Policy_Player(player_name, player_setting))
        elif player_type == "QLambda":
            players.append(DQAPlayer(player_name, player_setting))
        elif player_type == "BasicQ":
            players.append(BasicQPlayer(player_name, player_setting))
        else:
            players.append(ApproxQPlayer(player_name, player_setting))
            
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

    basic_q_survival = []
    fixed_policy_survival = []

    # Play for the required number of turns
    for turn_n in range(1, SimulationSettings.n_moves + 1):
        # Start a turn. Log turn's number
        log.add(f"\n== GAME {game_number} Turn {turn_n} ===")

        # Log all the players with their current position/money.
        # While we are at it, count alive players
        alive = 0

        for player_n, player in enumerate(players):

            if not player.is_bankrupt:
                alive += 1
                # Current player's position, money and net worth, looks like this:
                # - Player 'Experiment': $1220 (net $1320), at 21 (E1 Kentucky Avenue)
                log.add(f"- Player '{player.name}': " +
                        f"${int(player.money)} (net ${player.net_worth()}), " +
                        f"at {player.position} ({board.cells[player.position].name})")
            else:
                log.add(f"- Player {player_n}, '{player.name}': Bankrupt")

        # Log the number of available Houses/Hotels etc
        board.log_board_state(log)
        # Add an empty line before players' moves
        log.add("")

        # If there are less than 2 alive players
        # (0 alive is quite unlikely, but possible):
        # End the game
        if alive < 2:
            log.add("Only 1 player remains, game over")
            break

        # Players make their moves
        for player in players:
            # result will be "bankrupt" if player goes bankrupt
            result = player.make_a_move(board, players, dice, log)
            # If player goes bankrupt, log it in the data log file
            if result == "bankrupt":
                datalog.add(f"{game_number}\t{player}\t{turn_n}")

        for player in players:
            if isinstance(player, BasicQPlayer):
                basic_q_survival.append(1 if not player.is_bankrupt else 0)
            elif isinstance(player, Fixed_Policy_Player):
                fixed_policy_survival.append(1 if not player.is_bankrupt else 0)

    # Last thing to log in the game log: the final state of the board
    board.log_current_map(log)

    for player in players:
        if isinstance(player, DQAPlayer):
            player.agent.save_nn()
            
    # Save the logs
    log.save()
    datalog.save()

