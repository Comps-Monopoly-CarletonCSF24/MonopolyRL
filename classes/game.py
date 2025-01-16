''' Function, that wraps one game of monopoly:
from setting up boards, players etc to making moves by all players
'''
from settings import SimulationSettings, GameSettings, LogSettings

from classes.player import Player
from classes.board import Board
from classes.dice import Dice
from classes.log import Log
from classes.basic_q_learning_agent import Q_learning_agent
from classes.state import State
from classes.action import Action

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


    actions = ['buy_all', 'do_nothing']
    agent = Q_learning_agent(name="Q_Agent",settings=GameSettings)
    agent.actions = actions
    current_player = players[1]
    state_object = State(current_player, players)
    action_object = Action()
    current_state = tuple(state_object.state) #renamed from get_state_vector
    
    # Play for the required number of turns
    for turn_n in range(1, SimulationSettings.n_moves + 1):
        
        # Thinking of switching all this out for a call to agent turn function moved into q-learning agent file
        # chosen_action = agent.choose_action(get_state_vector)
        
        # property_idx = 0
        # # action_vector = action_object.get_action_vector(property_idx, chosen_action)
        # reward = get_reward(players[1], players)
        # next_state_instance = State(current_player, players) 

        # next_state_vector = next_state_instance.state

        # agent.updateQValue(get_state_vector, chosen_action, reward, next_state_vector)

        # get_state_vector = tuple(next_state_vector)
        current_state = agent.take_turn(action_object, current_player, board, players,current_state)

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

    def convert_state_to_serializable(state):
    # Convert tuple of np.float64 to list of regular floats
        numeric_values = [float(x) for x in state[0]]  # First part of state tuple
        action = state[1]  # Second part of state tuple (the action string)
        return f"{numeric_values}, {action}"  # Custom string format
    import json
        # Save Q-table as JSON
    q_table_serializable = {
        convert_state_to_serializable(state): values 
        for state, values in agent.qTable.items()}

    with open("q_table_output.json", "w") as f:
        json.dump(q_table_serializable, f, indent=4)
    

    # Last thing to log in the game log: the final state of the board
    board.log_current_map(log)

    # Save the logs
    log.save()
    datalog.save()

    # Useless return, but it is here to mark the end of the game
    return None


