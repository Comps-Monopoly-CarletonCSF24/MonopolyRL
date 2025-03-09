# Dake Peng '25, Last Modified: 3/8/2025
# Trains a QLambdaAgent on the Monopoly game environment based on settings.py
# The agent is trained for a number of batches, each batch consisting of a number of games

import random
from tqdm import tqdm
from settings import TrainingSettings, LogSettings
from classes.log import Log
from classes.game import monopoly_game
from classes.DQAgent.DQAgent import QLambdaAgent
from settings import TrainingSettings
from classes.state import get_test_state
from classes.DQAgent.action import Actions

def train_model(config: TrainingSettings, qlambda_agent):
    ''' Run the simulation
    In: Simulation parameters (number of games, seed etc)
    '''

    # Empty the game log file (list of all player actions)
    log = Log(LogSettings.game_log_file)
    log.reset()

    # Empty the data log (list of bankruptcy turns for each player)
    datalog = Log(LogSettings.data_log_file)
    datalog.reset("game_number\tplayer\tturn")
    pre_game_tests = []
    data_for_simulation = [
        (j + 1, random.random())
        for j in range(config.n_games_per_batch * config.n_batches)]
    for i in tqdm(range(config.n_batches)):
        for j in tqdm(range(config.n_games_per_batch)):
            pre_game_tests.append(test_before_each_game(qlambda_agent, 0))
            qlambda_agent.rewards.append([[],[],[]])
            qlambda_agent.choices.append([0,0,0,0])
            data_number = i * config.n_games_per_batch + j
            monopoly_game(data_for_simulation[data_number], qlambda_agent = qlambda_agent)
            qlambda_agent.end_game()
        ## Debugging statements that log q values, rewards, and #choices of each type
        # print(f"End of batch {i}: Epsilon: {qlambda_agent.epsilon}, Alpha: {qlambda_agent.alpha}\n")
        # with open("rewards.txt", "w") as file:
        #     for game in qlambda_agent.rewards:
        #         buy = sum(game[0]) /len (game[0]) if game[0] else 0
        #         sell = sum(game[1]) /len (game[1]) if game[1] else 0
        #         do_nothing = sum(game[2]) /len (game[2]) if game[2] else 0
        #         print(f"Buy: {buy}; Sell: {sell}; Do Nothing: {do_nothing}", file= file)
        # with open("choices.txt", "w") as file:
        #     for i in range(len(qlambda_agent.choices)):
        #         print(f"Game {i+1}: {qlambda_agent.choices[i]}", file = file)
        # with open("q_values.txt", "w") as file:
        #     for i in range(len(pre_game_tests)):
        #         print(f"Game {i+1}: {pre_game_tests[i]}", file = file)
def test_before_each_game(qlambda_agent, test_position):  
    test_state = get_test_state(test_position) 
    q_values = qlambda_agent.calculate_all_q_values(test_state)
    test_str = ""
    for i in range(len(q_values)):
        test_str += str(Actions[i]) + ": " + str(q_values[i].item()) + "    "
    return test_str
    
if __name__ == "__main__":
    qlambda_agent = QLambdaAgent(is_training = True)
    qlambda_agent.save_nn()
    print("Training...")
    train_model(TrainingSettings, qlambda_agent)
    qlambda_agent.save_nn()