import random
from tqdm import tqdm
from settings import TrainingSettings, LogSettings
from classes.log import Log
from classes.game import monopoly_game
from classes.DQAgent.DQAgent import QLambdaAgent
from settings import TrainingSettings, SimulationSettings
from monopoly_simulator import run_simulation
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
    data_for_simulation = [
        (i + 1, random.random())
        for i in range(config.n_games)]
    for i in tqdm(range(config.n_games)):
        test_before_each_game(qlambda_agent)
        monopoly_game(data_for_simulation[i], qlambda_agent = qlambda_agent)
        qlambda_agent.end_game()
    qlambda_agent.save_nn()

def test_before_each_game(qlambda_agent):  
    test_state = get_test_state() 
    q_values = qlambda_agent.calculate_all_q_values(test_state)
    test_str = ""
    for i in range(len(q_values)):
        test_str += str(Actions[i]) + ": " + str(q_values[i].item()) + "    "
    # print(test_str)
    
if __name__ == "__main__":
    qlambda_agent = QLambdaAgent(is_training = True)
    qlambda_agent.save_nn()
    print("Evaluating...")
    run_simulation(SimulationSettings)
    print("Training...")
    train_model(TrainingSettings, qlambda_agent)
    print("Evaluating...")
    run_simulation(SimulationSettings)