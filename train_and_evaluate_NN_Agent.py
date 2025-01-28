import random
from tqdm import tqdm
from settings import TrainingSettings, LogSettings
from classes.log import Log
from classes.game import monopoly_game
from classes.DQAgent_paper import QLambdaAgent
from settings import TrainingSettings, SimulationSettings
from monopoly_simulator import run_simulation

def train_model(config: TrainingSettings):
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
    qlambda_agent = QLambdaAgent(is_training = True)
    for i in tqdm(range(config.n_games)):
        monopoly_game(data_for_simulation[i], qlambda_agent = qlambda_agent)
        qlambda_agent.end_game()
    qlambda_agent.save_nn()
if __name__ == "__main__":
    print("Training...")
    train_model(TrainingSettings)
    print("Evaluating...")
    run_simulation(SimulationSettings)