''' Main file to run monopoly simulation
'''
import multiprocessing
import os

# Allow duplicate OpenMP runtime libraries to load
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import concurrent.futures

from tqdm import tqdm

from settings import TrainingSettings, LogSettings

from classes.analyze import Analyzer
from classes.log import Log
from classes.game import monopoly_game


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

    for i in tqdm(range(config.n_episodes)):
        for j in range(config.n_games_per_episode):
            data_for_simulation = (25 * i + j, random.random())
            monopoly_game(data_for_simulation)
    
if __name__ == "__main__":
    train_model(TrainingSettings)
