import random
from classes.log import Log
from classes.game import monopoly_game, Board, Dice
from classes.player import BasicQPlayer, Fixed_Policy_Player
from settings import GameSettings, SimulationSettings
from monopoly_simulator import run_simulation
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MonopolyTrainer:
    def __init__(self, num_episodes =1000):
        self.num_episodes = num_episodes
        self.settings = GameSettings()
        self.rewards_history = []
        self.win_history = []
    
    def create_game(self):
        """create a new game instance with players"""
        board = Board(self.settings)
        log = Log()
        dice = Dice(random.random(), 2, 6, log)
    
        #create players
        q_player = BasicQPlayer("Q-Learning Player", self.settings)
        fixed_players = [
            Fixed_Policy_Player(f"Fixed Player {i}", self.settings) 
            for i in range(1, 4)
        ]

        players = [q_player] + fixed_players
        return board, dice, log, players
    
    def train(self):
        """Main training loop"""
        print("Starting training...")

        for episode in tqdm(range(self.num_episodes)):
            #create a new game instance
            board,dice,log,players= self.create_game()
            q_player = players[0]

            #initialize episode variables
            episode_rewards = 0
            game_over = False

            #decreasd exploration rate over time
            q_player.epsilon = max(0.01, q_player.epsilon * 0.995)
            num_moves = 0
            #play one full game
            while not game_over:
                if num_moves > SimulationSettings.n_moves:
                    game_over = True
                num_moves += 1
                for player in players:
                    if player.is_bankrupt:
                        continue

                    result = player.make_a_move(board,players,dice,log)

                    if result == "bankrupt":
                        if player == q_player:
                            game_over = True
                            break
                    
                    #check if only one player is left
                    active_players = [p for p in players if not p.is_bankrupt]
                    if len(active_players) == 1:
                        game_over = True
                        break

            #calculate final reward for episode
            final_reward = q_player.calculate_reward(board, players)
            self.rewards_history.append(final_reward)

            #record if q_player won
            won = not q_player.is_bankrupt and len([p for p in players if not p.is_bankrupt]) == 1
            self.win_history.append(1 if won else 0)

            #save Q-table periodically
            if episode % 10 == 0:
                q_player.log_q_table()

                #add debug printing of q table size and sample entries
                num_entries = len(q_player.qTable)
                print(f"\nEpisode {episode}")
                print(f"Q-table size: {num_entries}")
                print(f"Current epsilon: {q_player.epsilon: .3f}")
                print(f"Last reward:{final_reward}")

                #print a few sample q-values if they exist
                if num_entries >0:
                    print("\nSample Q-values:")
                    sample_states = list(q_player.qTable.items())[:5]
                    for state_action, q_value in sample_states:
                        print(f"{state_action}: {q_value:.2f}")
                print("-"*50)
            #if episode %100 == 0:
                #self.plot_training_results()
                
        #self.plot_training_results()
'''
    def plot_training_results(self):
        """plot training metrics"""
        plt.figure(figsize=(12,5))
        #plot rewards
        plt.subplot(1,2,1)
        plt.plot(self.rewards_history)
        plt.title('Rewards per episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        #plot win rate
        plt.subplot(1, 2, 2)

        window_size =100
        win_rate = np.convolve(self.win_history, 
                            np.ones(window_size)/window_size, 
                            mode='valid')
        plt.plot(win_rate)
        plt.title('Win rate(Moving Averate {window_size} episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
       
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.close()
        '''

if __name__=="__main__":
    trainer = MonopolyTrainer(num_episodes=1000)
    trainer.train()
