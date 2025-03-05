import matplotlib.pyplot as plt
import pandas as pd

from settings import SimulationSettings, GameSettings, LogSettings
def plot_winning_rates():
    '''Plot win rates for each player over games'''
    df = pd.read_csv(LogSettings.data_log_file, sep='\t')
    game_numbers = range(0, 1501) 
    players = df['player'].unique()
    wins = {player: [] for player in players}
    draws = []

    # Calculate wins and draws
    for game in game_numbers:
        if game in df['game_number'].values:
            loser = df.loc[df['game_number'] == game, 'player'].values[0]
            winner = [p for p in players if p != loser][0]
            # Increment winner's count
            wins[winner].append(1)
            # Loser gets no win for this gamex
            wins[loser].append(0)
        else:
            # Game was a draw
            for player in players:
                wins[player].append(0)

    # Calculate cumulative wins 
    cumulative_wins = {player: pd.Series(wins[player]).cumsum() for player in players}
    cumulative_wins = {player: pd.Series(wins[player]).cumsum() for player in players}
    win_percentage = {player: cumulative_wins[player] / (pd.Series(range(1, len(wins[player]) + 1))) for player in players}

    # Plot cumulative wins
    plt.figure(figsize=(10, 8))
    for player, wins in win_percentage.items():
        plt.plot(game_numbers, wins, marker='o', label=f'{player} Win Rate')

    # Add labels, title, and legend
    plt.xlabel('Game Number')
    plt.ylabel('Win Rate')
    plt.title('Win Rates Over Games')
    plt.legend(title='Outcome')
    plt.grid(True)
    plt.show()

plot_winning_rates ()