import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "./training_results/datalog_DQ.txt"  # Update with your actual file path
df = pd.read_csv(file_path, sep="\t")

# Identify winners: the missing player in each game
winners = df.groupby("game_number")["player"].apply(set)

# Get all unique players
all_players = set(df["player"])

# Find the winner for each game
winner_series = winners.apply(lambda players: (all_players - players).pop() if len(all_players - players) == 1 else None)

# Create a cumulative win count
win_counts = winner_series.value_counts().sort_index()

# Plot the cumulative wins over game numbers
plt.figure(figsize=(10, 5))

fixed_policy_players = [player for player in all_players if "Fixed Policy" in player]
agent = [player for player in all_players if "Fixed Policy"  not in player][0]
fixed_policy_wins = pd.Index([])
for player in fixed_policy_players:
    fixed_policy_wins = fixed_policy_wins.union(winner_series[winner_series == player].index)
fixed_policy_wins = fixed_policy_wins.sort_values().tolist()


agent_wins = winner_series[winner_series == agent].index
agent_win = [1 if i in agent_wins else 0 for i in range(1, 1501)]
cumulative_wins_A = pd.Series(agent_win).cumsum()
win_rate_A = cumulative_wins_A / range(1, 1501)
plt.plot(range(1, 1501), win_rate_A, marker="o", label="Deep Q-Lambda Agent")   

fixed_policy_win = [1 if i in fixed_policy_wins else 0 for i in range(1, 1501)]
cumulative_wins_FP = pd.Series(fixed_policy_win).cumsum()
cumulative_wins_FP_average = cumulative_wins_FP / 3
win_rate_FP_average = cumulative_wins_FP_average / range(1, 1501)
plt.plot(range(1, 1501), win_rate_FP_average, marker="o", label="Fixed Policy Agent Average")

plt.xlabel("Game Number")
plt.ylabel("Cumulative Win Rate")
plt.title("Cumulative Win Rate While Training the Deep Q-Lambda Agent")
plt.legend()
plt.show()

