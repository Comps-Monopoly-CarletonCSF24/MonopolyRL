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
cumulative_wins = win_counts.cumsum()

# Plot the cumulative wins over game numbers
plt.figure(figsize=(10, 5))

fixed_policy_players = [player for player in all_players if "Fixed Policy" in player]
agent = [player for player in all_players if "Fixed Policy"  not in player][0]
fixed_policy_wins = pd.Index([])
for player in fixed_policy_players:
    fixed_policy_wins = fixed_policy_wins.union(winner_series[winner_series == player].index)

fixed_policy_average = fixed_policy_wins[::3]
plt.plot(fixed_policy_average, range(1, len(fixed_policy_average) + 1), marker="o", label="Fixed Policy Average")
 
agent_wins = winner_series[winner_series == agent].index
plt.plot(agent_wins, range(1, len(agent_wins) + 1), marker="o", label= "Deep Q-Lambda Agent")

plt.xlabel("Game Number")
plt.ylabel("Total Wins")
plt.title("Cumulative Wins While Training the Deep Q-Lambda Agent")
plt.legend()
plt.show()
