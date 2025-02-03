import matplotlib.pyplot as plt
import pickle

def plot_q_values_from_file():
    """Load and plot Q-values after the game."""
    try:
        with open("q_values.pkl", "rb") as f:
            q_values = pickle.load(f)

        plt.plot(q_values)
        plt.xlabel("Update Step")
        plt.ylabel("Q-value")
        plt.title("Q-value Stabilization Over Time")
        plt.show()
    except FileNotFoundError:
        print("No Q-values found. Run training first!")

plot_q_values_from_file()