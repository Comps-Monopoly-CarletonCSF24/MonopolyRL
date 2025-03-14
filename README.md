# MonopolyRL
Computer Science comps (senior thesis) project at Carleton College (Fall 24' and Winter 25') by Albert Jing, Dake Peng, Paul Claudel Izabayo, and Xiaoying Qu

This project uses code from the following resources:
- The Python Monopoly engine was borrowed from [gamescomputersplay/monopoly](https://github.com/gamescomputersplay/monopoly).
- The JavaScript (web runtime) Monopoly Eengine is borrowed from [intrepidcoder/monopoly](https://github.com/intrepidcoder/monopoly).
- The Deep Q-Lambda Agent code references [pmpailis/rl-monopoly](https://github.com/pmpailis/rl-monopoly)
- The construction of the state-actions spaces, the reward function, and the update process of the Deep Q-Lambda Agent references *Bailis, P., Fachantidis, A., & Vlahavas, I. (2014). Learning to play monopoly: A reinforcement learning approach. In Proceedings of the 50th Anniversary Convention of The Society for the Study of Artificial Intelligence and Simulation of Behaviour. AISB.*

Try playing against our Deep Q-Lambda Agent at [https://comps-monopoly-carletoncsf24.github.io/MonopolyRL/JavaScript%20Engine/](https://comps-monopoly-carletoncsf24.github.io/MonopolyRL/JavaScript%20Engine/)

## Prerequisites

install the following pacakages:

torch, onnx, tqdm, pandas, matplotlib [Add More]

## Training the Deep Q-Lambda Agent

To run the training session to train the QLambdaAgent:
1. in classes/DQAgent : delete model_parameters.pth. This will create a new .pth file with random weights.
2. in settings.py: set players_list to 1 QLambdaAgent (Experiment Player) and 3 FixedPolicyPlayers (Standard Players)
3. in settings.py: confirm participates_in_trades = True under StandardPlayer
4. in settings.py: confirm n_games_per_batch and n_batches
5. run train_and_evaluate_DQAgent.py

Train_and_evaluateDQAgent.py will then train the model through multiple batches, and logs the agent's learning progression for game data. For each iteration it calls monopoly_game in game.py which in turn runs the game. 

DQAgent.py is the neural network implementation (150 nodes with one hidden layer, backpropagating using SGD implemented by PyTorch), concatenates a state and action pair into a tensor to be taken as input for the neural network. The NN outputs a single q value, after looking through q values for all actions in a given state and choosing the highest or exploring a random action, and then subsequently giving that q value. Maintains a trace list managed by lambda decay parameter, and updates the trace values after every action. Uses trace values to update q values and saves it in a training batch that is used to train the neural network after every round (dice roll). Player.py's DQAgent class integrates specific game logic for the neural network, and manages the buy, sell, or do nothing actions in addition to the default player_logistics game functions. It also calls the train neural network function every time a player takes a turn. 

## Implemented Rules (from gamescomputersplay/monopoly)

The rules in this simulation are based on Hasbro's official manual for playing Monopoly, with the potential for tweaking parameters here and there to see how they affect the game's results. Some of the more complex rules are still a "Work In Progress";

## Fixed Policy Player Behavior (from gamescomputersplay/monopoly)

Players in the simulation follow the most common-sense logic to play, which is:
- Buy whatever you land on.
- Build at the first opportunity.
- Unmortgage property as soon as possible.
- Get out of jail on doubles; do not pay the fine until you have to.
- Maintain a certain cash threshold below which the player won't buy, improve, or unmortgage property.
- Trade 1-on-1 with the goal of completing the player's monopoly. Players who give cheaper property should provide compensation equal to the difference in the official price. Don't agree to a trade if the properties are too unequal.
