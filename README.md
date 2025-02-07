# MonopolyRL
Computer Science comps project at Carleton College (Fall and Winter 24') by Albert Jing, Dake Peng, Paul Claudel Izabayo, and Xiaoying Qu

This project uses code from the following github repositories:
- The Python Monopoly engine, as well as the Fixed policy logistics, are borrowed from gamescomputersplay/monopoly.
- The JavaScript (web runtime) Monopoly Eengine is borrowed from intrepidcoder/monopoly.
- The Q-Lambda Agent (SARSA) code is borrowed, translated, and reformatted from pmpailis/rl-monopoly.

## How to Use

1. Edit settings.py to set the parameters you want for your simulation.
2. Run monopoly_simulator.py.
3. Marvel at the results.

## Implemented Rules

The rules in this simulation are based on Hasbro's official manual for playing Monopoly, with the potential for tweaking parameters here and there to see how they affect the game's results. Some of the more complex rules are still a "Work In Progress"; see the TODO section for details.

## Player Behavior

Players in the simulation follow the most common-sense logic to play, which is:
- Buy whatever you land on.
- Build at the first opportunity.
- Unmortgage property as soon as possible.
- Get out of jail on doubles; do not pay the fine until you have to.
- Maintain a certain cash threshold below which the player won't buy, improve, or unmortgage property.
- Trade 1-on-1 with the goal of completing the player's monopoly. Players who give cheaper property should provide compensation equal to the difference in the official price. Don't agree to a trade if the properties are too unequal.