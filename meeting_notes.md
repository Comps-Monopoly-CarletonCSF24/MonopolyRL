# Meeting Notes
Keeps track of what we need to do and what we know.

## 11/14/2024
- Go over the code that Paul and Albert added
    - Do we need to use an NN? 
    - Why does tha agent only buy?
    - What exactly needs to happen in agent_turn?
- Find a place for the BQLA code to stay. Currently, they are in game.py, state.py, action.py, basic_q_learning_agent.py. We need to think about this in the big picture, since we are not only building 1 agent.
- Let's modularize our tests! i.e. make separate programs when you are testing/debugging.

TODO:

- Get NN Started
- Change Q learning table with action indices
- Change BQLA to become an instance of Player
- Make a test environment for the changes (by adding BQLA in players[])
- Finish agent_turn!! (move it to BQLA?)

## 11/18/2024
Notes from Dake on qnn.py

the NN that the short paper inplemented uses 3 layers
- the 1st is 24 nodes (linear), with 23 state and 1 action
- the 2nd is 150 nodes (sigmoid)
- the 3rd is 1 node, representing the Q value

the pytorch inplementation uses state as input and action as output, 
with Q values implicitly stored in the network. DECIDE WHICH ONE TO USE.

Here are the functions that I think should appear in this program:
update_trace():
