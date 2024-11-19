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

Here are the functions that we need to write:

- class trace:
  a dictionary of (state,action) -> value
  
  From Chat Sensei: Meaning of the Value in Eligibility Traces:
  - Higher Trace Values: A higher value indicates that the state-action pair is more "eligible" to be updated based on future rewards. Essentially, this means that the agent considers the state-action pair to be more responsible for achieving the current reward, and its Q-value will be updated more strongly.
  - Lower Trace Values: A lower value indicates that the state-action pair is less eligible to be updated. In this case, the agent considers that past state-action pair to be less influential in the current reward, and therefore its Q-value update will be weaker.
  - Using eligibility traces in Q-learning offers several benefits that can significantly improve the learning efficiency and convergence rate of the agent.

- Q table: **Remove**. Q values are now stored in the NN
  
- update_trace(trace, state, action, reward):
  modify the trace table according to [Incremental multi-step Q-learning](https://link.springer.com/article/10.1007/BF00114731):
![image](https://github.com/user-attachments/assets/c876b13d-1182-458a-8a7b-cecd84b973c7)

- in choose_action(state)
    - if the state is similar enough to a current state, use the current Q table
    - else, send (state, action) for all (possible) actions through the nn, find best.

- state.is_similar(self, other)
  
- is_possible_action(index)
    check if an action index is possible
  
- in angent.take_turn (this needs to be renamed make_a_move when we convert agent to a player instance):
    - roll
    - get current state
    - choose_action
    - execute_action
    - get reward
    - get updated Q value
    - update Trace table
    - train NN with the new Q


    
