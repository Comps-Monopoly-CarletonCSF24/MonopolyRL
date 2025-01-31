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
  - qnn: save the qnn to a file. (instead of the qtable)

- update_trace(trace, state, action, reward):
  modify the trace table according to [Incremental multi-step Q-learning](https://link.springer.com/article/10.1007/BF00114731):
![image](https://github.com/user-attachments/assets/c876b13d-1182-458a-8a7b-cecd84b973c7)

- in choose_action(state)
    - if the state is similar enough to a current state, run the nn.
    - else, send (state, action) for all (possible) actions through the nn, find best.

- state.is_similar(self, other)
  
- is_possible_action(index)
    check if an action index is possible
  
- in angent.take_turn (this needs to be renamed make_a_move when we convert agent to a player instance):
    - roll
    - get current state
    - get current reward
    - update previous trace
    - train NN with trace
    - train NN with new Q
    - choose_action
    - execute_action

Note: the short paper chose all actions in a loop until no possible actions. Then performed all of them.

- update_trace(trace, state, action, reward):
    - for each trace:
        - if similar to (state, action): set trace[state, action] to 1
        - if similar state but different action: remove, 
        - if different state and different action: update trace value with decay function
    - add new (state,action) if not (state action) exists, set value to 1

- train_neural_network_with_trace(trace, state, action, reward)
  *state,action mean the state and action of last round*
    - for each trace:
        - if similar to (state, action): continue
        - else: train nn
            - q_t = nn(trace.state, trace.action)
            - max_qt = nn(trace.state, choose_action(trace.state))
            - max_q = nn(state, choose_action(state)
            - q updated = nn(trace.state, newaction), where newaction is the result of choose_action(trace.state)
            - qvalue = q_t  + alpha * (traces[i].value) * (reward + gamma * max_qt - max_q);
            - train_neural_network(trace.state, trace.action, qvalue)

TODO:
- Normalize action to 0-1 or -1-1
- after each game, reduce values in trace[]
- endgame rewards for winner and loser
## 1/31/2025
### Connecting to JavaScript Agent
Replicate State, Action 
ApproxQ: Save weights & biases, and then call in JS - Paul
Basic Q: Save Q table, call in JS - Xiaoying
NN: Import .pth in JS, run nn for each step (replicate take_one_step()): Albert
Confirming that the NN is training, plot win rate growth: Dake
Replicate Player_Logistics, Replicate each player class