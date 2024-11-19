import math
import random
from classes.state import State
class Q_learning_agent:
    '''
    Implements the qlearning algorithm by updating the qvalues from the qtable
    '''

    def __init__(self, actions):
        self.qTable = {}
        self.actions = actions
        self.alpha = 0.2; 
        self.gamma = 0.95
        self.epsilon = 0.1  # Exploration rate

    def getQValue(self, state, action):
        '''
        gets the qvalues from the qtable
        '''

        # if (self.qTable[state + action]):
        #     return self.qTable[state + action]
        # return 0.0
        state_action = (tuple(state), action)
        return self.qTable.get(state_action, 0.0)

    def updateQValue(self, state, action, reward, nextState):
        '''
        updates the qtable by adding better qvalues depending on the most recent move, 
        if it had a better reward than the previous one
        '''
        
        bestNextQ = 0
        for act in self.actions:
            if self.getQValue (nextState, act) > bestNextQ:
                bestNextQ = self.getQValue (nextState, act)
        self.qTable[(state, action)] = self.getQValue(state, action) + self.alpha * (reward + self.gamma * bestNextQ - self.getQValue(state, action))

        # best_next_q = max(self.getQValue(nextState, a) for a in range(len(self.actions)))
        # current_q = self.getQValue(state, action)
        # self.qTable[(state, action)] = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
    
    def choose_action(self, state):

        ''' 
        choose the best action depending on the values in the qtable at the given state

        TODO: Return an index between 0 and 83
        '''

        best_action = self.actions[0]
        for action in self.actions[1:]:
            if self.getQValue(state, action) > self.getQValue(state, best_action):
                best_action = action
        return best_action

        # best_action = max(range(len(self.actions)), key=lambda a: self.getQValue(state, a))
        # return best_action
    
    def take_turn(self, action_obj, player, board, state):
        """
        Executes the agent's turn: chooses an action, maps it, performs it, and updates Q-values.
        """
        # Agent selects an action index (0 to 83) for the 1x84 vector of actions
        action_idx = self.choose_action(state)

        property_idx, action_type = action_obj.map_action_index(action_idx)
        
        action_obj.execute_action(player, board, property_idx, action_type)

        reward = self.get_reward(player)  # Make sure this method exists already right now or pass to the agent

        # Get the next state after action
        next_state_instance = State(player, board.players)
        next_state = next_state_instance.state

        # Update Q-table
        self.updateQValue(state, action_idx, reward, next_state)

        # Return the new state to the game logic
        return next_state

