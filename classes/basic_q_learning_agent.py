import math
import random
from classes.rewards import get_reward
from classes.state import State
from classes.player import Player

class Q_learning_agent(Player):
    '''
    Implements the qlearning algorithm by updating the qvalues from the qtable.
    Inherits from Player class
    '''

    def __init__(self, name, settings, position=0, money=1500):
        super().__init__(name,settings)
        self.settings = settings
        self.name = name
        self.actions =[] #using a list to store actions
        self.qTable = {}
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
        Choose the best action depending on the values in the qtable at the given state

        Return an index between 0 and 83
        '''
        
        best_action_idx = 0 #initialize best action index
        best_q_value = self.getQValue(state, best_action_idx)

        for action_idx in range(1,len(self.actions) *28):
            q_value = self.getQValue(state,action_idx)
            if q_value > best_q_value:
                best_action_idx = action_idx
                best_q_value = q_value
                
        return best_action_idx

    def take_turn(self, action_obj, current_player, board, players, state): 
        """
        Executes the agent's turn: chooses an action, maps it, performs it, and updates Q-values.
        Since the list of players alive are managed at game level, added players as a parameter.
        """
        # Agent selects an action index (0 to 83) for the 1x84 vector of actions
        action_idx = self.choose_action(state)

        property_idx, action_type = action_obj.map_action_index(action_idx)
        
        action_obj.execute_action(current_player, board, property_idx, action_type)

        reward = get_reward(current_player, players)  #method moved to rewards.py to aviod circular import
        # Get the next state after action
        next_state_instance = State(current_player, players)
        next_state = tuple(next_state_instance.state) 
        '''convert to tuple, we need the tuple so that it can be used as dictionary
        keys for Q-tables
        '''

        # Update Q-table
        self.updateQValue(state, action_idx, reward, next_state)

        # Return the new state to the game logic
        return next_state
    