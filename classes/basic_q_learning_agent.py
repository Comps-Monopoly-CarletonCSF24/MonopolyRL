import math
import random
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
        '''

        # we do not need to have this one here necessarily
        # # Explore (random action) or exploit (best action based on Q-value)
        # if random.random() < self.epsilon:
        #     # Explore: choose a random action
        #     return random.choice(self.actions)
        # else:
            # Exploit: choose the action with the highest Q-value

        best_action = self.actions[0]
        for action in self.actions[1:]:
            if self.getQValue(state, action) > self.getQValue(state, best_action):
                best_action = action
        return best_action

        # best_action = max(range(len(self.actions)), key=lambda a: self.getQValue(state, a))
        # return best_action
    

