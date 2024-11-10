import math
import random

class Q_learning_agent:
    def __init__(self, actions):
        self.qTable = {}
        self.actions = actions
        self.alpha = 0.2; 
        self.gamma = 0.95
    def getQValue(self, state, action):
        if (self.qTable[state + action]):
            return self.qTable[state + action]
        return 0.0

    def updateQValue(self, state, action, reward, nextState):
        bestNextQ = 0
        for act in self.actions:
            if self.getQValue (nextState, act) > bestNextQ:
                bestNextQ = self.getQValue (nextState, act)
        self.qTable[action+state] = self.getQValue(state, action) + self.alpha * (reward + self.gamma * bestNextQ - self.getQValue(state, action))
    
    
    
    def choose_action(self, state):

        # we do not need to have this one here necessarily
        # # Explore (random action) or exploit (best action based on Q-value)
        # if random.random() < self.epsilon:
        #     # Explore: choose a random action
        #     return random.choice(self.actions)
        # else:
            # Exploit: choose the action with the highest Q-value
            best_action = self.actions[0]
            for action in self.actions[1:]:
                if self.get_q_value(state, action) > self.get_q_value(state, best_action):
                    best_action = action
            return best_action
