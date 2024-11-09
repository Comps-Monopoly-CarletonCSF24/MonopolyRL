import Math

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
        self.qTable[action+state] = self.getQValue(state, action) +
            self.alpha * (reward + self.gamma * bestNextQ - self.getQValue(state, action))
    
    
    def chooseAction (self, state):
        return 