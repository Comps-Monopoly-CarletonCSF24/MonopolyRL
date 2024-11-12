import math
import random
from game import get_alive_players
class Q_learning_agent:
    '''
    Implements the qlearning algorithm by updating the qvalues from the qtable
    '''

    def __init__(self, actions):
        self.qTable = {}
        self.actions = actions
        self.alpha = 0.2; 
        self.gamma = 0.95

    def getQValue(self, state, action):
        '''
        gets the qvalues from the qtable
        '''

        if (self.qTable[state + action]):
            return self.qTable[state + action]
        return 0.0

    def updateQValue(self, state, action, reward, nextState):
        '''
        updates the qtable by adding better qvalues depending on the most recent move, 
        if it had a better reward than the previous one
        '''
        
        bestNextQ = 0
        for act in self.actions:
            if self.getQValue (nextState, act) > bestNextQ:
                bestNextQ = self.getQValue (nextState, act)
        self.qTable[action+state] = self.getQValue(state, action) + self.alpha * (reward + self.gamma * bestNextQ - self.getQValue(state, action))
    
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
                if self.get_q_value(state, action) > self.get_q_value(state, best_action):
                    best_action = action
            return best_action
    
    def get_reward (self, player):

        ''' 
        computes the reward accounting for the player's networth in comparison with their opponents money
        '''

        player_newtworth = player.neworth()
        alive_players = get_alive_players()

        all_players_worth = 0
        for player in alive_players:
            all_players_worth += player.networth()

        p = 0 # number of playters
        c = 0 # smothing factor 
        v = player_newtworth - all_players_worth # players total assets values (add up value of all properties in the possession of the player minus the properties of all his opponents)
        m = (player_newtworth/all_players_worth) * 100 # player's finance (percentage of the money the player has to the sum of all the players money)
        r = ((v/p)*c)/ (1+ abs((v/p)*c)-(1/p)*m)
        return r