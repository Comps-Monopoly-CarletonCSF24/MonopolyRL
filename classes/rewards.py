from classes.player import Player

'''This file is created to aviod duplications in using some functions in players and AI agents
'''
def get_alive_players(players):
    '''
    creates a list of all the alive players
    '''

    alive_players = []
    for  player in players:
        if player.net_worth() > 0:
            alive_players.append(player)

    return alive_players


def get_reward(player,players):
    '''
    compute the reward accountinf for the player's networth in compariosn with their opponents money
    '''

    player_networth = player.net_worth()
    alive_players = get_alive_players(players)
    
    all_players_worth =0
    for player in alive_players:
        all_players_worth += player.net_worth()
    
    p = 4 # number of players
    c = 0.450 # smothing factor 
    v = player_networth - all_players_worth # players total assets values (add up value of all properties in the possession of the player minus the properties of all his opponents)
    m = (player_networth/all_players_worth) * 100 # player's finance (percentage of the money the player has to the sum of all the players money)
    r = ((v/p)*c)/ (1+ abs((v/p)*c)-(1/p)*m)
    return r