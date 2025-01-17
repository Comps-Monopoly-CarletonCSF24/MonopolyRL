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

'''
def get_reward(player,players):
    """
    compute the reward accountinf for the player's networth in compariosn with their opponents money
    """

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
'''
def get_reward(player,players):
    """A get reward function that explicitly encourages buying properties"""
    reward =0
    #base reward for properties owned
    num_properties = len(player.owned)
    reward += num_properties * 10   
    '''
    #bonus for monopolies over a color
    # colors: brown. lightblue, pink, orange, red, yellow, green, indigo
    color_count = 
    for prop in player.owned:
        prev = prop.group
        if prop.group 
        reward += 50
    '''
    #reward for having houses/hotels
    for prop in player.owned:
        if hasattr(prop, 'num_houses'):
            reward += prop.num_houses * 15
    
    #Penalty for being close to bankrupt
    if player.money < 100:
        reward -= 50

    return reward
    
