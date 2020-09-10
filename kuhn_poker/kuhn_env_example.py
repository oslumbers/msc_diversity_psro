import gym
import gym_kuhn_poker
import numpy as np



env = gym.make('KuhnPoker-v0', **dict()) # optional secondary argument

''' 
    The format of the observations is as follows:
    obs = [obs_player1, obs_player2]
    obs_player1 = [0, 1,      0, 0, 1,       1, 0, 0,    1, 0, 0,   1, 0, 0,   1, 0, 0,      1, 1]
     - First two digits is player: [0,1] player 1; [1, 0] player 2
     - Next three is the card [0, 0, 1] or [0, 1, 0] or [1, 0, 0]
     - Next 4x3 digits is what happened in each turn:
         - [1, 0, 0] means the turn has not been played
         - [0, 1, 0] means pass 
         - [0, 0, 1] means bet
         - There are four groups of 3 digits: 
                - First group corresponds to what player 1 does
                - Second group corresponds to what player 2 does IF player 1 passes in first round
                - Third group corresponds to what player 1 does IF player 1 passes in first round
                - Fourth group corresponds to what player 2 does IF player 1 bets in first round
     - Last two digits correspond to the contribution to the pot of each player [1, 1], [2, 1], [1, 2], [2, 2]

'''

#These are all possible states
steps = 10000
obs = env.reset()
states1 = [obs[0]]
states2 = [obs[1]]
for step in range(steps):
    a = np.random.randint(2)
    obs, r, d, _ = env.step(a)
    states1.append(obs[0])
    states2.append(obs[1])

    if d:
        obs = env.reset()
        states1.append(obs[0])
        states2.append(obs[1])

states1 = np.array(states1)
states2 = np.array(states2)
states1 = np.unique(states1, axis=0)  # Player 1
states2 = np.unique(states2, axis=0)  # Player 2

# Visualise states1[:9, 5:] to see all possible games (regardless of card or player)

# These are all possible states where players have to take an action
steps = 10000
obs = env.reset()
states1 = [obs[0]]
states2 = []
player_turn = 0
for step in range(steps):
    a = np.random.randint(2)
    obs, r, d, _ = env.step(a)
    player_turn = 1-player_turn
    if not d and player_turn==0:
        states1.append(obs[0])
    elif not d and player_turn == 1:
        states2.append(obs[1])

    if d:
        obs = env.reset()
        player_turn = 0
        states1.append(obs[0])

states1 = np.array(states1)
states2 = np.array(states2)
states1 = np.unique(states1, axis=0)  # Player 1
states2 = np.unique(states2, axis=0)  # Player 2


aa = 1






















